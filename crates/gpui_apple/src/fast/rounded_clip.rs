//! Evaluate independent rounded padding edges directly in colour fragments.
//! Nested scopes append geometry and siblings restore the enclosing stack.
//! Only actual path batches restart the colour pass; rounded containers do
//! not allocate textures or encode mask render passes.

use std::ptr::NonNull;

use anyhow::{Context as _, Result};
use gpui::{AtlasTextureId, DevicePixels, LayerFrame, PrimitiveBatch, Scene, SceneBatch, Size};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{
    MTLClearColor, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLDevice,
    MTLRenderCommandEncoder, MTLResourceOptions, MTLTexture,
};

use crate::metal_renderer::{
    InstanceBindings, InstanceBufferWriter, MetalRenderer, SurfaceTextures,
    new_command_encoder_for_texture, write_instances,
};

type Encoder = ProtocolObject<dyn MTLRenderCommandEncoder>;

const CLIP_TEXTURE_BASE: u32 = 0xF100_0000;

/// Geometry is copied into Metal's command stream, never retained in a shared
/// mutable texture. A stack remains valid when sibling scopes overlap.
#[repr(C)]
#[derive(Clone, Copy)]
struct Clip {
    bounds: [f32; 4],
    horizontal: [f32; 4],
    vertical: [f32; 4],
}

fn group(scene: &Scene, texture: AtlasTextureId) -> Option<&LayerFrame> {
    let key = texture.index.checked_sub(0xF000_0000)?;
    if texture.index < CLIP_TEXTURE_BASE {
        return None;
    }
    scene
        .layers
        .frames
        .iter()
        .find(|frame| frame.key.0 == key && frame.tile_size == 0)
}

pub(crate) fn has_groups(scene: &Scene) -> bool {
    scene
        .polychrome_sprites
        .iter()
        .any(|sprite| group(scene, sprite.tile.texture_id).is_some())
}

/// Whether drawing `scene` rasterizes a path: one of its own, or one of a
/// scene it clips.
pub(crate) fn draws_paths(scene: &Scene) -> bool {
    !scene.paths.is_empty()
        || scene.polychrome_sprites.iter().any(|sprite| {
            group(scene, sprite.tile.texture_id)
                .and_then(|frame| frame.content.scene())
                .is_some_and(draws_paths)
        })
}

pub(crate) fn default_mask(encoder: &Encoder) {
    let count = 0u32;
    let empty = [0f32; 12];
    // SAFETY: Both values are copied from live locals of the lengths given, into the slots the
    // fragment shaders read the clip count and the clip stack from.
    unsafe {
        encoder.setFragmentBytes_length_atIndex(NonNull::from(&count).cast(), 4, 29);
        encoder.setFragmentBytes_length_atIndex(NonNull::from(&empty).cast(), 48, 28);
    }
}

fn bind(encoder: &Encoder, clips: &[Clip], device: &ProtocolObject<dyn MTLDevice>) -> Result<()> {
    let count = clips.len() as u32;
    // SAFETY: The count is copied from a live local into the clip count slot.
    unsafe {
        encoder.setFragmentBytes_length_atIndex(NonNull::from(&count).cast(), 4, 29);
    }
    if !clips.is_empty() {
        let length = std::mem::size_of_val(clips);
        let bytes = NonNull::from(clips).cast();
        if length <= 4096 {
            // SAFETY: The clips are copied from a live slice of `length` bytes.
            unsafe {
                encoder.setFragmentBytes_length_atIndex(bytes, length, 28);
            }
        } else {
            // SAFETY: Metal copies `length` bytes from the live slice into the new buffer.
            let buffer = unsafe {
                device.newBufferWithBytes_length_options(
                    bytes,
                    length,
                    MTLResourceOptions::StorageModeShared,
                )
            }
            .context("allocating rounded clip buffer")?;
            // SAFETY: The buffer holds the clip stack the fragment shaders read from slot 28,
            // and Metal retains it for the pass.
            unsafe {
                encoder.setFragmentBuffer_offset_atIndex(Some(&buffer), 0, 28);
            }
        }
    }
    Ok(())
}

pub(crate) fn raster_nested_tiles(renderer: &mut MetalRenderer, scene: &Scene) {
    for sprite in &scene.polychrome_sprites {
        if let Some(frame) = group(scene, sprite.tile.texture_id)
            && let Some(child) = frame.content.scene()
        {
            crate::fast::layers::raster::rasterize_tiles(renderer, child);
        }
    }
}

pub(crate) fn draw(
    renderer: &mut MetalRenderer,
    scene: &Scene,
    bindings: &InstanceBindings,
    writer: &mut InstanceBufferWriter,
    texture: &ProtocolObject<dyn MTLTexture>,
    viewport: Size<DevicePixels>,
    surface_textures: &mut SurfaceTextures,
) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
    let queue = renderer.command_queue.clone();
    let command_buffer = queue.commandBuffer().context("creating command buffer")?;
    let alpha = if renderer.opaque { 1. } else { 0. };
    let encoder = new_command_encoder_for_texture(
        &command_buffer,
        texture,
        viewport,
        Some(MTLClearColor {
            red: 0.,
            green: 0.,
            blue: 0.,
            alpha,
        }),
    )?;
    encode(
        renderer,
        scene,
        bindings,
        writer,
        texture,
        viewport,
        &command_buffer,
        &mut Vec::new(),
        encoder,
        surface_textures,
    )?
    .endEncoding();
    Ok(command_buffer)
}

pub(crate) fn draw_tile(
    renderer: &mut MetalRenderer,
    scene: &Scene,
    writer: &mut InstanceBufferWriter,
    texture: &ProtocolObject<dyn MTLTexture>,
    viewport: Size<DevicePixels>,
    commands: &ProtocolObject<dyn MTLCommandBuffer>,
    background: MTLClearColor,
    surface_textures: &mut SurfaceTextures,
) -> Result<()> {
    let old_texture = renderer.path_intermediate_texture.take();
    let old_msaa = renderer.path_intermediate_msaa_texture.take();
    let result = (|| {
        renderer.update_path_intermediate_textures(viewport)?;
        let bindings = write_instances(scene, writer)?;
        let encoder =
            new_command_encoder_for_texture(commands, texture, viewport, Some(background))?;
        encode(
            renderer,
            scene,
            &bindings,
            writer,
            texture,
            viewport,
            commands,
            &mut Vec::new(),
            encoder,
            surface_textures,
        )?
        .endEncoding();
        Ok(())
    })();
    renderer.path_intermediate_texture = old_texture;
    renderer.path_intermediate_msaa_texture = old_msaa;
    result
}

#[allow(clippy::too_many_arguments)]
fn encode(
    renderer: &mut MetalRenderer,
    scene: &Scene,
    bindings: &InstanceBindings,
    writer: &mut InstanceBufferWriter,
    texture: &ProtocolObject<dyn MTLTexture>,
    viewport: Size<DevicePixels>,
    commands: &ProtocolObject<dyn MTLCommandBuffer>,
    clips: &mut Vec<Clip>,
    mut encoder: Retained<Encoder>,
    surface_textures: &mut SurfaceTextures,
) -> Result<Retained<Encoder>> {
    renderer.fast_layers.note_composited(scene);
    for batch in scene.scene_batches() {
        let batch = match batch {
            SceneBatch::Primitives(batch) => batch,
            SceneBatch::Effect(effect_index) => {
                encoder = renderer.encode_effect(
                    scene,
                    effect_index,
                    writer,
                    commands,
                    texture,
                    viewport,
                    &encoder,
                    surface_textures,
                )?;
                continue;
            }
        };
        bind(&encoder, clips, &renderer.device)?;
        match batch {
            PrimitiveBatch::Shadows(range) => {
                renderer.draw_shadows(range, bindings, viewport, &encoder)
            }
            PrimitiveBatch::Quads(range) => {
                renderer.draw_quads(range, bindings, viewport, &encoder)
            }
            PrimitiveBatch::Underlines(range) => {
                renderer.draw_underlines(range, bindings, viewport, &encoder)
            }
            PrimitiveBatch::MonochromeSprites { texture_id, range } => {
                renderer.draw_monochrome_sprites(texture_id, range, bindings, viewport, &encoder)
            }
            PrimitiveBatch::PolychromeSprites { texture_id, range } => {
                if let Some(frame) = group(scene, texture_id) {
                    let child = frame.content.scene().context("rounded clip content")?;
                    let radii = child
                        .layers
                        .frames
                        .iter()
                        .find(|frame| frame.key.0 == u32::MAX)
                        .context("rounded clip geometry")?
                        .background;
                    for index in range {
                        let sprite = &scene.polychrome_sprites[index];
                        let b = sprite.bounds;
                        let r = sprite.corner_radii;
                        clips.push(Clip {
                            bounds: [b.origin.x.0, b.origin.y.0, b.right().0, b.bottom().0],
                            horizontal: [
                                r.top_left.0,
                                r.top_right.0,
                                r.bottom_right.0,
                                r.bottom_left.0,
                            ],
                            vertical: [radii.r, radii.g, radii.b, radii.a],
                        });
                        let child_bindings = write_instances(child, writer)?;
                        encoder = encode(
                            renderer,
                            child,
                            &child_bindings,
                            writer,
                            texture,
                            viewport,
                            commands,
                            clips,
                            encoder,
                            surface_textures,
                        )?;
                        clips.pop();
                        renderer.fast_layers.note_composited(scene);
                    }
                } else {
                    renderer
                        .draw_polychrome_sprites(texture_id, range, bindings, viewport, &encoder);
                }
            }
            PrimitiveBatch::Paths(range) => {
                encoder.endEncoding();
                let paths = &scene.paths[range];
                let path_textures = renderer.window_path_textures();
                let drawn = renderer.draw_paths_to_intermediate(
                    paths,
                    writer,
                    viewport,
                    commands,
                    &path_textures,
                )?;
                encoder = new_command_encoder_for_texture(commands, texture, viewport, None)?;
                bind(&encoder, clips, &renderer.device)?;
                if drawn {
                    renderer.draw_paths_from_intermediate(
                        paths,
                        writer,
                        viewport,
                        &encoder,
                        &path_textures,
                    )?;
                }
            }
            PrimitiveBatch::Surfaces(range) => renderer.draw_surfaces(
                &scene.surfaces[range.clone()],
                range.start,
                bindings,
                viewport,
                &encoder,
                surface_textures,
            )?,
            PrimitiveBatch::SubpixelSprites { .. } => unreachable!("Metal uses grayscale glyphs"),
        }
    }
    Ok(encoder)
}
