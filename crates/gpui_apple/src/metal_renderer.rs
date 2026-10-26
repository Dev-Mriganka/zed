use crate::metal_atlas::MetalAtlas;
use anyhow::{Context as _, Result};
use block2::RcBlock;
use gpui::{
    AtlasTextureId, Background, Bounds, ContentMask, DevicePixels, PaintEffect, PaintSurface, Path,
    Point, PrimitiveBatch, ScaledPixels, Scene, SceneBatch, Size, point, size,
};
#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
use image::RgbaImage;
use objc2::{Message, rc::Retained, runtime::ProtocolObject};
use objc2_core_foundation::{CFRetained, CGSize};
use objc2_core_graphics::{CGColorSpace, kCGColorSpaceDisplayP3};
use objc2_core_video::{
    CVMetalTexture, CVMetalTextureCache, CVMetalTextureGetTexture, CVPixelBufferGetHeight,
    CVPixelBufferGetHeightOfPlane, CVPixelBufferGetPixelFormatType, CVPixelBufferGetWidth,
    CVPixelBufferGetWidthOfPlane, kCVPixelFormatType_420YpCbCr8BiPlanarFullRange, kCVReturnSuccess,
};
use objc2_foundation::{NSRange, NSString};
#[cfg(target_os = "macos")]
use objc2_metal::MTLCopyAllDevices;
use objc2_metal::{
    MTLBlendFactor, MTLBlendOperation, MTLBuffer, MTLClearColor, MTLCommandBuffer,
    MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLDrawable, MTLFunction, MTLGPUFamily, MTLLibrary, MTLLoadAction, MTLPixelFormat,
    MTLPrimitiveType, MTLRenderCommandEncoder, MTLRenderPassDescriptor,
    MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLResourceOptions, MTLSamplerAddressMode,
    MTLSamplerDescriptor, MTLSamplerMinMagFilter, MTLSamplerState, MTLStorageMode, MTLStoreAction,
    MTLTexture, MTLTextureDescriptor, MTLTextureType, MTLTextureUsage, MTLViewport,
};
#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
use objc2_metal::{MTLBlitCommandEncoder, MTLOrigin, MTLRegion, MTLSize};
use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};
use parking_lot::Mutex;

use std::{
    cell::Cell, collections::HashMap, iter, mem, mem::MaybeUninit, ops::Range, ptr, slice,
    sync::Arc,
};

// Exported to metal
pub(crate) type PointF = gpui::Point<f32>;

#[cfg(not(feature = "runtime_shaders"))]
const SHADERS_METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders.metallib"));
#[cfg(feature = "runtime_shaders")]
const SHADERS_SOURCE_FILE: &str = include_str!(concat!(env!("OUT_DIR"), "/stitched_shaders.metal"));
// Use 4x MSAA, all devices support it.
// https://developer.apple.com/documentation/metal/mtldevice/1433355-supportstexturesamplecount
const PATH_SAMPLE_COUNT: u32 = 4;
/// Metal requires the offset a buffer is bound at to be 256-byte aligned.
const INSTANCE_BUFFER_ALIGNMENT: usize = 256;
const MAX_INSTANCE_BUFFER_SIZE: usize = 256 * 1024 * 1024;
/// Bound cached pipelines and remembered failures across config reloads.
const MAX_EFFECT_PIPELINE_CACHE_SIZE: usize = 64;
/// Uniform uploads are padded to this so that an empty block still binds a
/// buffer range Metal accepts.
const MIN_EFFECT_UNIFORM_SIZE: usize = 16;

pub type Context = Arc<Mutex<InstanceBufferPool>>;
pub type Renderer = MetalRenderer;

pub fn new_renderer(
    context: self::Context,
    transparent: bool,
    use_display_p3: bool,
) -> Result<Renderer> {
    MetalRenderer::new(context, transparent, use_display_p3)
}

pub fn new_overlay_renderer(context: self::Context, base: &Renderer) -> Result<Renderer> {
    crate::fast::composition::new_overlay_renderer(context, base)
}

/// The CoreVideo textures a frame's surfaces are drawn from, kept alive until
/// the GPU has read them.
pub(crate) type SurfaceTextures = Vec<CFRetained<CVMetalTexture>>;

pub struct InstanceBufferPool {
    buffer_size: usize,
    buffers: Vec<InstanceBuffer>,
}

impl Default for InstanceBufferPool {
    fn default() -> Self {
        Self {
            buffer_size: 2 * 1024 * 1024,
            buffers: Vec::new(),
        }
    }
}

pub(crate) struct InstanceBuffer {
    metal_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    size: usize,
}

// SAFETY: Each buffer has one CPU writer and returns to the mutex-protected pool only after GPU completion.
unsafe impl Send for InstanceBuffer {}

impl InstanceBufferPool {
    pub(crate) fn reset(&mut self, buffer_size: usize) {
        self.buffer_size = buffer_size;
        self.buffers.clear();
    }

    fn acquire(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        unified_memory: bool,
    ) -> Result<InstanceBuffer> {
        if let Some(buffer) = self.buffers.pop() {
            return Ok(buffer);
        }
        let options = if unified_memory {
            MTLResourceOptions::StorageModeShared
                // Buffers are write only which can benefit from the combined cache
                // https://developer.apple.com/documentation/metal/mtlresourceoptions/cpucachemodewritecombined
                | MTLResourceOptions::CPUCacheModeWriteCombined
        } else {
            MTLResourceOptions::StorageModeManaged
        };
        let buffer = device
            .newBufferWithLength_options(self.buffer_size, options)
            .with_context(|| format!("allocating {}-byte instance buffer", self.buffer_size))?;
        Ok(InstanceBuffer {
            metal_buffer: buffer,
            size: self.buffer_size,
        })
    }

    pub(crate) fn release(&mut self, buffer: InstanceBuffer) {
        if buffer.size == self.buffer_size {
            self.buffers.push(buffer)
        }
    }
}

pub struct MetalRenderer {
    pub(crate) device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub(crate) layer: Option<Retained<CAMetalLayer>>,
    is_apple_gpu: bool,
    pub(crate) is_unified_memory: bool,
    presents_with_transaction: bool,
    /// For headless rendering, tracks whether output should be opaque
    pub(crate) opaque: bool,
    pub(crate) command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub(crate) paths_rasterization_pipeline_state:
        Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    path_sprites_pipeline_state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    shadows_pipeline_state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    quads_pipeline_state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    underlines_pipeline_state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    monochrome_sprites_pipeline_state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    pub(crate) polychrome_sprites_pipeline_state:
        Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    surfaces_pipeline_state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    pub(crate) unit_vertices: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(crate) instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
    pub(crate) sprite_atlas: Arc<MetalAtlas>,
    core_video_texture_cache: CFRetained<CVMetalTextureCache>,
    pub(crate) path_intermediate_texture: Option<Retained<ProtocolObject<dyn MTLTexture>>>,
    pub(crate) path_intermediate_msaa_texture: Option<Retained<ProtocolObject<dyn MTLTexture>>>,
    path_sample_count: u32,
    effect_composite_pipeline_state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    effect_pass_vertex_function: Retained<ProtocolObject<dyn MTLFunction>>,
    effect_sampler: Retained<ProtocolObject<dyn MTLSamplerState>>,
    effect_targets: HashMap<u64, EffectTarget>,
    effect_pipelines: HashMap<Arc<str>, EffectPipeline>,
    effect_frame: u64,
    /// Offscreen render target reused across `render_scene` calls when
    /// rendering headlessly without reading pixels back.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    headless_render_target: Option<Retained<ProtocolObject<dyn MTLTexture>>>,
    pub(crate) fast_layers: crate::fast::layers::TileCache,
}

#[repr(C)]
pub struct PathRasterizationVertex {
    pub xy_position: Point<ScaledPixels>,
    pub st_position: Point<f32>,
    pub color: Background,
    pub bounds: Bounds<ScaledPixels>,
}

impl MetalRenderer {
    /// Creates a new MetalRenderer with a CAMetalLayer for window-based rendering.
    pub fn new(
        instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
        transparent: bool,
        use_display_p3: bool,
    ) -> Result<Self> {
        let device = Self::create_device()?;
        let layer = CAMetalLayer::new();
        Self::configure_layer(&layer, &device, transparent, use_display_p3);
        Self::new_internal(device, Some(layer), !transparent, instance_buffer_pool)
    }

    /// Creates a renderer for a CAMetalLayer owned by a platform view, such as
    /// the backing layer UIKit creates for a view whose `layerClass` is
    /// `CAMetalLayer`. The renderer retains the layer.
    pub fn from_layer(
        instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
        layer: &CAMetalLayer,
        transparent: bool,
    ) -> Result<Self> {
        let device = Self::create_device()?;
        let layer = layer.retain();
        Self::configure_layer(&layer, &device, transparent, false);
        Self::new_internal(device, Some(layer), !transparent, instance_buffer_pool)
    }

    pub(crate) fn configure_layer(
        layer: &CAMetalLayer,
        device: &ProtocolObject<dyn MTLDevice>,
        transparent: bool,
        use_display_p3: bool,
    ) {
        layer.setDevice(Some(device));
        layer.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        if use_display_p3 {
            // SAFETY: The constant is extern, from Core Graphics
            let display_p3_name = unsafe { kCGColorSpaceDisplayP3 };
            if let Some(display_p3_color_space) = CGColorSpace::with_name(Some(display_p3_name)) {
                layer.setColorspace(Some(&display_p3_color_space));
            } else {
                log::error!("unable to create the Display P3 color space");
            }
        }
        // Support direct-to-display rendering if the window is not transparent
        // https://developer.apple.com/documentation/metal/managing-your-game-window-for-metal-in-macos
        layer.setOpaque(!transparent);
        layer.setMaximumDrawableCount(3);
        // Allow texture reading for visual tests and UI automation screenshots
        // (captures without ScreenCaptureKit). Only debug builds pay the
        // presentation cost, even when `test-support` is compiled in.
        #[cfg(all(feature = "test-support", debug_assertions))]
        layer.setFramebufferOnly(false);
        layer.setAllowsNextDrawableTimeout(false);
        layer.setNeedsDisplayOnBoundsChange(true);
        // UIKit sizes a view's backing layer itself; only AppKit-hosted
        // layers need to track their superlayer's bounds.
        #[cfg(target_os = "macos")]
        layer.setAutoresizingMask(
            objc2_quartz_core::CAAutoresizingMask::LayerWidthSizable
                | objc2_quartz_core::CAAutoresizingMask::LayerHeightSizable,
        );
    }

    /// Creates a new headless MetalRenderer for offscreen rendering without a window.
    ///
    /// This renderer can render scenes to images without requiring a CAMetalLayer,
    /// window, or AppKit. Use `render_scene_to_image()` to render scenes.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    pub fn new_headless(instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>) -> Result<Self> {
        let device = Self::create_device()?;
        Self::new_internal(device, None, true, instance_buffer_pool)
    }

    #[cfg(target_os = "macos")]
    fn create_device() -> Result<Retained<ProtocolObject<dyn MTLDevice>>> {
        // Prefer low‐power integrated GPUs on Intel Mac. On Apple
        // Silicon, there is only ever one GPU, so this is equivalent to
        // `metal::Device::system_default()`.
        if let Some(device) = MTLCopyAllDevices()
            .iter()
            .min_by_key(|device| (device.isRemovable(), !device.isLowPower()))
        {
            Ok(device)
        } else {
            // For some reason `all()` can return an empty list, see https://github.com/zed-industries/zed/issues/37689
            // In that case, we fall back to the system default device.
            log::error!(
                "Unable to enumerate Metal devices; attempting to use system default device"
            );
            Self::system_default_device()
        }
    }

    #[cfg(target_os = "ios")]
    fn create_device() -> Result<Retained<ProtocolObject<dyn MTLDevice>>> {
        Self::system_default_device()
    }

    fn system_default_device() -> Result<Retained<ProtocolObject<dyn MTLDevice>>> {
        MTLCreateSystemDefaultDevice().context("accessing a compatible Metal device")
    }

    pub(crate) fn new_internal(
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        layer: Option<Retained<CAMetalLayer>>,
        opaque: bool,
        instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
    ) -> Result<Self> {
        #[cfg(feature = "runtime_shaders")]
        let library = device
            .newLibraryWithSource_options_error(
                &NSString::from_str(SHADERS_SOURCE_FILE),
                Some(&objc2_metal::MTLCompileOptions::new()),
            )
            .context("building Metal library")?;
        #[cfg(not(feature = "runtime_shaders"))]
        let library = device
            .newLibraryWithData_error(&dispatch2::DispatchData::from_static_bytes(
                SHADERS_METALLIB,
            ))
            .context("loading Metal library")?;

        fn to_float2_bits(point: PointF) -> u64 {
            let mut output = point.y.to_bits() as u64;
            output <<= 32;
            output |= point.x.to_bits() as u64;
            output
        }

        // Shared memory can be used only if CPU and GPU share the same memory space.
        // https://developer.apple.com/documentation/metal/setting-resource-storage-modes
        // iOS does not support managed resources. Its simulator may report a
        // non-unified host GPU even though resources must still use shared storage.
        let is_unified_memory = cfg!(target_os = "ios") || device.hasUnifiedMemory();
        // Apple GPU families support memoryless textures, which can significantly reduce
        // memory usage by keeping render targets in on-chip tile memory instead of
        // allocating backing store in system memory.
        // https://developer.apple.com/documentation/metal/mtlgpufamily
        let is_apple_gpu = device.supportsFamily(MTLGPUFamily::Apple1);

        let unit_vertices = [
            to_float2_bits(point(0., 0.)),
            to_float2_bits(point(1., 0.)),
            to_float2_bits(point(0., 1.)),
            to_float2_bits(point(0., 1.)),
            to_float2_bits(point(1., 0.)),
            to_float2_bits(point(1., 1.)),
        ];
        // SAFETY: Metal copies the live array using its exact byte length.
        let unit_vertices = unsafe {
            device.newBufferWithBytes_length_options(
                ptr::NonNull::from(&unit_vertices).cast(),
                mem::size_of_val(&unit_vertices),
                if is_unified_memory {
                    MTLResourceOptions::StorageModeShared
                        | MTLResourceOptions::CPUCacheModeWriteCombined
                } else {
                    MTLResourceOptions::StorageModeManaged
                },
            )
        }
        .context("allocating unit vertex buffer")?;

        let paths_rasterization_pipeline_state = build_path_rasterization_pipeline_state(
            &device,
            &library,
            "paths_rasterization",
            "path_rasterization_vertex",
            "path_rasterization_fragment",
            MTLPixelFormat::BGRA8Unorm,
            PATH_SAMPLE_COUNT,
        )?;
        let path_sprites_pipeline_state = build_path_sprite_pipeline_state(
            &device,
            &library,
            "path_sprites",
            "path_sprite_vertex",
            "path_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        )?;
        let shadows_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "shadows",
            "shadow_vertex",
            "shadow_fragment",
            MTLPixelFormat::BGRA8Unorm,
        )?;
        let quads_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "quads",
            "quad_vertex",
            "quad_fragment",
            MTLPixelFormat::BGRA8Unorm,
        )?;
        let underlines_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "underlines",
            "underline_vertex",
            "underline_fragment",
            MTLPixelFormat::BGRA8Unorm,
        )?;
        let monochrome_sprites_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "monochrome_sprites",
            "monochrome_sprite_vertex",
            "monochrome_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        )?;
        let polychrome_sprites_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "polychrome_sprites",
            "polychrome_sprite_vertex",
            "polychrome_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        )?;
        let surfaces_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "surfaces",
            "surface_vertex",
            "surface_fragment",
            MTLPixelFormat::BGRA8Unorm,
        )?;
        // Offscreen effect textures hold premultiplied colour, so they composite
        // with the same blend state as path sprites.
        let effect_composite_pipeline_state = build_path_sprite_pipeline_state(
            &device,
            &library,
            "effect_composite",
            "effect_composite_vertex",
            "effect_composite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        )?;
        let effect_pass_vertex_function = library
            .newFunctionWithName(&NSString::from_str("effect_pass_vertex"))
            .context("locating vertex function effect_pass_vertex")?;
        let effect_sampler = {
            // Shadertoy's defaults, which Ghostty's shaders are written against.
            let descriptor = MTLSamplerDescriptor::new();
            descriptor.setMinFilter(MTLSamplerMinMagFilter::Linear);
            descriptor.setMagFilter(MTLSamplerMinMagFilter::Linear);
            descriptor.setSAddressMode(MTLSamplerAddressMode::ClampToEdge);
            descriptor.setTAddressMode(MTLSamplerAddressMode::ClampToEdge);
            device
                .newSamplerStateWithDescriptor(&descriptor)
                .context("creating shader effect sampler")?
        };

        let command_queue = device.newCommandQueue().context("creating command queue")?;
        let supports_shared_storage = cfg!(target_os = "ios") || is_apple_gpu;
        #[expect(
            clippy::arc_with_non_send_sync,
            reason = "PlatformWindow::sprite_atlas requires Arc, not Send or Sync"
        )]
        let sprite_atlas = Arc::new(MetalAtlas::new(device.clone(), supports_shared_storage));
        let mut cache = ptr::null_mut();
        // SAFETY: The output pointer is writable and both optional attribute dictionaries are absent.
        let status = unsafe {
            CVMetalTextureCache::create(None, None, &device, None, ptr::NonNull::from(&mut cache))
        };
        anyhow::ensure!(
            status == kCVReturnSuccess,
            "creating CoreVideo Metal texture cache: status {status}"
        );
        let cache = ptr::NonNull::new(cache).context("retrieving CoreVideo Metal texture cache")?;
        // SAFETY: The successful create call transfers ownership of this non-null cache.
        let core_video_texture_cache = unsafe { CFRetained::from_raw(cache) };

        Ok(Self {
            device,
            layer,
            presents_with_transaction: false,
            is_apple_gpu,
            is_unified_memory,
            opaque,
            command_queue,
            paths_rasterization_pipeline_state,
            path_sprites_pipeline_state,
            shadows_pipeline_state,
            quads_pipeline_state,
            underlines_pipeline_state,
            monochrome_sprites_pipeline_state,
            polychrome_sprites_pipeline_state,
            surfaces_pipeline_state,
            unit_vertices,
            instance_buffer_pool,
            sprite_atlas,
            core_video_texture_cache,
            path_intermediate_texture: None,
            path_intermediate_msaa_texture: None,
            path_sample_count: PATH_SAMPLE_COUNT,
            effect_composite_pipeline_state,
            effect_pass_vertex_function,
            effect_sampler,
            effect_targets: HashMap::default(),
            effect_pipelines: HashMap::default(),
            effect_frame: 0,
            #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
            headless_render_target: None,
            fast_layers: crate::fast::layers::TileCache::default(),
        })
    }

    pub fn layer(&self) -> Option<&CAMetalLayer> {
        self.layer.as_deref()
    }

    pub fn layer_ptr(&self) -> *mut CAMetalLayer {
        self.layer
            .as_ref()
            .map(|layer| Retained::as_ptr(layer).cast_mut())
            .unwrap_or(ptr::null_mut())
    }

    pub fn sprite_atlas(&self) -> &Arc<MetalAtlas> {
        &self.sprite_atlas
    }

    pub fn set_presents_with_transaction(&mut self, presents_with_transaction: bool) {
        self.presents_with_transaction = presents_with_transaction;
        if let Some(layer) = &self.layer {
            layer.setPresentsWithTransaction(presents_with_transaction);
        }
    }

    /// Ghostex: the path textures are not reallocated here. A live resize calls this for every
    /// frame and most frames draw no path, so the textures are dropped and the next frame that
    /// draws a path allocates them at its own size (`ensure_window_path_textures`).
    pub fn update_drawable_size(&mut self, size: Size<DevicePixels>) {
        if let Some(layer) = &self.layer {
            layer.setDrawableSize(CGSize::new(size.width.0 as f64, size.height.0 as f64));
        }
        self.release_window_path_textures();
        crate::fast::layers::TileCache::clear(&mut self.fast_layers);
    }

    /// Ghostex: frees the window-sized path textures (about 4 bytes per drawable pixel, ~28 MB
    /// for a full-screen Retina window) until a frame draws a path again. Called when the window
    /// is resized and when it stops being visible (minimized, hidden, on another Space, covered).
    pub fn release_window_path_textures(&mut self) {
        self.path_intermediate_texture = None;
        self.path_intermediate_msaa_texture = None;
    }

    /// Ghostex: allocates the window's path textures at `size` unless they already have it.
    /// Called by a frame that draws paths, before it rasterizes them.
    pub(crate) fn ensure_window_path_textures(&mut self, size: Size<DevicePixels>) -> Result<()> {
        let current = self
            .path_intermediate_texture
            .as_ref()
            .is_some_and(|texture| {
                texture.width() == size.width.0 as usize
                    && texture.height() == size.height.0 as usize
            });
        if current {
            return Ok(());
        }
        self.update_path_intermediate_textures(size)
    }

    /// The window-sized textures paths drawn straight into the window, or
    /// into a texture of its size, are rasterized through.
    pub(crate) fn window_path_textures(&self) -> PathIntermediateTextures {
        PathIntermediateTextures {
            texture: self.path_intermediate_texture.clone(),
            msaa_texture: self.path_intermediate_msaa_texture.clone(),
        }
    }

    pub(crate) fn update_path_intermediate_textures(
        &mut self,
        size: Size<DevicePixels>,
    ) -> Result<()> {
        // We are uncertain when this happens, but sometimes size can be 0 here. Most likely before
        // the layout pass on window creation. Zero-sized texture creation causes SIGABRT.
        // https://github.com/zed-industries/zed/issues/36229
        self.path_intermediate_texture = None;
        self.path_intermediate_msaa_texture = None;
        if size.width.0 <= 0 || size.height.0 <= 0 {
            return Ok(());
        }

        let textures = new_path_intermediate_textures(
            &self.device,
            size,
            self.path_sample_count,
            self.is_apple_gpu,
        )?;
        self.path_intermediate_texture = textures.texture;
        self.path_intermediate_msaa_texture = textures.msaa_texture;
        Ok(())
    }

    pub fn update_transparency(&mut self, transparent: bool) {
        self.opaque = !transparent;
        if let Some(layer) = &self.layer {
            layer.setOpaque(!transparent);
        }
    }

    pub fn destroy(&self) {
        // nothing to do
    }

    pub fn draw(&mut self, scene: &Scene) {
        let layer = match &self.layer {
            Some(l) => l.clone(),
            None => {
                log::error!(
                    "draw() called on headless renderer - use render_scene_to_image() instead"
                );
                return;
            }
        };
        let viewport_size = layer.drawableSize();
        let viewport_size: Size<DevicePixels> = size(
            (viewport_size.width.ceil() as i32).into(),
            (viewport_size.height.ceil() as i32).into(),
        );
        let drawable = if let Some(drawable) = layer.nextDrawable() {
            drawable
        } else {
            log::error!(
                "failed to retrieve next drawable, drawable size: {:?}",
                viewport_size
            );
            return;
        };

        let command_buffer = match self.render_frame(scene, &drawable.texture(), viewport_size) {
            Ok(command_buffer) => command_buffer,
            Err(error) => {
                log::error!("failed to render: {error:#}");
                return;
            }
        };

        if self.presents_with_transaction {
            command_buffer.commit();
            command_buffer.waitUntilScheduled();
            drawable.present();
        } else {
            command_buffer.presentDrawable(ProtocolObject::from_ref(&*drawable));
            command_buffer.commit();
        }
    }

    pub(crate) fn render_frame(
        &mut self,
        scene: &Scene,
        texture: &ProtocolObject<dyn MTLTexture>,
        viewport_size: Size<DevicePixels>,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        crate::fast::layers::raster::rasterize_tiles(self, scene);
        let mut writer = InstanceBufferWriter::new(
            &self.device,
            &self.instance_buffer_pool,
            self.is_unified_memory,
        )?;
        let instance_bindings = write_instances(scene, &mut writer).with_context(|| {
            format!(
                "writing instance data: {} paths, {} shadows, {} quads, {} underlines, {} mono, {} poly, {} surfaces",
                scene.paths.len(),
                scene.shadows.len(),
                scene.quads.len(),
                scene.underlines.len(),
                scene.monochrome_sprites.len(),
                scene.polychrome_sprites.len(),
                scene.surfaces.len(),
            )
        })?;
        let mut surface_textures = Vec::new();
        self.effect_frame += 1;
        let command_buffer = self.draw_primitives_to_texture(
            scene,
            &instance_bindings,
            &mut writer,
            texture,
            viewport_size,
            &mut surface_textures,
        );
        self.retire_effect_resources();
        let command_buffer = command_buffer?;

        let instance_buffer_pool = self.instance_buffer_pool.clone();
        let frame_resources = Cell::new(Some((writer.finish(), surface_textures)));
        let block = RcBlock::new(
            move |_: ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                if let Some((instance_buffer, _surface_textures)) = frame_resources.take() {
                    instance_buffer_pool.lock().release(instance_buffer);
                }
            },
        );
        // SAFETY: Metal copies the live block and invokes it once after this frame completes.
        // Its captures own the frame resources and use the pool mutex when returning the buffer.
        unsafe {
            command_buffer.addCompletedHandler(RcBlock::as_ptr(&block));
        }

        Ok(command_buffer)
    }

    /// Renders the scene to a texture and returns the pixel data as an RGBA image.
    /// This does not present the frame to screen - useful for visual testing
    /// where we want to capture what would be rendered without displaying it.
    ///
    /// Note: This requires a layer-backed renderer. For headless rendering,
    /// use `render_scene_to_image()` instead.
    #[cfg(any(test, feature = "test-support"))]
    pub fn render_to_image(&mut self, scene: &Scene) -> Result<RgbaImage> {
        let layer = self
            .layer
            .clone()
            .ok_or_else(|| anyhow::anyhow!("render_to_image requires a layer-backed renderer"))?;
        let viewport_size = layer.drawableSize();
        let viewport_size: Size<DevicePixels> = size(
            (viewport_size.width.ceil() as i32).into(),
            (viewport_size.height.ceil() as i32).into(),
        );
        let drawable = layer
            .nextDrawable()
            .ok_or_else(|| anyhow::anyhow!("Failed to get drawable for render_to_image"))?;

        let command_buffer = self.render_frame(scene, &drawable.texture(), viewport_size)?;

        // Commit and wait for completion without presenting
        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        read_texture_to_image(&drawable.texture())
    }

    /// Renders a scene to an image without requiring a window or CAMetalLayer.
    ///
    /// This is the primary method for headless rendering. It creates an offscreen
    /// texture, renders the scene to it, and returns the pixel data as an RGBA image.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    pub fn render_scene_to_image(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
    ) -> Result<RgbaImage> {
        if size.width.0 <= 0 || size.height.0 <= 0 {
            anyhow::bail!("Invalid size for render_scene_to_image: {:?}", size);
        }

        // Headless callers do not have a Cocoa event-loop pool to release
        // autoreleased command buffers and render-pass descriptors.
        objc2::rc::autoreleasepool(|_| {
            // Update path intermediate textures for this size
            self.update_path_intermediate_textures(size)?;

            // Create an offscreen texture as render target
            let texture_descriptor = MTLTextureDescriptor::new();
            unsafe {
                texture_descriptor.setWidth(size.width.0 as usize);
                texture_descriptor.setHeight(size.height.0 as usize);
            }
            texture_descriptor.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
            texture_descriptor
                .setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
            // Like the atlas, only Apple GPUs can create shared textures on macOS;
            // Intel Macs cannot, even with unified memory. iOS has no managed storage.
            let uses_shared_storage = cfg!(target_os = "ios") || self.is_apple_gpu;
            texture_descriptor.setStorageMode(if uses_shared_storage {
                MTLStorageMode::Shared
            } else {
                MTLStorageMode::Managed
            });
            let target_texture = self
                .device
                .newTextureWithDescriptor(&texture_descriptor)
                .context("allocating headless render target")?;

            let command_buffer = self.render_frame(scene, &target_texture, size)?;

            // Managed textures require an explicit blit synchronize before the CPU
            // can read back the rendered data. Without this, get_bytes returns
            // stale zeros.
            if !uses_shared_storage {
                let blit = command_buffer
                    .blitCommandEncoder()
                    .context("creating blit command encoder")?;
                blit.synchronizeResource(ProtocolObject::from_ref(&*target_texture));
                blit.endEncoding();
            }

            // Commit and wait for completion
            command_buffer.commit();
            command_buffer.waitUntilCompleted();

            read_texture_to_image(&target_texture)
        })
    }

    /// Renders a scene to a reused offscreen texture without reading pixels
    /// back or blocking on GPU completion.
    ///
    /// This mirrors the CPU cost of presenting a frame to a window (scene
    /// encoding, instance buffer writes, command submission) and is used by
    /// headless benchmark rendering, where the produced pixels are never
    /// inspected.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    pub fn render_scene(&mut self, scene: &Scene, size: Size<DevicePixels>) -> Result<()> {
        if size.width.0 <= 0 || size.height.0 <= 0 {
            anyhow::bail!("Invalid size for render_scene: {:?}", size);
        }

        objc2::rc::autoreleasepool(|_| {
            self.update_path_intermediate_textures(size)?;

            let needs_new_target = self.headless_render_target.as_ref().is_none_or(|texture| {
                texture.width() != size.width.0 as usize
                    || texture.height() != size.height.0 as usize
            });
            if needs_new_target {
                let texture_descriptor = MTLTextureDescriptor::new();
                unsafe {
                    texture_descriptor.setWidth(size.width.0 as usize);
                    texture_descriptor.setHeight(size.height.0 as usize);
                }
                texture_descriptor.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
                texture_descriptor
                    .setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
                texture_descriptor.setStorageMode(MTLStorageMode::Private);
                self.headless_render_target = Some(
                    self.device
                        .newTextureWithDescriptor(&texture_descriptor)
                        .context("allocating headless render target")?,
                );
            }
            let target_texture = self
                .headless_render_target
                .clone()
                .context("retrieving headless render target")?;

            let command_buffer = self.render_frame(scene, &target_texture, size)?;

            // Commit without waiting, mirroring presentation to a real window where
            // the CPU doesn't block on the GPU.
            command_buffer.commit();
            Ok(())
        })
    }

    fn draw_primitives_to_texture(
        &mut self,
        scene: &Scene,
        instance_bindings: &InstanceBindings,
        writer: &mut InstanceBufferWriter,
        texture: &ProtocolObject<dyn MTLTexture>,
        viewport_size: Size<DevicePixels>,
        surface_textures: &mut SurfaceTextures,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        if let Some(command_buffer) = crate::fast::paths::draw_primitives_to_texture(
            self,
            scene,
            instance_bindings,
            writer,
            texture,
            viewport_size,
            surface_textures,
        )? {
            return Ok(command_buffer);
        }
        let command_queue = self.command_queue.clone();
        let command_buffer = command_queue
            .commandBuffer()
            .context("creating command buffer")?;
        let alpha = if self.opaque { 1. } else { 0. };
        let path_textures = self.window_path_textures();

        self.encode_scene(
            scene,
            instance_bindings,
            writer,
            &command_buffer,
            texture,
            viewport_size,
            MTLClearColor {
                red: 0.,
                green: 0.,
                blue: 0.,
                alpha,
            },
            &path_textures,
            surface_textures,
        )?;

        Ok(command_buffer)
    }

    /// Encodes `scene` into `texture`, whose top-left corner is the scene's
    /// origin. Leaves no render encoder open, including on error.
    #[allow(clippy::too_many_arguments)]
    fn encode_scene(
        &mut self,
        scene: &Scene,
        instance_bindings: &InstanceBindings,
        writer: &mut InstanceBufferWriter,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        texture: &ProtocolObject<dyn MTLTexture>,
        viewport_size: Size<DevicePixels>,
        clear_color: MTLClearColor,
        path_textures: &PathIntermediateTextures,
        surface_textures: &mut SurfaceTextures,
    ) -> Result<()> {
        let mut command_encoder = new_command_encoder_for_texture(
            command_buffer,
            texture,
            viewport_size,
            Some(clear_color),
        )?;

        for batch in scene.scene_batches() {
            let batch = match batch {
                SceneBatch::Primitives(batch) => batch,
                SceneBatch::Effect(effect_index) => {
                    command_encoder = self.encode_effect(
                        scene,
                        effect_index,
                        writer,
                        command_buffer,
                        texture,
                        viewport_size,
                        &command_encoder,
                        surface_textures,
                    )?;
                    continue;
                }
            };
            match batch {
                PrimitiveBatch::Shadows(range) => {
                    self.draw_shadows(range, instance_bindings, viewport_size, &command_encoder)
                }
                PrimitiveBatch::Quads(range) => {
                    self.draw_quads(range, instance_bindings, viewport_size, &command_encoder)
                }
                PrimitiveBatch::Paths(range) => {
                    let paths = &scene.paths[range];
                    command_encoder.endEncoding();

                    let did_draw = self.draw_paths_to_intermediate(
                        paths,
                        writer,
                        viewport_size,
                        command_buffer,
                        path_textures,
                    )?;

                    command_encoder = new_command_encoder_for_texture(
                        command_buffer,
                        texture,
                        viewport_size,
                        None,
                    )?;

                    if did_draw {
                        if let Err(error) = self.draw_paths_from_intermediate(
                            paths,
                            writer,
                            viewport_size,
                            &command_encoder,
                            path_textures,
                        ) {
                            command_encoder.endEncoding();
                            return Err(error);
                        }
                    }
                }
                PrimitiveBatch::Underlines(range) => {
                    self.draw_underlines(range, instance_bindings, viewport_size, &command_encoder)
                }
                PrimitiveBatch::MonochromeSprites { texture_id, range } => self
                    .draw_monochrome_sprites(
                        texture_id,
                        range,
                        instance_bindings,
                        viewport_size,
                        &command_encoder,
                    ),
                PrimitiveBatch::PolychromeSprites { texture_id, range } => self
                    .draw_polychrome_sprites(
                        texture_id,
                        range,
                        instance_bindings,
                        viewport_size,
                        &command_encoder,
                    ),
                PrimitiveBatch::Surfaces(range) => {
                    if let Err(error) = self.draw_surfaces(
                        &scene.surfaces[range.clone()],
                        range.start,
                        instance_bindings,
                        viewport_size,
                        &command_encoder,
                        surface_textures,
                    ) {
                        command_encoder.endEncoding();
                        return Err(error);
                    }
                }
                PrimitiveBatch::SubpixelSprites { .. } => unreachable!(),
            }
        }

        command_encoder.endEncoding();

        Ok(())
    }

    pub(crate) fn draw_paths_to_intermediate(
        &self,
        paths: &[Path<ScaledPixels>],
        writer: &mut InstanceBufferWriter,
        viewport_size: Size<DevicePixels>,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        path_textures: &PathIntermediateTextures,
    ) -> Result<bool> {
        if paths.is_empty() {
            return Ok(false);
        }
        let intermediate_texture = path_textures
            .texture
            .as_ref()
            .context("missing path intermediate texture")?;

        let mut vertices = Vec::new();
        for path in paths {
            vertices.extend(path.vertices.iter().map(|v| PathRasterizationVertex {
                xy_position: v.xy_position,
                st_position: v.st_position,
                color: path.color,
                bounds: path.bounds.intersect(&path.content_mask.bounds),
            }));
        }
        let vertex_instance_bindings = writer.write(&vertices)?;

        let render_pass_descriptor = MTLRenderPassDescriptor::new();
        // SAFETY: Index zero is within the render pass's color attachment array.
        let color_attachment = unsafe {
            render_pass_descriptor
                .colorAttachments()
                .objectAtIndexedSubscript(0)
        };
        color_attachment.setLoadAction(MTLLoadAction::Clear);
        color_attachment.setClearColor(MTLClearColor {
            red: 0.,
            green: 0.,
            blue: 0.,
            alpha: 0.,
        });

        if let Some(msaa_texture) = &path_textures.msaa_texture {
            color_attachment.setTexture(Some(msaa_texture));
            color_attachment.setResolveTexture(Some(intermediate_texture));
            color_attachment.setStoreAction(MTLStoreAction::MultisampleResolve);
        } else {
            color_attachment.setTexture(Some(intermediate_texture));
            color_attachment.setStoreAction(MTLStoreAction::Store);
        }

        let command_encoder = command_buffer
            .renderCommandEncoderWithDescriptor(&render_pass_descriptor)
            .context("creating path render command encoder")?;
        command_encoder.setRenderPipelineState(&self.paths_rasterization_pipeline_state);
        // SAFETY: The binding contains vertices.len() shader-compatible vertices at an aligned offset.
        // Metal retains the buffer, and writer.finish publishes CPU writes before submission.
        // The viewport bytes are copied from a live value into the matching shader slot.
        unsafe {
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&vertex_instance_bindings.buffer),
                vertex_instance_bindings.offset,
                PathRasterizationInputIndex::Vertices as usize,
            );
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(&viewport_size).cast(),
                mem::size_of_val(&viewport_size),
                PathRasterizationInputIndex::ViewportSize as usize,
            );
            command_encoder.setFragmentBuffer_offset_atIndex(
                Some(&vertex_instance_bindings.buffer),
                vertex_instance_bindings.offset,
                PathRasterizationInputIndex::Vertices as usize,
            );
            command_encoder.drawPrimitives_vertexStart_vertexCount(
                MTLPrimitiveType::Triangle,
                0,
                vertices.len(),
            );
        }

        command_encoder.endEncoding();
        Ok(true)
    }

    pub(crate) fn draw_shadows(
        &self,
        shadows: Range<usize>,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    ) {
        if shadows.is_empty() {
            return;
        }

        command_encoder.setRenderPipelineState(&self.shadows_pipeline_state);
        // SAFETY: Six unit vertices and the scene's shadow range match the uploaded shader layouts and slots.
        // Metal retains the buffers, writer.finish publishes their writes, and viewport bytes are copied.
        unsafe {
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&self.unit_vertices),
                0,
                ShadowInputIndex::Vertices as usize,
            );
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&instance_bindings.shadows.buffer),
                instance_bindings.shadows.offset,
                ShadowInputIndex::Shadows as usize,
            );
            command_encoder.setFragmentBuffer_offset_atIndex(
                Some(&instance_bindings.shadows.buffer),
                instance_bindings.shadows.offset,
                ShadowInputIndex::Shadows as usize,
            );
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(&viewport_size).cast(),
                mem::size_of_val(&viewport_size),
                ShadowInputIndex::ViewportSize as usize,
            );
            command_encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                MTLPrimitiveType::Triangle,
                0,
                6,
                shadows.len(),
                shadows.start,
            );
        }
    }

    pub(crate) fn draw_quads(
        &self,
        quads: Range<usize>,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    ) {
        if quads.is_empty() {
            return;
        }

        command_encoder.setRenderPipelineState(&self.quads_pipeline_state);
        // SAFETY: Six unit vertices and the scene's quad range match the uploaded shader layouts and slots.
        // Metal retains the buffers, writer.finish publishes their writes, and viewport bytes are copied.
        unsafe {
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&self.unit_vertices),
                0,
                QuadInputIndex::Vertices as usize,
            );
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&instance_bindings.quads.buffer),
                instance_bindings.quads.offset,
                QuadInputIndex::Quads as usize,
            );
            command_encoder.setFragmentBuffer_offset_atIndex(
                Some(&instance_bindings.quads.buffer),
                instance_bindings.quads.offset,
                QuadInputIndex::Quads as usize,
            );
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(&viewport_size).cast(),
                mem::size_of_val(&viewport_size),
                QuadInputIndex::ViewportSize as usize,
            );
            command_encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                MTLPrimitiveType::Triangle,
                0,
                6,
                quads.len(),
                quads.start,
            );
        }
    }

    pub(crate) fn draw_paths_from_intermediate(
        &self,
        paths: &[Path<ScaledPixels>],
        writer: &mut InstanceBufferWriter,
        viewport_size: Size<DevicePixels>,
        command_encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
        path_textures: &PathIntermediateTextures,
    ) -> Result<()> {
        let (Some(first_path), Some(last_path)) = (paths.first(), paths.last()) else {
            return Ok(());
        };
        let intermediate_texture = path_textures
            .texture
            .as_ref()
            .context("missing path intermediate texture")?;

        command_encoder.setRenderPipelineState(&self.path_sprites_pipeline_state);
        // SAFETY: The shader slots match the unit vertices, copied viewport, and retained path texture.
        // The preceding render pass writes this tracked texture before it is sampled.
        unsafe {
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&self.unit_vertices),
                0,
                SpriteInputIndex::Vertices as usize,
            );
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(&viewport_size).cast(),
                mem::size_of_val(&viewport_size),
                SpriteInputIndex::ViewportSize as usize,
            );
            command_encoder.setFragmentTexture_atIndex(
                Some(intermediate_texture),
                SpriteInputIndex::AtlasTexture as usize,
            );
        }

        // When copying paths from the intermediate texture to the drawable,
        // each pixel must only be copied once, in case of transparent paths.
        //
        // If all paths have the same draw order, then their bounds are all
        // disjoint, so we can copy each path's bounds individually. If this
        // batch combines different draw orders, we perform a single copy
        // for a minimal spanning rect.
        let sprites;
        if last_path.order == first_path.order {
            sprites = paths
                .iter()
                .map(|path| PathSprite {
                    bounds: path.clipped_bounds(),
                })
                .collect();
        } else {
            let mut bounds = first_path.clipped_bounds();
            for path in paths.iter().skip(1) {
                bounds = bounds.union(&path.clipped_bounds());
            }
            sprites = vec![PathSprite { bounds }];
        }

        let sprite_instance_bindings = writer.write(&sprites)?;
        // SAFETY: The draw uses six unit vertices and exactly the sprites written at this aligned binding.
        // Metal retains the buffer, and writer.finish publishes CPU writes before submission.
        unsafe {
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&sprite_instance_bindings.buffer),
                sprite_instance_bindings.offset,
                SpriteInputIndex::Sprites as usize,
            );
            command_encoder.drawPrimitives_vertexStart_vertexCount_instanceCount(
                MTLPrimitiveType::Triangle,
                0,
                6,
                sprites.len(),
            );
        }
        Ok(())
    }

    pub(crate) fn draw_underlines(
        &self,
        underlines: Range<usize>,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    ) {
        if underlines.is_empty() {
            return;
        }

        command_encoder.setRenderPipelineState(&self.underlines_pipeline_state);
        // SAFETY: Six unit vertices and the scene's underline range match the uploaded shader layouts and slots.
        // Metal retains the buffers, writer.finish publishes their writes, and viewport bytes are copied.
        unsafe {
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&self.unit_vertices),
                0,
                UnderlineInputIndex::Vertices as usize,
            );
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&instance_bindings.underlines.buffer),
                instance_bindings.underlines.offset,
                UnderlineInputIndex::Underlines as usize,
            );
            command_encoder.setFragmentBuffer_offset_atIndex(
                Some(&instance_bindings.underlines.buffer),
                instance_bindings.underlines.offset,
                UnderlineInputIndex::Underlines as usize,
            );
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(&viewport_size).cast(),
                mem::size_of_val(&viewport_size),
                UnderlineInputIndex::ViewportSize as usize,
            );
            command_encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                MTLPrimitiveType::Triangle,
                0,
                6,
                underlines.len(),
                underlines.start,
            );
        }
    }

    pub(crate) fn draw_monochrome_sprites(
        &self,
        texture_id: AtlasTextureId,
        sprites: Range<usize>,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    ) {
        if sprites.is_empty() {
            return;
        }

        let Some(texture) = self.sprite_atlas.metal_texture(texture_id) else {
            return;
        };
        let texture_size = size(
            DevicePixels(texture.width() as i32),
            DevicePixels(texture.height() as i32),
        );
        command_encoder.setRenderPipelineState(&self.monochrome_sprites_pipeline_state);
        // SAFETY: The scene's sprite range, six unit vertices, and copied sizes match the shader layouts and slots.
        // Metal retains the buffers and texture, and writer.finish publishes the instance writes.
        unsafe {
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&self.unit_vertices),
                0,
                SpriteInputIndex::Vertices as usize,
            );
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&instance_bindings.monochrome_sprites.buffer),
                instance_bindings.monochrome_sprites.offset,
                SpriteInputIndex::Sprites as usize,
            );
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(&viewport_size).cast(),
                mem::size_of_val(&viewport_size),
                SpriteInputIndex::ViewportSize as usize,
            );
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(&texture_size).cast(),
                mem::size_of_val(&texture_size),
                SpriteInputIndex::AtlasTextureSize as usize,
            );
            command_encoder.setFragmentBuffer_offset_atIndex(
                Some(&instance_bindings.monochrome_sprites.buffer),
                instance_bindings.monochrome_sprites.offset,
                SpriteInputIndex::Sprites as usize,
            );
            command_encoder.setFragmentTexture_atIndex(
                Some(&texture),
                SpriteInputIndex::AtlasTexture as usize,
            );
            command_encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                MTLPrimitiveType::Triangle,
                0,
                6,
                sprites.len(),
                sprites.start,
            );
        }
    }

    pub(crate) fn draw_polychrome_sprites(
        &self,
        texture_id: AtlasTextureId,
        sprites: Range<usize>,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    ) {
        if sprites.is_empty() {
            return;
        }
        if crate::fast::layers::composite::draw_tiles(
            self,
            texture_id,
            &sprites,
            instance_bindings,
            viewport_size,
            command_encoder,
        ) {
            return;
        }

        let Some(texture) = self.sprite_atlas.metal_texture(texture_id) else {
            return;
        };
        let texture_size = size(
            DevicePixels(texture.width() as i32),
            DevicePixels(texture.height() as i32),
        );
        command_encoder.setRenderPipelineState(&self.polychrome_sprites_pipeline_state);
        // SAFETY: The scene's sprite range, six unit vertices, and copied sizes match the shader layouts and slots.
        // Metal retains the buffers and texture, and writer.finish publishes the instance writes.
        unsafe {
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&self.unit_vertices),
                0,
                SpriteInputIndex::Vertices as usize,
            );
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&instance_bindings.polychrome_sprites.buffer),
                instance_bindings.polychrome_sprites.offset,
                SpriteInputIndex::Sprites as usize,
            );
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(&viewport_size).cast(),
                mem::size_of_val(&viewport_size),
                SpriteInputIndex::ViewportSize as usize,
            );
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(&texture_size).cast(),
                mem::size_of_val(&texture_size),
                SpriteInputIndex::AtlasTextureSize as usize,
            );
            command_encoder.setFragmentBuffer_offset_atIndex(
                Some(&instance_bindings.polychrome_sprites.buffer),
                instance_bindings.polychrome_sprites.offset,
                SpriteInputIndex::Sprites as usize,
            );
            command_encoder.setFragmentTexture_atIndex(
                Some(&texture),
                SpriteInputIndex::AtlasTexture as usize,
            );
            command_encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                MTLPrimitiveType::Triangle,
                0,
                6,
                sprites.len(),
                sprites.start,
            );
        }
    }

    pub(crate) fn draw_surfaces(
        &mut self,
        surfaces: &[PaintSurface],
        first_surface: usize,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
        surface_textures: &mut SurfaceTextures,
    ) -> Result<()> {
        if surfaces.is_empty() {
            return Ok(());
        }

        command_encoder.setRenderPipelineState(&self.surfaces_pipeline_state);
        // SAFETY: The unit vertices, SurfaceBounds instances, and copied viewport match the shader layouts and slots.
        // Metal retains the buffers, and writer.finish publishes CPU writes before submission.
        unsafe {
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&self.unit_vertices),
                0,
                SurfaceInputIndex::Vertices as usize,
            );
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&instance_bindings.surfaces.buffer),
                instance_bindings.surfaces.offset,
                SurfaceInputIndex::Surfaces as usize,
            );
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(&viewport_size).cast(),
                mem::size_of_val(&viewport_size),
                SurfaceInputIndex::ViewportSize as usize,
            );
        }

        for (index, surface) in surfaces.iter().enumerate() {
            let image_buffer = &surface.image_buffer;
            let texture_size = size(
                DevicePixels::from(CVPixelBufferGetWidth(image_buffer) as i32),
                DevicePixels::from(CVPixelBufferGetHeight(image_buffer) as i32),
            );

            let pixel_format = CVPixelBufferGetPixelFormatType(image_buffer);
            anyhow::ensure!(
                pixel_format == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                "unsupported surface pixel format: {pixel_format:#x}"
            );

            let plane_texture = |plane, format| -> Result<CFRetained<CVMetalTexture>> {
                let width = CVPixelBufferGetWidthOfPlane(image_buffer, plane);
                let height = CVPixelBufferGetHeightOfPlane(image_buffer, plane);
                let mut texture = ptr::null_mut();
                // SAFETY: The buffer has two planes matching these formats and dimensions, and the output pointer is writable.
                let status = unsafe {
                    CVMetalTextureCache::create_texture_from_image(
                        None,
                        &self.core_video_texture_cache,
                        image_buffer,
                        None,
                        format,
                        width,
                        height,
                        plane,
                        ptr::NonNull::from(&mut texture),
                    )
                };
                anyhow::ensure!(
                    status == kCVReturnSuccess,
                    "creating CoreVideo Metal texture for plane {plane}: status {status}"
                );
                let texture =
                    ptr::NonNull::new(texture).context("retrieving CoreVideo Metal texture")?;
                // SAFETY: The successful create call transfers ownership of this non-null texture.
                Ok(unsafe { CFRetained::from_raw(texture) })
            };
            let y_texture = plane_texture(0, MTLPixelFormat::R8Unorm)?;
            let cb_cr_texture = plane_texture(1, MTLPixelFormat::RG8Unorm)?;
            let y_metal_texture =
                CVMetalTextureGetTexture(&y_texture).context("retrieving luma Metal texture")?;
            let cb_cr_metal_texture = CVMetalTextureGetTexture(&cb_cr_texture)
                .context("retrieving chroma Metal texture")?;

            // SAFETY: The size is copied from a live value, and both planes match their shader texture slots.
            // The frame completion handler retains the CoreVideo wrappers until GPU reads finish.
            unsafe {
                command_encoder.setVertexBytes_length_atIndex(
                    ptr::NonNull::from(&texture_size).cast(),
                    mem::size_of_val(&texture_size),
                    SurfaceInputIndex::TextureSize as usize,
                );
                command_encoder.setFragmentTexture_atIndex(
                    Some(&y_metal_texture),
                    SurfaceInputIndex::YTexture as usize,
                );
                command_encoder.setFragmentTexture_atIndex(
                    Some(&cb_cr_metal_texture),
                    SurfaceInputIndex::CbCrTexture as usize,
                );
            }
            surface_textures.extend([y_texture, cb_cr_texture]);

            // SAFETY: Six unit vertices draw one instance within the uploaded scene's surface range.
            unsafe {
                command_encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                    MTLPrimitiveType::Triangle,
                    0,
                    6,
                    1,
                    first_surface + index,
                );
            }
        }
        Ok(())
    }

    /// Draws the scene's effect at `effect_index` into `texture`. The effect renders into its own
    /// textures, so the pass on `texture` (`command_encoder`) is closed and a new one, loading its
    /// contents, is returned with the effect composited, the same way paths are drawn.
    ///
    /// Ghostex: every renderer a frame can take draws its effects through this (`encode_scene`,
    /// and GPUI Fast's path and rounded-clip renderers, `fast::paths` and `fast::rounded_clip`).
    /// The GPUI Fast ones walked only the primitive batches, so a frame with a path or a rounded
    /// clip dropped every effect, which left a terminal with a custom shader blank.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_effect(
        &mut self,
        scene: &Scene,
        effect_index: usize,
        writer: &mut InstanceBufferWriter,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        texture: &ProtocolObject<dyn MTLTexture>,
        viewport_size: Size<DevicePixels>,
        command_encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
        surface_textures: &mut SurfaceTextures,
    ) -> Result<Retained<ProtocolObject<dyn MTLRenderCommandEncoder>>> {
        command_encoder.endEncoding();
        let effect = scene.effects.get(effect_index);
        let output = match effect {
            Some(effect) => self.render_effect(effect, writer, command_buffer, surface_textures)?,
            None => None,
        };
        let command_encoder =
            new_command_encoder_for_texture(command_buffer, texture, viewport_size, None)?;
        if let (Some(effect), Some(output)) = (effect, output)
            && let Err(error) =
                self.composite_effect(effect, &output, writer, viewport_size, &command_encoder)
        {
            command_encoder.endEncoding();
            return Err(error);
        }
        Ok(command_encoder)
    }

    /// Renders the effect's captured scene into its offscreen texture, runs
    /// the shader passes over it, and returns the texture holding the result.
    /// Must be called with no render encoder open on `command_buffer`.
    fn render_effect(
        &mut self,
        effect: &PaintEffect,
        writer: &mut InstanceBufferWriter,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        surface_textures: &mut SurfaceTextures,
    ) -> Result<Option<Retained<ProtocolObject<dyn MTLTexture>>>> {
        // Effect bounds are snapped to whole device pixels when painted, so the
        // texture maps 1:1 onto the window.
        let texture_size = size(
            DevicePixels(effect.bounds.size.width.0.round() as i32),
            DevicePixels(effect.bounds.size.height.0.round() as i32),
        );
        if texture_size.width.0 <= 0 || texture_size.height.0 <= 0 {
            return Ok(None);
        }
        if texture_size.width.0 > gpui::MAX_SHADER_EFFECT_TEXTURE_SIZE
            || texture_size.height.0 > gpui::MAX_SHADER_EFFECT_TEXTURE_SIZE
        {
            log::error!(
                "skipping shader effect {}: {texture_size:?} exceeds the maximum texture size",
                effect.effect.id
            );
            return Ok(None);
        }

        let pipelines: Vec<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> = effect
            .effect
            .shaders
            .iter()
            .map(|source| self.effect_pipeline(source))
            .collect::<Option<Vec<_>>>()
            .unwrap_or_default();

        let frame = self.effect_frame;
        let needs_target = self
            .effect_targets
            .get(&effect.effect.id)
            .is_none_or(|target| target.size != texture_size);
        if needs_target {
            let target = EffectTarget::new(&self.device, texture_size)?;
            self.effect_targets.insert(effect.effect.id, target);
        }
        let device = &self.device;
        let target = self
            .effect_targets
            .get_mut(&effect.effect.id)
            .context("missing shader effect target")?;
        target.last_used_frame = frame;
        if !effect.scene.paths.is_empty() && target.path_textures.texture.is_none() {
            target.path_textures = new_path_intermediate_textures(
                device,
                texture_size,
                self.path_sample_count,
                self.is_apple_gpu,
            )?;
        }
        let [mut input, mut output] = target.textures.clone();
        let path_textures = target.path_textures.clone();

        let local_scene = scene_in_effect_space(&effect.scene, effect.bounds.origin);
        let instance_bindings = write_instances(&local_scene, writer)?;
        self.encode_scene(
            &local_scene,
            &instance_bindings,
            writer,
            command_buffer,
            &input,
            texture_size,
            MTLClearColor {
                red: 0.,
                green: 0.,
                blue: 0.,
                alpha: 0.,
            },
            &path_textures,
            surface_textures,
        )?;

        if pipelines.is_empty() {
            return Ok(Some(input));
        }

        // The palette makes Ghostty's uniform block larger than the 4 KiB
        // `setFragmentBytes` allows, so it goes through the instance buffer,
        // whose 256-byte offset alignment also suits constant buffers.
        let uniforms =
            write_padded_bytes(writer, &effect.effect.uniforms, MIN_EFFECT_UNIFORM_SIZE)?;
        for pipeline in &pipelines {
            let render_pass_descriptor = MTLRenderPassDescriptor::new();
            // SAFETY: Index zero is within the render pass's color attachment array.
            let color_attachment = unsafe {
                render_pass_descriptor
                    .colorAttachments()
                    .objectAtIndexedSubscript(0)
            };
            color_attachment.setTexture(Some(&output));
            color_attachment.setLoadAction(MTLLoadAction::Clear);
            color_attachment.setClearColor(MTLClearColor {
                red: 0.,
                green: 0.,
                blue: 0.,
                alpha: 0.,
            });
            color_attachment.setStoreAction(MTLStoreAction::Store);

            let command_encoder = command_buffer
                .renderCommandEncoderWithDescriptor(&render_pass_descriptor)
                .context("creating shader effect render command encoder")?;
            command_encoder.setRenderPipelineState(pipeline);
            // SAFETY: The pass's fragment shader reads the input texture at slot 0, the sampler
            // at slot 0 and the uniform block at slot 1, which writer.finish publishes before
            // submission. The full-screen triangle is generated from the vertex id.
            unsafe {
                command_encoder.setFragmentTexture_atIndex(Some(&input), 0);
                command_encoder.setFragmentSamplerState_atIndex(Some(&self.effect_sampler), 0);
                command_encoder.setFragmentBuffer_offset_atIndex(
                    Some(&uniforms.buffer),
                    uniforms.offset,
                    1,
                );
                command_encoder.drawPrimitives_vertexStart_vertexCount(
                    MTLPrimitiveType::Triangle,
                    0,
                    3,
                );
            }
            command_encoder.endEncoding();

            mem::swap(&mut input, &mut output);
        }

        Ok(Some(input))
    }

    fn composite_effect(
        &self,
        effect: &PaintEffect,
        effect_texture: &ProtocolObject<dyn MTLTexture>,
        writer: &mut InstanceBufferWriter,
        viewport_size: Size<DevicePixels>,
        command_encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    ) -> Result<()> {
        let bounds_binding = writer.write(&[EffectCompositeBounds {
            bounds: effect.bounds,
            content_mask: effect.content_mask,
        }])?;

        command_encoder.setRenderPipelineState(&self.effect_composite_pipeline_state);
        // SAFETY: Six unit vertices draw the one EffectCompositeBounds written at this aligned
        // binding, which writer.finish publishes before submission; the viewport bytes are copied
        // from a live value, and the effect texture is retained by Metal.
        unsafe {
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&self.unit_vertices),
                0,
                EffectCompositeInputIndex::Vertices as usize,
            );
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&bounds_binding.buffer),
                bounds_binding.offset,
                EffectCompositeInputIndex::Bounds as usize,
            );
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(&viewport_size).cast(),
                mem::size_of_val(&viewport_size),
                EffectCompositeInputIndex::ViewportSize as usize,
            );
            command_encoder.setFragmentTexture_atIndex(
                Some(effect_texture),
                EffectCompositeInputIndex::Texture as usize,
            );
            command_encoder.drawPrimitives_vertexStart_vertexCount(
                MTLPrimitiveType::Triangle,
                0,
                6,
            );
        }
        Ok(())
    }

    /// Returns the compiled pass for `source`, compiling it on first use.
    /// Compile failures are cached as well, so a broken shader is reported
    /// once instead of being recompiled every frame.
    fn effect_pipeline(
        &mut self,
        source: &Arc<str>,
    ) -> Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
        let frame = self.effect_frame;
        if let Some(pipeline) = self.effect_pipelines.get_mut(source.as_ref()) {
            pipeline.last_used_frame = frame;
            return pipeline.state.clone();
        }

        let state = match self.compile_effect_pipeline(source) {
            Ok(state) => Some(state),
            Err(error) => {
                log::error!("failed to compile shader effect pass; disabling the chain: {error:#}");
                None
            }
        };
        self.effect_pipelines.insert(
            source.clone(),
            EffectPipeline {
                state: state.clone(),
                last_used_frame: frame,
            },
        );
        state
    }

    fn compile_effect_pipeline(
        &self,
        source: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
        let library = self
            .device
            .newLibraryWithSource_options_error(
                &NSString::from_str(source),
                Some(&MTLCompileOptions::new()),
            )
            .map_err(|error| anyhow::anyhow!("{}", error.localizedDescription()))
            .context("compiling Metal source")?;
        let fragment_function = library
            .newFunctionWithName(&NSString::from_str("main0"))
            .context("locating fragment function main0")?;

        let descriptor = MTLRenderPipelineDescriptor::new();
        descriptor.setLabel(Some(&NSString::from_str("shader_effect_pass")));
        descriptor.setVertexFunction(Some(&self.effect_pass_vertex_function));
        descriptor.setFragmentFunction(Some(&fragment_function));
        // SAFETY: Index zero is within the pipeline's color attachment array.
        let color_attachment = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) };
        color_attachment.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        // Each pass replaces its target, as Ghostty's custom shader passes do.
        color_attachment.setBlendingEnabled(false);

        self.device
            .newRenderPipelineStateWithDescriptor_error(&descriptor)
            .map_err(|error| anyhow::anyhow!("{}", error.localizedDescription()))
            .context("creating render pipeline")
    }

    /// Drops textures for effects not drawn this frame and bounds the compiled
    /// source cache across configuration changes.
    fn retire_effect_resources(&mut self) {
        let frame = self.effect_frame;
        self.effect_targets
            .retain(|_, target| target.last_used_frame == frame);
        // Keep ordinary pane switches from recompiling shaders. Only evict
        // when enough distinct configurations have accumulated to fill the cache.
        while self.effect_pipelines.len() > MAX_EFFECT_PIPELINE_CACHE_SIZE {
            let oldest = self
                .effect_pipelines
                .iter()
                .min_by_key(|(_, pipeline)| pipeline.last_used_frame)
                .map(|(source, _)| source.clone());
            if let Some(source) = oldest {
                self.effect_pipelines.remove(&source);
            } else {
                break;
            }
        }
    }
}

/// Offscreen textures for one effect id. The captured scene is drawn into the
/// first texture, then shader passes ping-pong between the two.
struct EffectTarget {
    size: Size<DevicePixels>,
    textures: [Retained<ProtocolObject<dyn MTLTexture>>; 2],
    /// Allocated only once the effect's scene contains paths.
    path_textures: PathIntermediateTextures,
    last_used_frame: u64,
}

impl EffectTarget {
    fn new(device: &ProtocolObject<dyn MTLDevice>, size: Size<DevicePixels>) -> Result<Self> {
        let texture_descriptor = MTLTextureDescriptor::new();
        // SAFETY: The dimensions are positive and this descriptor creates a 2D texture.
        unsafe {
            texture_descriptor.setWidth(size.width.0 as usize);
            texture_descriptor.setHeight(size.height.0 as usize);
        }
        texture_descriptor.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        texture_descriptor.setStorageMode(MTLStorageMode::Private);
        texture_descriptor.setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
        let texture = || {
            device
                .newTextureWithDescriptor(&texture_descriptor)
                .context("allocating shader effect texture")
        };
        Ok(Self {
            size,
            textures: [texture()?, texture()?],
            path_textures: PathIntermediateTextures::default(),
            last_used_frame: 0,
        })
    }
}

struct EffectPipeline {
    /// `None` when the source failed to compile.
    state: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    last_used_frame: u64,
}

/// The textures paths are rasterized into before being copied to their target.
/// They must match the target's size, because the path sprite shader maps
/// positions to texture coordinates through the viewport size.
#[derive(Clone, Default)]
pub(crate) struct PathIntermediateTextures {
    pub(crate) texture: Option<Retained<ProtocolObject<dyn MTLTexture>>>,
    pub(crate) msaa_texture: Option<Retained<ProtocolObject<dyn MTLTexture>>>,
}

fn new_path_intermediate_textures(
    device: &ProtocolObject<dyn MTLDevice>,
    size: Size<DevicePixels>,
    path_sample_count: u32,
    is_apple_gpu: bool,
) -> Result<PathIntermediateTextures> {
    let texture_descriptor = MTLTextureDescriptor::new();
    // SAFETY: The dimensions are positive and this descriptor creates a 2D texture.
    unsafe {
        texture_descriptor.setWidth(size.width.0 as usize);
        texture_descriptor.setHeight(size.height.0 as usize);
    }
    texture_descriptor.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
    texture_descriptor.setStorageMode(MTLStorageMode::Private);
    texture_descriptor.setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
    let texture = device
        .newTextureWithDescriptor(&texture_descriptor)
        .context("allocating path intermediate texture")?;

    let msaa_texture = if path_sample_count > 1 {
        // https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus
        // Rendering MSAA textures are done in a single pass, so we can use memory-less storage on Apple Silicon
        let storage_mode = if is_apple_gpu {
            MTLStorageMode::Memoryless
        } else {
            MTLStorageMode::Private
        };

        let msaa_descriptor = texture_descriptor;
        msaa_descriptor.setTextureType(MTLTextureType::Type2DMultisample);
        msaa_descriptor.setStorageMode(storage_mode);
        // SAFETY: The multisample texture uses the same sample count as its render pipeline.
        unsafe {
            msaa_descriptor.setSampleCount(path_sample_count as _);
        }
        Some(
            device
                .newTextureWithDescriptor(&msaa_descriptor)
                .context("allocating multisample path texture")?,
        )
    } else {
        None
    };

    Ok(PathIntermediateTextures {
        texture: Some(texture),
        msaa_texture,
    })
}

/// Copies `scene` with every primitive moved by `-origin`, so that an effect's
/// children land at the top-left of its texture.
///
/// Moving the geometry, rather than offsetting the viewport, is required:
/// fragment shaders compare `[[position]]` against primitive bounds for rounded
/// corners, borders, gradients and clipping.
fn scene_in_effect_space(scene: &Scene, origin: Point<ScaledPixels>) -> Scene {
    let offset_point = |position: Point<ScaledPixels>| Point {
        x: position.x - origin.x,
        y: position.y - origin.y,
    };
    let offset_bounds = |bounds: Bounds<ScaledPixels>| Bounds {
        origin: offset_point(bounds.origin),
        size: bounds.size,
    };
    let offset_mask = |mask: ContentMask<ScaledPixels>| ContentMask {
        bounds: offset_bounds(mask.bounds),
    };
    // Sprites are positioned by their transformation, which is applied after
    // their bounds, so only its translation moves.
    let offset_transformation = |mut transformation: gpui::TransformationMatrix| {
        transformation.translation[0] -= origin.x.0;
        transformation.translation[1] -= origin.y.0;
        transformation
    };

    let mut local = Scene::default();
    local.shadows = scene
        .shadows
        .iter()
        .map(|shadow| gpui::Shadow {
            bounds: offset_bounds(shadow.bounds),
            content_mask: offset_mask(shadow.content_mask),
            element_bounds: offset_bounds(shadow.element_bounds),
            ..*shadow
        })
        .collect();
    local.quads = scene
        .quads
        .iter()
        .map(|quad| gpui::Quad {
            bounds: offset_bounds(quad.bounds),
            content_mask: offset_mask(quad.content_mask),
            ..*quad
        })
        .collect();
    local.paths = scene
        .paths
        .iter()
        .map(|path| {
            let mut path = path.clone();
            path.bounds = offset_bounds(path.bounds);
            path.content_mask = offset_mask(path.content_mask);
            for vertex in &mut path.vertices {
                vertex.xy_position = offset_point(vertex.xy_position);
                vertex.content_mask = offset_mask(vertex.content_mask);
            }
            path
        })
        .collect();
    local.underlines = scene
        .underlines
        .iter()
        .map(|underline| gpui::Underline {
            bounds: offset_bounds(underline.bounds),
            content_mask: offset_mask(underline.content_mask),
            ..*underline
        })
        .collect();
    local.monochrome_sprites = scene
        .monochrome_sprites
        .iter()
        .map(|sprite| gpui::MonochromeSprite {
            content_mask: offset_mask(sprite.content_mask),
            transformation: offset_transformation(sprite.transformation),
            ..*sprite
        })
        .collect();
    local.subpixel_sprites = scene
        .subpixel_sprites
        .iter()
        .map(|sprite| gpui::SubpixelSprite {
            content_mask: offset_mask(sprite.content_mask),
            transformation: offset_transformation(sprite.transformation),
            ..*sprite
        })
        .collect();
    local.polychrome_sprites = scene
        .polychrome_sprites
        .iter()
        .map(|sprite| gpui::PolychromeSprite {
            bounds: offset_bounds(sprite.bounds),
            content_mask: offset_mask(sprite.content_mask),
            ..*sprite
        })
        .collect();
    local.surfaces = scene
        .surfaces
        .iter()
        .map(|surface| PaintSurface {
            bounds: offset_bounds(surface.bounds),
            content_mask: offset_mask(surface.content_mask),
            ..surface.clone()
        })
        .collect();
    local
}

fn write_padded_bytes(
    writer: &mut InstanceBufferWriter,
    bytes: &[u8],
    minimum_len: usize,
) -> Result<InstanceBinding> {
    let (binding, destination) = writer.allocate::<u8>(bytes.len().max(minimum_len))?;
    for (slot, byte) in destination
        .iter_mut()
        .zip(bytes.iter().copied().chain(iter::repeat(0)))
    {
        slot.write(byte);
    }
    Ok(binding)
}

pub(crate) fn new_command_encoder_for_texture(
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    texture: &ProtocolObject<dyn MTLTexture>,
    viewport_size: Size<DevicePixels>,
    clear_color: Option<MTLClearColor>,
) -> Result<Retained<ProtocolObject<dyn MTLRenderCommandEncoder>>> {
    let render_pass_descriptor = MTLRenderPassDescriptor::new();
    // SAFETY: Index zero is within the render pass's color attachment array.
    let color_attachment = unsafe {
        render_pass_descriptor
            .colorAttachments()
            .objectAtIndexedSubscript(0)
    };
    color_attachment.setTexture(Some(texture));
    color_attachment.setStoreAction(MTLStoreAction::Store);
    if let Some(clear_color) = clear_color {
        color_attachment.setLoadAction(MTLLoadAction::Clear);
        color_attachment.setClearColor(clear_color);
    } else {
        color_attachment.setLoadAction(MTLLoadAction::Load);
    }

    let command_encoder = command_buffer
        .renderCommandEncoderWithDescriptor(&render_pass_descriptor)
        .context("creating render command encoder")?;
    command_encoder.setViewport(MTLViewport {
        originX: 0.0,
        originY: 0.0,
        width: i32::from(viewport_size.width) as f64,
        height: i32::from(viewport_size.height) as f64,
        znear: 0.0,
        zfar: 1.0,
    });
    crate::fast::rounded_clip::default_mask(&command_encoder);
    Ok(command_encoder)
}

#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
fn read_texture_to_image(texture: &ProtocolObject<dyn MTLTexture>) -> Result<RgbaImage> {
    let width = texture.width() as u32;
    let height = texture.height() as u32;
    let bytes_per_row = width as usize * 4;
    let mut pixels = vec![0u8; height as usize * bytes_per_row];

    let region = MTLRegion {
        origin: MTLOrigin { x: 0, y: 0, z: 0 },
        size: MTLSize {
            width: width as usize,
            height: height as usize,
            depth: 1,
        },
    };
    unsafe {
        texture.getBytes_bytesPerRow_fromRegion_mipmapLevel(
            ptr::NonNull::from(pixels.as_mut_slice()).cast(),
            bytes_per_row,
            region,
            0,
        );
    }

    // Convert BGRA to RGBA (swap B and R channels)
    for chunk in pixels.chunks_exact_mut(4) {
        chunk.swap(0, 2);
    }

    RgbaImage::from_raw(width, height, pixels).context("creating RGBA image from pixel data")
}

fn build_pipeline_state(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: MTLPixelFormat,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
    let vertex_fn = library
        .newFunctionWithName(&NSString::from_str(vertex_fn_name))
        .with_context(|| format!("locating vertex function {vertex_fn_name}"))?;
    let fragment_fn = library
        .newFunctionWithName(&NSString::from_str(fragment_fn_name))
        .with_context(|| format!("locating fragment function {fragment_fn_name}"))?;

    let descriptor = MTLRenderPipelineDescriptor::new();
    descriptor.setLabel(Some(&NSString::from_str(label)));
    descriptor.setVertexFunction(Some(vertex_fn.as_ref()));
    descriptor.setFragmentFunction(Some(fragment_fn.as_ref()));
    // SAFETY: Index zero is within the pipeline's color attachment array.
    let color_attachment = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) };
    color_attachment.setPixelFormat(pixel_format);
    color_attachment.setBlendingEnabled(true);
    color_attachment.setRgbBlendOperation(MTLBlendOperation::Add);
    color_attachment.setAlphaBlendOperation(MTLBlendOperation::Add);
    color_attachment.setSourceRGBBlendFactor(MTLBlendFactor::SourceAlpha);
    color_attachment.setSourceAlphaBlendFactor(MTLBlendFactor::One);
    color_attachment.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
    // CDXC:PlatformSupport 2026-09-19 WHY:
    // Alpha composites "over" like the colour, as wgpu's `ALPHA_BLENDING` does on Linux. Adding it (`One`) summed the coverage of every primitive sharing an anti-aliased edge, and a div paints its background and its border as two quads with the same edge, so a transparent window's rounded corners came out nearly opaque with only the colour of a half-covered pixel, which macOS composited as a dark dotted rim on light backgrounds. Opaque windows ignore this channel.
    color_attachment.setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);

    device
        .newRenderPipelineStateWithDescriptor_error(&descriptor)
        .with_context(|| format!("creating {label} render pipeline"))
}

fn build_path_sprite_pipeline_state(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: MTLPixelFormat,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
    let vertex_fn = library
        .newFunctionWithName(&NSString::from_str(vertex_fn_name))
        .with_context(|| format!("locating vertex function {vertex_fn_name}"))?;
    let fragment_fn = library
        .newFunctionWithName(&NSString::from_str(fragment_fn_name))
        .with_context(|| format!("locating fragment function {fragment_fn_name}"))?;

    let descriptor = MTLRenderPipelineDescriptor::new();
    descriptor.setLabel(Some(&NSString::from_str(label)));
    descriptor.setVertexFunction(Some(vertex_fn.as_ref()));
    descriptor.setFragmentFunction(Some(fragment_fn.as_ref()));
    // SAFETY: Index zero is within the pipeline's color attachment array.
    let color_attachment = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) };
    color_attachment.setPixelFormat(pixel_format);
    color_attachment.setBlendingEnabled(true);
    color_attachment.setRgbBlendOperation(MTLBlendOperation::Add);
    color_attachment.setAlphaBlendOperation(MTLBlendOperation::Add);
    color_attachment.setSourceRGBBlendFactor(MTLBlendFactor::One);
    color_attachment.setSourceAlphaBlendFactor(MTLBlendFactor::One);
    color_attachment.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
    // Alpha must accumulate as `src.a + dst.a * (1 - src.a)` like the other pipelines. An
    // additive `One` saturates to opaque wherever a path's antialiased edge lands on an already
    // translucent pixel, and on a transparent window that punches a dark fringe into the blur.
    color_attachment.setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);

    device
        .newRenderPipelineStateWithDescriptor_error(&descriptor)
        .with_context(|| format!("creating {label} render pipeline"))
}

fn build_path_rasterization_pipeline_state(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: MTLPixelFormat,
    path_sample_count: u32,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
    let vertex_fn = library
        .newFunctionWithName(&NSString::from_str(vertex_fn_name))
        .with_context(|| format!("locating vertex function {vertex_fn_name}"))?;
    let fragment_fn = library
        .newFunctionWithName(&NSString::from_str(fragment_fn_name))
        .with_context(|| format!("locating fragment function {fragment_fn_name}"))?;

    let descriptor = MTLRenderPipelineDescriptor::new();
    descriptor.setLabel(Some(&NSString::from_str(label)));
    descriptor.setVertexFunction(Some(vertex_fn.as_ref()));
    descriptor.setFragmentFunction(Some(fragment_fn.as_ref()));
    if path_sample_count > 1 {
        descriptor.setRasterSampleCount(path_sample_count as _);
        descriptor.setAlphaToCoverageEnabled(false);
    }
    // SAFETY: Index zero is within the pipeline's color attachment array.
    let color_attachment = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) };
    color_attachment.setPixelFormat(pixel_format);
    color_attachment.setBlendingEnabled(true);
    color_attachment.setRgbBlendOperation(MTLBlendOperation::Add);
    color_attachment.setAlphaBlendOperation(MTLBlendOperation::Add);
    color_attachment.setSourceRGBBlendFactor(MTLBlendFactor::One);
    color_attachment.setSourceAlphaBlendFactor(MTLBlendFactor::One);
    color_attachment.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);

    device
        .newRenderPipelineStateWithDescriptor_error(&descriptor)
        .with_context(|| format!("creating {label} render pipeline"))
}

#[derive(Clone)]
pub(crate) struct InstanceBinding {
    pub(crate) buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(crate) offset: usize,
}

pub(crate) struct InstanceBindings {
    pub(crate) quads: InstanceBinding,
    pub(crate) shadows: InstanceBinding,
    pub(crate) underlines: InstanceBinding,
    pub(crate) monochrome_sprites: InstanceBinding,
    pub(crate) polychrome_sprites: InstanceBinding,
    pub(crate) surfaces: InstanceBinding,
}

pub(crate) fn write_instances(
    scene: &Scene,
    writer: &mut InstanceBufferWriter,
) -> Result<InstanceBindings> {
    Ok(InstanceBindings {
        quads: writer.write(&scene.quads)?,
        shadows: writer.write(&scene.shadows)?,
        underlines: writer.write(&scene.underlines)?,
        monochrome_sprites: writer.write(&scene.monochrome_sprites)?,
        polychrome_sprites: writer.write(&scene.polychrome_sprites)?,
        surfaces: writer.write_iter(scene.surfaces.iter().map(|surface| SurfaceBounds {
            bounds: surface.bounds,
            content_mask: surface.content_mask,
        }))?,
    })
}

pub(crate) struct InstanceBufferWriter {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    pool: Arc<Mutex<InstanceBufferPool>>,
    unified_memory: bool,
    filled: Vec<(InstanceBuffer, usize)>,
    current: InstanceBuffer,
    offset: usize,
}

impl InstanceBufferWriter {
    pub(crate) fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        pool: &Arc<Mutex<InstanceBufferPool>>,
        unified_memory: bool,
    ) -> Result<Self> {
        let current = pool.lock().acquire(device, unified_memory)?;
        Ok(Self {
            device: device.retain(),
            pool: pool.clone(),
            unified_memory,
            filled: Vec::new(),
            current,
            offset: 0,
        })
    }

    fn allocate<T>(&mut self, count: usize) -> Result<(InstanceBinding, &mut [MaybeUninit<T>])> {
        let size = mem::size_of::<T>() * count;
        let mut offset = self.offset.next_multiple_of(INSTANCE_BUFFER_ALIGNMENT);
        if offset + size > self.current.size {
            self.grow(size)?;
            offset = 0;
        }
        self.offset = offset + size;

        let binding = InstanceBinding {
            buffer: self.current.metal_buffer.clone(),
            offset,
        };
        // SAFETY: The aligned reservation lies within this frame's exclusively owned buffer and does not overlap prior writes.
        let values = unsafe {
            let start = self
                .current
                .metal_buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(offset);
            slice::from_raw_parts_mut(start.cast::<MaybeUninit<T>>(), count)
        };
        Ok((binding, values))
    }

    pub(crate) fn write<T>(&mut self, values: &[T]) -> Result<InstanceBinding> {
        let (binding, destination) = self.allocate::<T>(values.len())?;
        // SAFETY: The disjoint source and destination hold the same number of T values, and the destination stays MaybeUninit.
        unsafe {
            ptr::copy_nonoverlapping(
                values.as_ptr(),
                destination.as_mut_ptr().cast::<T>(),
                values.len(),
            );
        }
        Ok(binding)
    }

    pub(crate) fn write_iter<T>(
        &mut self,
        values: impl ExactSizeIterator<Item = T>,
    ) -> Result<InstanceBinding> {
        let (binding, destination) = self.allocate::<T>(values.len())?;
        for (slot, value) in destination.iter_mut().zip(values) {
            slot.write(value);
        }
        Ok(binding)
    }

    fn grow(&mut self, required: usize) -> Result<()> {
        let mut pool = self.pool.lock();
        let buffer_size = (pool.buffer_size * 2)
            .max(required.next_power_of_two())
            .min(MAX_INSTANCE_BUFFER_SIZE);
        anyhow::ensure!(
            buffer_size >= required,
            "instance buffer needs {required} bytes, above the maximum of {MAX_INSTANCE_BUFFER_SIZE}"
        );
        anyhow::ensure!(
            buffer_size > self.current.size,
            "frame instance data exceeds the {MAX_INSTANCE_BUFFER_SIZE}-byte maximum"
        );
        if buffer_size != pool.buffer_size {
            log::info!("increased instance buffer size to {buffer_size}");
            pool.reset(buffer_size);
        }
        let buffer = pool.acquire(&self.device, self.unified_memory)?;
        drop(pool);

        let filled = mem::replace(&mut self.current, buffer);
        self.filled.push((filled, self.offset));
        self.offset = 0;
        Ok(())
    }

    pub(crate) fn finish(self) -> InstanceBuffer {
        let Self {
            unified_memory,
            filled,
            current,
            offset,
            ..
        } = self;

        if !unified_memory {
            for (buffer, written) in &filled {
                if *written == 0 {
                    continue;
                }
                buffer.metal_buffer.didModifyRange(NSRange {
                    location: 0,
                    length: *written,
                });
            }
            if offset > 0 {
                current.metal_buffer.didModifyRange(NSRange {
                    location: 0,
                    length: offset,
                });
            }
        }

        // Metal retains encoded resources until the command buffer completes.
        // Only the final, largest buffer is worth keeping in the pool.
        drop(filled);
        current
    }
}

#[repr(C)]
enum ShadowInputIndex {
    Vertices = 0,
    Shadows = 1,
    ViewportSize = 2,
}

#[repr(C)]
enum QuadInputIndex {
    Vertices = 0,
    Quads = 1,
    ViewportSize = 2,
}

#[repr(C)]
enum UnderlineInputIndex {
    Vertices = 0,
    Underlines = 1,
    ViewportSize = 2,
}

#[repr(C)]
pub(crate) enum SpriteInputIndex {
    Vertices = 0,
    Sprites = 1,
    ViewportSize = 2,
    AtlasTextureSize = 3,
    AtlasTexture = 4,
}

#[repr(C)]
enum SurfaceInputIndex {
    Vertices = 0,
    Surfaces = 1,
    ViewportSize = 2,
    TextureSize = 3,
    YTexture = 4,
    CbCrTexture = 5,
}

#[repr(C)]
pub(crate) enum PathRasterizationInputIndex {
    Vertices = 0,
    ViewportSize = 1,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct PathSprite {
    pub bounds: Bounds<ScaledPixels>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct SurfaceBounds {
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
}

#[repr(C)]
enum EffectCompositeInputIndex {
    Vertices = 0,
    Bounds = 1,
    ViewportSize = 2,
    Texture = 3,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct EffectCompositeBounds {
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
}

#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
pub struct MetalHeadlessRenderer {
    renderer: MetalRenderer,
}

#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
impl MetalHeadlessRenderer {
    pub fn new() -> Result<Self> {
        let instance_buffer_pool = Arc::new(Mutex::new(InstanceBufferPool::default()));
        let renderer = MetalRenderer::new_headless(instance_buffer_pool)?;
        Ok(Self { renderer })
    }
}

#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
impl gpui::PlatformHeadlessRenderer for MetalHeadlessRenderer {
    fn render_scene_to_image(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
    ) -> anyhow::Result<image::RgbaImage> {
        self.renderer.render_scene_to_image(scene, size)
    }

    fn render_scene(&mut self, scene: &Scene, size: Size<DevicePixels>) -> anyhow::Result<()> {
        self.renderer.render_scene(scene, size)
    }

    fn sprite_atlas(&self) -> Arc<dyn gpui::PlatformAtlas> {
        self.renderer.sprite_atlas().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_core_foundation::{CFDictionary, CFType};
    use objc2_core_video::{
        CVPixelBufferCreate, CVPixelBufferGetBaseAddressOfPlane,
        CVPixelBufferGetBytesPerRowOfPlane, CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags,
        CVPixelBufferUnlockBaseAddress, kCVPixelBufferIOSurfacePropertiesKey,
    };
    use objc2_metal::MTLCommandBufferStatus;

    #[test]
    fn render_core_video_surface() {
        let surface_properties_key = unsafe { kCVPixelBufferIOSurfacePropertiesKey };
        let attributes = CFDictionary::from_slices(
            &[surface_properties_key],
            &[&*CFDictionary::<CFType, CFType>::empty()],
        );
        let mut image_buffer = ptr::null_mut();
        assert_eq!(
            unsafe {
                CVPixelBufferCreate(
                    None,
                    16,
                    16,
                    kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                    Some(attributes.as_opaque()),
                    ptr::NonNull::from(&mut image_buffer),
                )
            },
            kCVReturnSuccess
        );
        let image_buffer = ptr::NonNull::new(image_buffer).unwrap();
        let image_buffer = unsafe { CFRetained::from_raw(image_buffer) };
        unsafe {
            assert_eq!(
                CVPixelBufferLockBaseAddress(&image_buffer, CVPixelBufferLockFlags::empty()),
                kCVReturnSuccess
            );
            for plane in 0..2 {
                let bytes = CVPixelBufferGetBaseAddressOfPlane(&image_buffer, plane).cast::<u8>();
                let length = CVPixelBufferGetBytesPerRowOfPlane(&image_buffer, plane)
                    * CVPixelBufferGetHeightOfPlane(&image_buffer, plane);
                slice::from_raw_parts_mut(bytes, length).fill(128);
            }
            assert_eq!(
                CVPixelBufferUnlockBaseAddress(&image_buffer, CVPixelBufferLockFlags::empty()),
                kCVReturnSuccess
            );
        }

        let bounds = Bounds {
            origin: Point::default(),
            size: size(ScaledPixels(16.0), ScaledPixels(16.0)),
        };
        let mut scene = Scene::default();
        scene.insert_primitive(PaintSurface {
            order: 0,
            bounds,
            content_mask: ContentMask { bounds },
            image_buffer: image_buffer.clone(),
        });
        drop(image_buffer);
        scene.finish();

        let mut renderer =
            MetalRenderer::new_headless(Arc::new(Mutex::new(InstanceBufferPool::default())))
                .unwrap();
        let image = renderer
            .render_scene_to_image(&scene, size(DevicePixels(16), DevicePixels(16)))
            .unwrap();
        for image::Rgba([red, green, blue, alpha]) in image.pixels() {
            assert!(red.abs_diff(128) <= 2);
            assert!(green.abs_diff(128) <= 2);
            assert!(blue.abs_diff(128) <= 2);
            assert_eq!(*alpha, 255);
        }

        let descriptor = MTLTextureDescriptor::new();
        unsafe {
            descriptor.setWidth(16);
            descriptor.setHeight(16);
        }
        descriptor.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        descriptor.setUsage(MTLTextureUsage::RenderTarget);
        let uses_shared_storage = cfg!(target_os = "ios") || renderer.is_apple_gpu;
        descriptor.setStorageMode(if uses_shared_storage {
            MTLStorageMode::Shared
        } else {
            MTLStorageMode::Managed
        });
        let target_texture = renderer
            .device
            .newTextureWithDescriptor(&descriptor)
            .unwrap();
        let command_buffer = renderer
            .render_frame(
                &scene,
                &target_texture,
                size(DevicePixels(16), DevicePixels(16)),
            )
            .unwrap();
        if !uses_shared_storage {
            let blit = command_buffer.blitCommandEncoder().unwrap();
            blit.synchronizeResource(ProtocolObject::from_ref(&*target_texture));
            blit.endEncoding();
        }
        drop(scene);
        drop(renderer);
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);
        assert_eq!(read_texture_to_image(&target_texture).unwrap(), image);
    }
}
