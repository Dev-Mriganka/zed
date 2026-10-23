use crate::metal_atlas::MetalAtlas;
use anyhow::{Context as _, Result};
use block2::RcBlock;
use core_graphics::geometry::CGSize;
use gpui::{
    AtlasTextureId, Background, Bounds, ContentMask, DevicePixels, PaintEffect, PaintSurface, Path,
    Point, PrimitiveBatch, ScaledPixels, Scene, SceneBatch, Size, point, size,
};
#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
use image::RgbaImage;
use objc2::runtime::AnyObject;

use core_foundation::base::TCFType;
use core_video::{
    metal_texture::CVMetalTextureGetTexture, metal_texture_cache::CVMetalTextureCache,
    pixel_buffer::kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
};
use foreign_types::{ForeignType, ForeignTypeRef};
use metal::{
    CAMetalLayer, CommandQueue, MTLGPUFamily, MTLPixelFormat, MTLResourceOptions, NSRange,
    NSUInteger,
};
use parking_lot::Mutex;

use std::{
    cell::Cell, collections::HashMap, ffi::c_void, iter, mem, mem::MaybeUninit, ops::Range, ptr,
    slice, sync::Arc,
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

pub unsafe fn new_renderer(
    context: self::Context,
    _native_window: *mut c_void,
    _native_view: *mut c_void,
    _bounds: gpui::Size<f32>,
    transparent: bool,
) -> Renderer {
    MetalRenderer::new(context, transparent)
}

pub fn new_overlay_renderer(context: self::Context, base: &Renderer) -> Renderer {
    crate::fast::composition::new_overlay_renderer(context, base)
}

pub struct InstanceBufferPool {
    buffer_size: usize,
    buffers: Vec<metal::Buffer>,
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
    metal_buffer: metal::Buffer,
    size: usize,
}

impl InstanceBufferPool {
    pub(crate) fn reset(&mut self, buffer_size: usize) {
        self.buffer_size = buffer_size;
        self.buffers.clear();
    }

    pub(crate) fn acquire(
        &mut self,
        device: &metal::Device,
        unified_memory: bool,
    ) -> InstanceBuffer {
        let buffer = self.buffers.pop().unwrap_or_else(|| {
            let options = if unified_memory {
                MTLResourceOptions::StorageModeShared
                    // Buffers are write only which can benefit from the combined cache
                    // https://developer.apple.com/documentation/metal/mtlresourceoptions/cpucachemodewritecombined
                    | MTLResourceOptions::CPUCacheModeWriteCombined
            } else {
                MTLResourceOptions::StorageModeManaged
            };

            device.new_buffer(self.buffer_size as u64, options)
        });
        InstanceBuffer {
            metal_buffer: buffer,
            size: self.buffer_size,
        }
    }

    pub(crate) fn release(&mut self, buffer: InstanceBuffer) {
        if buffer.size == self.buffer_size {
            self.buffers.push(buffer.metal_buffer)
        }
    }
}

pub struct MetalRenderer {
    pub(crate) device: metal::Device,
    layer: Option<metal::MetalLayer>,
    is_apple_gpu: bool,
    pub(crate) is_unified_memory: bool,
    presents_with_transaction: bool,
    /// For headless rendering, tracks whether output should be opaque
    pub(crate) opaque: bool,
    pub(crate) command_queue: CommandQueue,
    pub(crate) paths_rasterization_pipeline_state: metal::RenderPipelineState,
    path_sprites_pipeline_state: metal::RenderPipelineState,
    shadows_pipeline_state: metal::RenderPipelineState,
    quads_pipeline_state: metal::RenderPipelineState,
    underlines_pipeline_state: metal::RenderPipelineState,
    monochrome_sprites_pipeline_state: metal::RenderPipelineState,
    pub(crate) polychrome_sprites_pipeline_state: metal::RenderPipelineState,
    surfaces_pipeline_state: metal::RenderPipelineState,
    pub(crate) unit_vertices: metal::Buffer,
    #[allow(clippy::arc_with_non_send_sync)]
    pub(crate) instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
    pub(crate) sprite_atlas: Arc<MetalAtlas>,
    core_video_texture_cache: core_video::metal_texture_cache::CVMetalTextureCache,
    pub(crate) path_intermediate_texture: Option<metal::Texture>,
    pub(crate) path_intermediate_msaa_texture: Option<metal::Texture>,
    path_sample_count: u32,
    effect_composite_pipeline_state: metal::RenderPipelineState,
    effect_pass_vertex_function: metal::Function,
    effect_sampler: metal::SamplerState,
    effect_targets: HashMap<u64, EffectTarget>,
    effect_pipelines: HashMap<Arc<str>, EffectPipeline>,
    effect_frame: u64,
    /// Offscreen render target reused across `render_scene` calls when
    /// rendering headlessly without reading pixels back.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    headless_render_target: Option<metal::Texture>,
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
    pub fn new(instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>, transparent: bool) -> Self {
        let device = Self::create_device();
        let layer = metal::MetalLayer::new();
        Self::configure_layer(&layer, &device, transparent);
        Self::new_internal(device, Some(layer), !transparent, instance_buffer_pool)
    }

    /// Creates a renderer for a CAMetalLayer owned by a platform view, such as
    /// the backing layer UIKit creates for a view whose `layerClass` is
    /// `CAMetalLayer`. The renderer retains the layer.
    pub fn from_layer(
        instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
        layer: &objc2_quartz_core::CAMetalLayer,
        transparent: bool,
    ) -> Self {
        let device = Self::create_device();
        // Both types bind the same Objective-C class, so this only changes which
        // Rust wrapper views the live layer. `to_owned` retains it.
        let layer = unsafe {
            metal::MetalLayerRef::from_ptr(ptr::from_ref(layer).cast_mut().cast::<CAMetalLayer>())
        }
        .to_owned();
        Self::configure_layer(&layer, &device, transparent);
        Self::new_internal(device, Some(layer), !transparent, instance_buffer_pool)
    }

    pub(crate) fn configure_layer(layer: &metal::MetalLayerRef, device: &metal::DeviceRef, transparent: bool) {
        layer.set_device(device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        // Support direct-to-display rendering if the window is not transparent
        // https://developer.apple.com/documentation/metal/managing-your-game-window-for-metal-in-macos
        layer.set_opaque(!transparent);
        layer.set_maximum_drawable_count(3);
        // Allow texture reading for visual tests and UI automation screenshots
        // (captures without ScreenCaptureKit). Only debug builds pay the
        // presentation cost, even when `test-support` is compiled in.
        #[cfg(all(feature = "test-support", debug_assertions))]
        layer.set_framebuffer_only(false);
        // metal-rs doesn't bind these setters, so view the same object through
        // objc2's typed CAMetalLayer binding.
        let objc2_layer: &objc2_quartz_core::CAMetalLayer = unsafe { &*layer.as_ptr().cast() };
        objc2_layer.setAllowsNextDrawableTimeout(false);
        objc2_layer.setNeedsDisplayOnBoundsChange(true);
        // UIKit sizes a view's backing layer itself; only AppKit-hosted
        // layers need to track their superlayer's bounds.
        #[cfg(target_os = "macos")]
        objc2_layer.setAutoresizingMask(
            objc2_quartz_core::CAAutoresizingMask::LayerWidthSizable
                | objc2_quartz_core::CAAutoresizingMask::LayerHeightSizable,
        );
    }

    /// Creates a new headless MetalRenderer for offscreen rendering without a window.
    ///
    /// This renderer can render scenes to images without requiring a CAMetalLayer,
    /// window, or AppKit. Use `render_scene_to_image()` to render scenes.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    pub fn new_headless(instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>) -> Self {
        let device = Self::create_device();
        Self::new_internal(device, None, true, instance_buffer_pool)
    }

    #[cfg(target_os = "macos")]
    fn create_device() -> metal::Device {
        // Prefer low‐power integrated GPUs on Intel Mac. On Apple
        // Silicon, there is only ever one GPU, so this is equivalent to
        // `metal::Device::system_default()`.
        if let Some(d) = metal::Device::all()
            .into_iter()
            .min_by_key(|d| (d.is_removable(), !d.is_low_power()))
        {
            d
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
    fn create_device() -> metal::Device {
        Self::system_default_device()
    }

    fn system_default_device() -> metal::Device {
        metal::Device::system_default().unwrap_or_else(|| {
            log::error!("unable to access a compatible graphics device");
            std::process::exit(1);
        })
    }

    pub(crate) fn new_internal(
        device: metal::Device,
        layer: Option<metal::MetalLayer>,
        opaque: bool,
        instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
    ) -> Self {
        #[cfg(feature = "runtime_shaders")]
        let library = device
            .new_library_with_source(&SHADERS_SOURCE_FILE, &metal::CompileOptions::new())
            .expect("error building metal library");
        #[cfg(not(feature = "runtime_shaders"))]
        let library = device
            .new_library_with_data(SHADERS_METALLIB)
            .expect("error building metal library");

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
        let is_unified_memory = cfg!(target_os = "ios") || device.has_unified_memory();
        // Apple GPU families support memoryless textures, which can significantly reduce
        // memory usage by keeping render targets in on-chip tile memory instead of
        // allocating backing store in system memory.
        // https://developer.apple.com/documentation/metal/mtlgpufamily
        let is_apple_gpu = device.supports_family(MTLGPUFamily::Apple1);

        let unit_vertices = [
            to_float2_bits(point(0., 0.)),
            to_float2_bits(point(1., 0.)),
            to_float2_bits(point(0., 1.)),
            to_float2_bits(point(0., 1.)),
            to_float2_bits(point(1., 0.)),
            to_float2_bits(point(1., 1.)),
        ];
        let unit_vertices = device.new_buffer_with_data(
            unit_vertices.as_ptr() as *const c_void,
            mem::size_of_val(&unit_vertices) as u64,
            if is_unified_memory {
                MTLResourceOptions::StorageModeShared
                    | MTLResourceOptions::CPUCacheModeWriteCombined
            } else {
                MTLResourceOptions::StorageModeManaged
            },
        );

        let paths_rasterization_pipeline_state = build_path_rasterization_pipeline_state(
            &device,
            &library,
            "paths_rasterization",
            "path_rasterization_vertex",
            "path_rasterization_fragment",
            MTLPixelFormat::BGRA8Unorm,
            PATH_SAMPLE_COUNT,
        );
        let path_sprites_pipeline_state = build_path_sprite_pipeline_state(
            &device,
            &library,
            "path_sprites",
            "path_sprite_vertex",
            "path_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let shadows_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "shadows",
            "shadow_vertex",
            "shadow_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let quads_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "quads",
            "quad_vertex",
            "quad_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let underlines_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "underlines",
            "underline_vertex",
            "underline_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let monochrome_sprites_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "monochrome_sprites",
            "monochrome_sprite_vertex",
            "monochrome_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let polychrome_sprites_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "polychrome_sprites",
            "polychrome_sprite_vertex",
            "polychrome_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let surfaces_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "surfaces",
            "surface_vertex",
            "surface_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        // Offscreen effect textures hold premultiplied colour, so they composite
        // with the same blend state as path sprites.
        let effect_composite_pipeline_state = build_path_sprite_pipeline_state(
            &device,
            &library,
            "effect_composite",
            "effect_composite_vertex",
            "effect_composite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let effect_pass_vertex_function = library
            .get_function("effect_pass_vertex", None)
            .expect("error locating vertex function");
        let effect_sampler = {
            // Shadertoy's defaults, which Ghostty's shaders are written against.
            let descriptor = metal::SamplerDescriptor::new();
            descriptor.set_min_filter(metal::MTLSamplerMinMagFilter::Linear);
            descriptor.set_mag_filter(metal::MTLSamplerMinMagFilter::Linear);
            descriptor.set_address_mode_s(metal::MTLSamplerAddressMode::ClampToEdge);
            descriptor.set_address_mode_t(metal::MTLSamplerAddressMode::ClampToEdge);
            device.new_sampler(&descriptor)
        };

        let command_queue = device.new_command_queue();
        let supports_shared_storage = cfg!(target_os = "ios") || is_apple_gpu;
        let sprite_atlas = Arc::new(MetalAtlas::new(device.clone(), supports_shared_storage));
        let core_video_texture_cache =
            CVMetalTextureCache::new(None, device.clone(), None).unwrap();

        Self {
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
        }
    }

    pub fn layer(&self) -> Option<&metal::MetalLayerRef> {
        self.layer.as_ref().map(|l| l.as_ref())
    }

    pub fn layer_ptr(&self) -> *mut CAMetalLayer {
        self.layer
            .as_ref()
            .map(|l| l.as_ptr())
            .unwrap_or(ptr::null_mut())
    }

    pub fn sprite_atlas(&self) -> &Arc<MetalAtlas> {
        &self.sprite_atlas
    }

    pub fn set_presents_with_transaction(&mut self, presents_with_transaction: bool) {
        self.presents_with_transaction = presents_with_transaction;
        if let Some(layer) = &self.layer {
            layer.set_presents_with_transaction(presents_with_transaction);
        }
    }

    /// Ghostex: the path textures are not reallocated here. A live resize calls this for every
    /// frame and most frames draw no path, so the textures are dropped and the next frame that
    /// draws a path allocates them at its own size (`ensure_window_path_textures`).
    pub fn update_drawable_size(&mut self, size: Size<DevicePixels>) {
        if let Some(layer) = &self.layer {
            layer.set_drawable_size(CGSize::new(size.width.0 as f64, size.height.0 as f64));
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
    pub(crate) fn ensure_window_path_textures(&mut self, size: Size<DevicePixels>) {
        let current = self
            .path_intermediate_texture
            .as_ref()
            .is_some_and(|texture| {
                texture.width() == size.width.0 as u64 && texture.height() == size.height.0 as u64
            });
        if !current {
            self.update_path_intermediate_textures(size);
        }
    }

    /// The window-sized textures paths drawn straight into the window, or
    /// into a texture of its size, are rasterized through.
    pub(crate) fn window_path_textures(&self) -> PathIntermediateTextures {
        PathIntermediateTextures {
            texture: self.path_intermediate_texture.clone(),
            msaa_texture: self.path_intermediate_msaa_texture.clone(),
        }
    }

    fn update_path_intermediate_textures(&mut self, size: Size<DevicePixels>) {
        // We are uncertain when this happens, but sometimes size can be 0 here. Most likely before
        // the layout pass on window creation. Zero-sized texture creation causes SIGABRT.
        // https://github.com/zed-industries/zed/issues/36229
        if size.width.0 <= 0 || size.height.0 <= 0 {
            self.path_intermediate_texture = None;
            self.path_intermediate_msaa_texture = None;
            return;
        }

        let textures = new_path_intermediate_textures(
            &self.device,
            size,
            self.path_sample_count,
            self.is_apple_gpu,
        );
        self.path_intermediate_texture = textures.texture;
        self.path_intermediate_msaa_texture = textures.msaa_texture;
    }

    pub fn update_transparency(&mut self, transparent: bool) {
        self.opaque = !transparent;
        if let Some(layer) = &self.layer {
            layer.set_opaque(!transparent);
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
        let viewport_size = layer.drawable_size();
        let viewport_size: Size<DevicePixels> = size(
            (viewport_size.width.ceil() as i32).into(),
            (viewport_size.height.ceil() as i32).into(),
        );
        let drawable = if let Some(drawable) = layer.next_drawable() {
            drawable
        } else {
            log::error!(
                "failed to retrieve next drawable, drawable size: {:?}",
                viewport_size
            );
            return;
        };

        let command_buffer = match self.render_frame(scene, drawable.texture(), viewport_size) {
            Ok(command_buffer) => command_buffer,
            Err(error) => {
                log::error!("failed to render: {error:#}");
                return;
            }
        };

        if self.presents_with_transaction {
            command_buffer.commit();
            command_buffer.wait_until_scheduled();
            drawable.present();
        } else {
            command_buffer.present_drawable(drawable);
            command_buffer.commit();
        }
    }

    fn render_frame(
        &mut self,
        scene: &Scene,
        texture: &metal::TextureRef,
        viewport_size: Size<DevicePixels>,
    ) -> Result<metal::CommandBuffer> {
        crate::fast::layers::raster::rasterize_tiles(self, scene);
        let mut writer = InstanceBufferWriter::new(
            &self.device,
            &self.instance_buffer_pool,
            self.is_unified_memory,
        );
        let instance_bindings = write_instances(scene, &mut writer).with_context(|| {
            format!(
                "scene too large: {} paths, {} shadows, {} quads, {} underlines, {} mono, {} poly, {} surfaces",
                scene.paths.len(),
                scene.shadows.len(),
                scene.quads.len(),
                scene.underlines.len(),
                scene.monochrome_sprites.len(),
                scene.polychrome_sprites.len(),
                scene.surfaces.len(),
            )
        })?;
        self.effect_frame += 1;
        let command_buffer = self.draw_primitives_to_texture(
            scene,
            &instance_bindings,
            &mut writer,
            texture,
            viewport_size,
        );
        self.retire_effect_resources();
        let command_buffer = command_buffer?;

        let instance_buffer_pool = self.instance_buffer_pool.clone();
        let instance_buffer = Cell::new(Some(writer.finish()));
        let block = RcBlock::new(move |_: ptr::NonNull<AnyObject>| {
            if let Some(instance_buffer) = instance_buffer.take() {
                instance_buffer_pool.lock().release(instance_buffer);
            }
        });
        // SAFETY: Both pointee types are opaque views of the same Objective-C block pointer ABI.
        unsafe {
            command_buffer.add_completed_handler(&*RcBlock::as_ptr(&block).cast());
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
        let viewport_size = layer.drawable_size();
        let viewport_size: Size<DevicePixels> = size(
            (viewport_size.width.ceil() as i32).into(),
            (viewport_size.height.ceil() as i32).into(),
        );
        let drawable = layer
            .next_drawable()
            .ok_or_else(|| anyhow::anyhow!("Failed to get drawable for render_to_image"))?;

        let command_buffer = self.render_frame(scene, drawable.texture(), viewport_size)?;

        // Commit and wait for completion without presenting
        command_buffer.commit();
        command_buffer.wait_until_completed();

        read_texture_to_image(drawable.texture())
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
            self.update_path_intermediate_textures(size);

            // Create an offscreen texture as render target
            let texture_descriptor = metal::TextureDescriptor::new();
            texture_descriptor.set_width(size.width.0 as u64);
            texture_descriptor.set_height(size.height.0 as u64);
            texture_descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
            texture_descriptor.set_usage(
                metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead,
            );
            // Like the atlas, only Apple GPUs can create shared textures on macOS;
            // Intel Macs cannot, even with unified memory. iOS has no managed storage.
            let uses_shared_storage = cfg!(target_os = "ios") || self.is_apple_gpu;
            texture_descriptor.set_storage_mode(if uses_shared_storage {
                metal::MTLStorageMode::Shared
            } else {
                metal::MTLStorageMode::Managed
            });
            let target_texture = self.device.new_texture(&texture_descriptor);

            let command_buffer = self.render_frame(scene, &target_texture, size)?;

            // Managed textures require an explicit blit synchronize before the CPU
            // can read back the rendered data. Without this, get_bytes returns
            // stale zeros.
            if !uses_shared_storage {
                let blit = command_buffer.new_blit_command_encoder();
                blit.synchronize_resource(&target_texture);
                blit.end_encoding();
            }

            // Commit and wait for completion
            command_buffer.commit();
            command_buffer.wait_until_completed();

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
            self.update_path_intermediate_textures(size);

            let needs_new_target = self.headless_render_target.as_ref().is_none_or(|texture| {
                texture.width() != size.width.0 as u64 || texture.height() != size.height.0 as u64
            });
            if needs_new_target {
                let texture_descriptor = metal::TextureDescriptor::new();
                texture_descriptor.set_width(size.width.0 as u64);
                texture_descriptor.set_height(size.height.0 as u64);
                texture_descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
                texture_descriptor.set_usage(
                    metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead,
                );
                texture_descriptor.set_storage_mode(metal::MTLStorageMode::Private);
                self.headless_render_target = Some(self.device.new_texture(&texture_descriptor));
            }
            let target_texture = self
                .headless_render_target
                .clone()
                .expect("just ensured the render target exists");

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
        texture: &metal::TextureRef,
        viewport_size: Size<DevicePixels>,
    ) -> Result<metal::CommandBuffer> {
        if let Some(command_buffer) = crate::fast::paths::draw_primitives_to_texture(
            self,
            scene,
            instance_bindings,
            writer,
            texture,
            viewport_size,
        )? {
            return Ok(command_buffer);
        }
        let command_queue = self.command_queue.clone();
        let command_buffer = command_queue.new_command_buffer();
        let alpha = if self.opaque { 1. } else { 0. };
        let path_textures = self.window_path_textures();

        self.encode_scene(
            scene,
            instance_bindings,
            writer,
            command_buffer,
            texture,
            viewport_size,
            metal::MTLClearColor::new(0., 0., 0., alpha),
            &path_textures,
        )?;

        Ok(command_buffer.to_owned())
    }

    /// Encodes `scene` into `texture`, whose top-left corner is the scene's
    /// origin. Leaves no render encoder open, including on error.
    #[allow(clippy::too_many_arguments)]
    fn encode_scene(
        &mut self,
        scene: &Scene,
        instance_bindings: &InstanceBindings,
        writer: &mut InstanceBufferWriter,
        command_buffer: &metal::CommandBufferRef,
        texture: &metal::TextureRef,
        viewport_size: Size<DevicePixels>,
        clear_color: metal::MTLClearColor,
        path_textures: &PathIntermediateTextures,
    ) -> Result<()> {
        let mut command_encoder = new_command_encoder_for_texture(
            command_buffer,
            texture,
            viewport_size,
            Some(clear_color),
        );

        for batch in scene.scene_batches() {
            let batch = match batch {
                SceneBatch::Primitives(batch) => batch,
                SceneBatch::Effect(effect_index) => {
                    // The effect renders into its own textures, so the pass on
                    // `texture` is closed and reopened (loading its contents)
                    // around it, the same way paths are drawn.
                    command_encoder.end_encoding();
                    let effect = scene.effects.get(effect_index);
                    let output = match effect {
                        Some(effect) => self.render_effect(effect, writer, command_buffer)?,
                        None => None,
                    };
                    command_encoder = new_command_encoder_for_texture(
                        command_buffer,
                        texture,
                        viewport_size,
                        None,
                    );
                    if let (Some(effect), Some(output)) = (effect, output) {
                        if let Err(error) = self.composite_effect(
                            effect,
                            &output,
                            writer,
                            viewport_size,
                            command_encoder,
                        ) {
                            command_encoder.end_encoding();
                            return Err(error);
                        }
                    }
                    continue;
                }
            };
            match batch {
                PrimitiveBatch::Shadows(range) => {
                    self.draw_shadows(range, instance_bindings, viewport_size, command_encoder)
                }
                PrimitiveBatch::Quads(range) => {
                    self.draw_quads(range, instance_bindings, viewport_size, command_encoder)
                }
                PrimitiveBatch::Paths(range) => {
                    let paths = &scene.paths[range];
                    command_encoder.end_encoding();

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
                    );

                    if did_draw {
                        if let Err(error) = self.draw_paths_from_intermediate(
                            paths,
                            writer,
                            viewport_size,
                            command_encoder,
                            path_textures,
                        ) {
                            command_encoder.end_encoding();
                            return Err(error);
                        }
                    }
                }
                PrimitiveBatch::Underlines(range) => {
                    self.draw_underlines(range, instance_bindings, viewport_size, command_encoder)
                }
                PrimitiveBatch::MonochromeSprites { texture_id, range } => self
                    .draw_monochrome_sprites(
                        texture_id,
                        range,
                        instance_bindings,
                        viewport_size,
                        command_encoder,
                    ),
                PrimitiveBatch::PolychromeSprites { texture_id, range } => self
                    .draw_polychrome_sprites(
                        texture_id,
                        range,
                        instance_bindings,
                        viewport_size,
                        command_encoder,
                    ),
                PrimitiveBatch::Surfaces(range) => self.draw_surfaces(
                    &scene.surfaces[range.clone()],
                    range.start,
                    instance_bindings,
                    viewport_size,
                    command_encoder,
                ),
                PrimitiveBatch::SubpixelSprites { .. } => unreachable!(),
            }
        }

        command_encoder.end_encoding();

        Ok(())
    }

    fn draw_paths_to_intermediate(
        &self,
        paths: &[Path<ScaledPixels>],
        writer: &mut InstanceBufferWriter,
        viewport_size: Size<DevicePixels>,
        command_buffer: &metal::CommandBufferRef,
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

        let render_pass_descriptor = metal::RenderPassDescriptor::new();
        let color_attachment = render_pass_descriptor
            .color_attachments()
            .object_at(0)
            .unwrap();
        color_attachment.set_load_action(metal::MTLLoadAction::Clear);
        color_attachment.set_clear_color(metal::MTLClearColor::new(0., 0., 0., 0.));

        if let Some(msaa_texture) = &path_textures.msaa_texture {
            color_attachment.set_texture(Some(msaa_texture));
            color_attachment.set_resolve_texture(Some(intermediate_texture));
            color_attachment.set_store_action(metal::MTLStoreAction::MultisampleResolve);
        } else {
            color_attachment.set_texture(Some(intermediate_texture));
            color_attachment.set_store_action(metal::MTLStoreAction::Store);
        }

        let command_encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);
        command_encoder.set_render_pipeline_state(&self.paths_rasterization_pipeline_state);
        command_encoder.set_vertex_buffer(
            PathRasterizationInputIndex::Vertices as u64,
            Some(&vertex_instance_bindings.buffer),
            vertex_instance_bindings.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            PathRasterizationInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_buffer(
            PathRasterizationInputIndex::Vertices as u64,
            Some(&vertex_instance_bindings.buffer),
            vertex_instance_bindings.offset as u64,
        );
        command_encoder.draw_primitives(
            metal::MTLPrimitiveType::Triangle,
            0,
            vertices.len() as u64,
        );

        command_encoder.end_encoding();
        Ok(true)
    }

    pub(crate) fn draw_shadows(
        &self,
        shadows: Range<usize>,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if shadows.is_empty() {
            return;
        }

        command_encoder.set_render_pipeline_state(&self.shadows_pipeline_state);
        command_encoder.set_vertex_buffer(
            ShadowInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            ShadowInputIndex::Shadows as u64,
            Some(&instance_bindings.shadows.buffer),
            instance_bindings.shadows.offset as u64,
        );
        command_encoder.set_fragment_buffer(
            ShadowInputIndex::Shadows as u64,
            Some(&instance_bindings.shadows.buffer),
            instance_bindings.shadows.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            ShadowInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        command_encoder.draw_primitives_instanced_base_instance(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            shadows.len() as u64,
            shadows.start as u64,
        );
    }

    pub(crate) fn draw_quads(
        &self,
        quads: Range<usize>,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if quads.is_empty() {
            return;
        }

        command_encoder.set_render_pipeline_state(&self.quads_pipeline_state);
        command_encoder.set_vertex_buffer(
            QuadInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            QuadInputIndex::Quads as u64,
            Some(&instance_bindings.quads.buffer),
            instance_bindings.quads.offset as u64,
        );
        command_encoder.set_fragment_buffer(
            QuadInputIndex::Quads as u64,
            Some(&instance_bindings.quads.buffer),
            instance_bindings.quads.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            QuadInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        command_encoder.draw_primitives_instanced_base_instance(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            quads.len() as u64,
            quads.start as u64,
        );
    }

    pub(crate) fn draw_paths_from_intermediate(
        &self,
        paths: &[Path<ScaledPixels>],
        writer: &mut InstanceBufferWriter,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
        path_textures: &PathIntermediateTextures,
    ) -> Result<()> {
        let Some(first_path) = paths.first() else {
            return Ok(());
        };
        let intermediate_texture = path_textures
            .texture
            .as_ref()
            .context("missing path intermediate texture")?;

        command_encoder.set_render_pipeline_state(&self.path_sprites_pipeline_state);
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        command_encoder.set_fragment_texture(
            SpriteInputIndex::AtlasTexture as u64,
            Some(intermediate_texture),
        );

        // When copying paths from the intermediate texture to the drawable,
        // each pixel must only be copied once, in case of transparent paths.
        //
        // If all paths have the same draw order, then their bounds are all
        // disjoint, so we can copy each path's bounds individually. If this
        // batch combines different draw orders, we perform a single copy
        // for a minimal spanning rect.
        let sprites;
        if paths.last().unwrap().order == first_path.order {
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
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&sprite_instance_bindings.buffer),
            sprite_instance_bindings.offset as u64,
        );

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            sprites.len() as u64,
        );
        Ok(())
    }

    pub(crate) fn draw_underlines(
        &self,
        underlines: Range<usize>,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if underlines.is_empty() {
            return;
        }

        command_encoder.set_render_pipeline_state(&self.underlines_pipeline_state);
        command_encoder.set_vertex_buffer(
            UnderlineInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            UnderlineInputIndex::Underlines as u64,
            Some(&instance_bindings.underlines.buffer),
            instance_bindings.underlines.offset as u64,
        );
        command_encoder.set_fragment_buffer(
            UnderlineInputIndex::Underlines as u64,
            Some(&instance_bindings.underlines.buffer),
            instance_bindings.underlines.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            UnderlineInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        command_encoder.draw_primitives_instanced_base_instance(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            underlines.len() as u64,
            underlines.start as u64,
        );
    }

    pub(crate) fn draw_monochrome_sprites(
        &self,
        texture_id: AtlasTextureId,
        sprites: Range<usize>,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
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
        command_encoder.set_render_pipeline_state(&self.monochrome_sprites_pipeline_state);
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_bindings.monochrome_sprites.buffer),
            instance_bindings.monochrome_sprites.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::AtlasTextureSize as u64,
            mem::size_of_val(&texture_size) as u64,
            &texture_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_bindings.monochrome_sprites.buffer),
            instance_bindings.monochrome_sprites.offset as u64,
        );
        command_encoder.set_fragment_texture(SpriteInputIndex::AtlasTexture as u64, Some(&texture));

        command_encoder.draw_primitives_instanced_base_instance(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            sprites.len() as u64,
            sprites.start as u64,
        );
    }

    pub(crate) fn draw_polychrome_sprites(
        &self,
        texture_id: AtlasTextureId,
        sprites: Range<usize>,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
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
        command_encoder.set_render_pipeline_state(&self.polychrome_sprites_pipeline_state);
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_bindings.polychrome_sprites.buffer),
            instance_bindings.polychrome_sprites.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::AtlasTextureSize as u64,
            mem::size_of_val(&texture_size) as u64,
            &texture_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_bindings.polychrome_sprites.buffer),
            instance_bindings.polychrome_sprites.offset as u64,
        );
        command_encoder.set_fragment_texture(SpriteInputIndex::AtlasTexture as u64, Some(&texture));

        command_encoder.draw_primitives_instanced_base_instance(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            sprites.len() as u64,
            sprites.start as u64,
        );
    }

    pub(crate) fn draw_surfaces(
        &mut self,
        surfaces: &[PaintSurface],
        first_surface: usize,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if surfaces.is_empty() {
            return;
        }

        command_encoder.set_render_pipeline_state(&self.surfaces_pipeline_state);
        command_encoder.set_vertex_buffer(
            SurfaceInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            SurfaceInputIndex::Surfaces as u64,
            Some(&instance_bindings.surfaces.buffer),
            instance_bindings.surfaces.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            SurfaceInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        for (index, surface) in surfaces.iter().enumerate() {
            let texture_size = size(
                DevicePixels::from(surface.image_buffer.get_width() as i32),
                DevicePixels::from(surface.image_buffer.get_height() as i32),
            );

            assert_eq!(
                surface.image_buffer.get_pixel_format(),
                kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
            );

            let y_texture = self
                .core_video_texture_cache
                .create_texture_from_image(
                    surface.image_buffer.as_concrete_TypeRef(),
                    None,
                    MTLPixelFormat::R8Unorm,
                    surface.image_buffer.get_width_of_plane(0),
                    surface.image_buffer.get_height_of_plane(0),
                    0,
                )
                .unwrap();
            let cb_cr_texture = self
                .core_video_texture_cache
                .create_texture_from_image(
                    surface.image_buffer.as_concrete_TypeRef(),
                    None,
                    MTLPixelFormat::RG8Unorm,
                    surface.image_buffer.get_width_of_plane(1),
                    surface.image_buffer.get_height_of_plane(1),
                    1,
                )
                .unwrap();

            command_encoder.set_vertex_bytes(
                SurfaceInputIndex::TextureSize as u64,
                mem::size_of_val(&texture_size) as u64,
                &texture_size as *const Size<DevicePixels> as *const _,
            );
            // let y_texture = y_texture.get_texture().unwrap().
            command_encoder.set_fragment_texture(SurfaceInputIndex::YTexture as u64, unsafe {
                let texture = CVMetalTextureGetTexture(y_texture.as_concrete_TypeRef());
                Some(metal::TextureRef::from_ptr(texture as *mut _))
            });
            command_encoder.set_fragment_texture(SurfaceInputIndex::CbCrTexture as u64, unsafe {
                let texture = CVMetalTextureGetTexture(cb_cr_texture.as_concrete_TypeRef());
                Some(metal::TextureRef::from_ptr(texture as *mut _))
            });

            command_encoder.draw_primitives_instanced_base_instance(
                metal::MTLPrimitiveType::Triangle,
                0,
                6,
                1,
                (first_surface + index) as u64,
            );
        }
    }

    /// Renders the effect's captured scene into its offscreen texture, runs
    /// the shader passes over it, and returns the texture holding the result.
    /// Must be called with no render encoder open on `command_buffer`.
    fn render_effect(
        &mut self,
        effect: &PaintEffect,
        writer: &mut InstanceBufferWriter,
        command_buffer: &metal::CommandBufferRef,
    ) -> Result<Option<metal::Texture>> {
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

        let pipelines: Vec<metal::RenderPipelineState> = effect
            .effect
            .shaders
            .iter()
            .map(|source| self.effect_pipeline(source))
            .collect::<Option<Vec<_>>>()
            .unwrap_or_default();

        let frame = self.effect_frame;
        let device = &self.device;
        let target = self
            .effect_targets
            .entry(effect.effect.id)
            .or_insert_with(|| EffectTarget::new(device, texture_size));
        if target.size != texture_size {
            *target = EffectTarget::new(device, texture_size);
        }
        target.last_used_frame = frame;
        if !effect.scene.paths.is_empty() && target.path_textures.texture.is_none() {
            target.path_textures = new_path_intermediate_textures(
                device,
                texture_size,
                self.path_sample_count,
                self.is_apple_gpu,
            );
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
            metal::MTLClearColor::new(0., 0., 0., 0.),
            &path_textures,
        )?;

        if pipelines.is_empty() {
            return Ok(Some(input));
        }

        // The palette makes Ghostty's uniform block larger than the 4 KiB
        // `set_fragment_bytes` allows, so it goes through the instance buffer,
        // whose 256-byte offset alignment also suits constant buffers.
        let uniforms =
            write_padded_bytes(writer, &effect.effect.uniforms, MIN_EFFECT_UNIFORM_SIZE)?;
        for pipeline in &pipelines {
            let render_pass_descriptor = metal::RenderPassDescriptor::new();
            let color_attachment = render_pass_descriptor
                .color_attachments()
                .object_at(0)
                .context("missing color attachment")?;
            color_attachment.set_texture(Some(&output));
            color_attachment.set_load_action(metal::MTLLoadAction::Clear);
            color_attachment.set_clear_color(metal::MTLClearColor::new(0., 0., 0., 0.));
            color_attachment.set_store_action(metal::MTLStoreAction::Store);

            let command_encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);
            command_encoder.set_render_pipeline_state(pipeline);
            command_encoder.set_fragment_texture(0, Some(&input));
            command_encoder.set_fragment_sampler_state(0, Some(&self.effect_sampler));
            command_encoder.set_fragment_buffer(1, Some(&uniforms.buffer), uniforms.offset as u64);
            command_encoder.draw_primitives(metal::MTLPrimitiveType::Triangle, 0, 3);
            command_encoder.end_encoding();

            mem::swap(&mut input, &mut output);
        }

        Ok(Some(input))
    }

    fn composite_effect(
        &self,
        effect: &PaintEffect,
        effect_texture: &metal::TextureRef,
        writer: &mut InstanceBufferWriter,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> Result<()> {
        let bounds_binding = writer.write(&[EffectCompositeBounds {
            bounds: effect.bounds,
            content_mask: effect.content_mask,
        }])?;

        command_encoder.set_render_pipeline_state(&self.effect_composite_pipeline_state);
        command_encoder.set_vertex_buffer(
            EffectCompositeInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            EffectCompositeInputIndex::Bounds as u64,
            Some(&bounds_binding.buffer),
            bounds_binding.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            EffectCompositeInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_texture(
            EffectCompositeInputIndex::Texture as u64,
            Some(effect_texture),
        );
        command_encoder.draw_primitives(metal::MTLPrimitiveType::Triangle, 0, 6);
        Ok(())
    }

    /// Returns the compiled pass for `source`, compiling it on first use.
    /// Compile failures are cached as well, so a broken shader is reported
    /// once instead of being recompiled every frame.
    fn effect_pipeline(&mut self, source: &Arc<str>) -> Option<metal::RenderPipelineState> {
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

    fn compile_effect_pipeline(&self, source: &str) -> Result<metal::RenderPipelineState> {
        let library = self
            .device
            .new_library_with_source(source, &metal::CompileOptions::new())
            .map_err(anyhow::Error::msg)
            .context("compiling Metal source")?;
        let fragment_function = library
            .get_function("main0", None)
            .map_err(anyhow::Error::msg)
            .context("locating fragment function main0")?;

        let descriptor = metal::RenderPipelineDescriptor::new();
        descriptor.set_label("shader_effect_pass");
        descriptor.set_vertex_function(Some(self.effect_pass_vertex_function.as_ref()));
        descriptor.set_fragment_function(Some(fragment_function.as_ref()));
        let color_attachment = descriptor
            .color_attachments()
            .object_at(0)
            .context("missing color attachment")?;
        color_attachment.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        // Each pass replaces its target, as Ghostty's custom shader passes do.
        color_attachment.set_blending_enabled(false);

        self.device
            .new_render_pipeline_state(&descriptor)
            .map_err(anyhow::Error::msg)
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
    textures: [metal::Texture; 2],
    /// Allocated only once the effect's scene contains paths.
    path_textures: PathIntermediateTextures,
    last_used_frame: u64,
}

impl EffectTarget {
    fn new(device: &metal::DeviceRef, size: Size<DevicePixels>) -> Self {
        let texture_descriptor = metal::TextureDescriptor::new();
        texture_descriptor.set_width(size.width.0 as u64);
        texture_descriptor.set_height(size.height.0 as u64);
        texture_descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        texture_descriptor.set_storage_mode(metal::MTLStorageMode::Private);
        texture_descriptor
            .set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);
        Self {
            size,
            textures: [
                device.new_texture(&texture_descriptor),
                device.new_texture(&texture_descriptor),
            ],
            path_textures: PathIntermediateTextures::default(),
            last_used_frame: 0,
        }
    }
}

struct EffectPipeline {
    /// `None` when the source failed to compile.
    state: Option<metal::RenderPipelineState>,
    last_used_frame: u64,
}

/// The textures paths are rasterized into before being copied to their target.
/// They must match the target's size, because the path sprite shader maps
/// positions to texture coordinates through the viewport size.
#[derive(Clone, Default)]
pub(crate) struct PathIntermediateTextures {
    texture: Option<metal::Texture>,
    msaa_texture: Option<metal::Texture>,
}

fn new_path_intermediate_textures(
    device: &metal::DeviceRef,
    size: Size<DevicePixels>,
    path_sample_count: u32,
    is_apple_gpu: bool,
) -> PathIntermediateTextures {
    let texture_descriptor = metal::TextureDescriptor::new();
    texture_descriptor.set_width(size.width.0 as u64);
    texture_descriptor.set_height(size.height.0 as u64);
    texture_descriptor.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
    texture_descriptor.set_storage_mode(metal::MTLStorageMode::Private);
    texture_descriptor
        .set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);
    let texture = Some(device.new_texture(&texture_descriptor));

    let msaa_texture = if path_sample_count > 1 {
        // https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus
        // Rendering MSAA textures are done in a single pass, so we can use memory-less storage on Apple Silicon
        let storage_mode = if is_apple_gpu {
            metal::MTLStorageMode::Memoryless
        } else {
            metal::MTLStorageMode::Private
        };

        let msaa_descriptor = texture_descriptor;
        msaa_descriptor.set_texture_type(metal::MTLTextureType::D2Multisample);
        msaa_descriptor.set_storage_mode(storage_mode);
        msaa_descriptor.set_sample_count(path_sample_count as _);
        Some(device.new_texture(&msaa_descriptor))
    } else {
        None
    };

    PathIntermediateTextures {
        texture,
        msaa_texture,
    }
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

pub(crate) fn new_command_encoder_for_texture<'a>(
    command_buffer: &'a metal::CommandBufferRef,
    texture: &'a metal::TextureRef,
    viewport_size: Size<DevicePixels>,
    clear_color: Option<metal::MTLClearColor>,
) -> &'a metal::RenderCommandEncoderRef {
    let render_pass_descriptor = metal::RenderPassDescriptor::new();
    let color_attachment = render_pass_descriptor
        .color_attachments()
        .object_at(0)
        .unwrap();
    color_attachment.set_texture(Some(texture));
    color_attachment.set_store_action(metal::MTLStoreAction::Store);
    if let Some(clear_color) = clear_color {
        color_attachment.set_load_action(metal::MTLLoadAction::Clear);
        color_attachment.set_clear_color(clear_color);
    } else {
        color_attachment.set_load_action(metal::MTLLoadAction::Load);
    }

    let command_encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);
    command_encoder.set_viewport(metal::MTLViewport {
        originX: 0.0,
        originY: 0.0,
        width: i32::from(viewport_size.width) as f64,
        height: i32::from(viewport_size.height) as f64,
        znear: 0.0,
        zfar: 1.0,
    });
    command_encoder
}

#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
fn read_texture_to_image(texture: &metal::TextureRef) -> Result<RgbaImage> {
    let width = texture.width() as u32;
    let height = texture.height() as u32;
    let bytes_per_row = width as usize * 4;
    let mut pixels = vec![0u8; height as usize * bytes_per_row];

    let region = metal::MTLRegion {
        origin: metal::MTLOrigin { x: 0, y: 0, z: 0 },
        size: metal::MTLSize {
            width: width as u64,
            height: height as u64,
            depth: 1,
        },
    };
    texture.get_bytes(
        pixels.as_mut_ptr() as *mut std::ffi::c_void,
        bytes_per_row as u64,
        region,
        0,
    );

    // Convert BGRA to RGBA (swap B and R channels)
    for chunk in pixels.chunks_exact_mut(4) {
        chunk.swap(0, 2);
    }

    RgbaImage::from_raw(width, height, pixels).context("failed to create RgbaImage from pixel data")
}

fn build_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::SourceAlpha);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    // CDXC:PlatformSupport 2026-09-19 WHY:
    // Alpha composites "over" like the colour, as wgpu's `ALPHA_BLENDING` does on Linux. Adding it (`One`) summed the coverage of every primitive sharing an anti-aliased edge, and a div paints its background and its border as two quads with the same edge, so a transparent window's rounded corners came out nearly opaque with only the colour of a half-covered pixel, which macOS composited as a dark dotted rim on light backgrounds. Opaque windows ignore this channel.
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

fn build_path_sprite_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    // Alpha must accumulate as `src.a + dst.a * (1 - src.a)` like the other pipelines. An
    // additive `One` saturates to opaque wherever a path's antialiased edge lands on an already
    // translucent pixel, and on a transparent window that punches a dark fringe into the blur.
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

fn build_path_rasterization_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
    path_sample_count: u32,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    if path_sample_count > 1 {
        descriptor.set_raster_sample_count(path_sample_count as _);
        descriptor.set_alpha_to_coverage_enabled(false);
    }
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

#[derive(Clone)]
pub(crate) struct InstanceBinding {
    pub(crate) buffer: metal::Buffer,
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

fn write_instances(scene: &Scene, writer: &mut InstanceBufferWriter) -> Result<InstanceBindings> {
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
    device: metal::Device,
    pool: Arc<Mutex<InstanceBufferPool>>,
    unified_memory: bool,
    filled: Vec<(InstanceBuffer, usize)>,
    current: InstanceBuffer,
    offset: usize,
}

impl InstanceBufferWriter {
    pub(crate) fn new(
        device: &metal::Device,
        pool: &Arc<Mutex<InstanceBufferPool>>,
        unified_memory: bool,
    ) -> Self {
        let current = pool.lock().acquire(device, unified_memory);
        Self {
            device: device.clone(),
            pool: pool.clone(),
            unified_memory,
            filled: Vec::new(),
            current,
            offset: 0,
        }
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
        // Safety: the reservation lies within a buffer this frame owns
        // exclusively, and never overlaps one handed out earlier.
        let values = unsafe {
            let start = (self.current.metal_buffer.contents() as *mut u8).add(offset);
            slice::from_raw_parts_mut(start.cast::<MaybeUninit<T>>(), count)
        };
        Ok((binding, values))
    }

    pub(crate) fn write<T>(&mut self, values: &[T]) -> Result<InstanceBinding> {
        let (binding, destination) = self.allocate::<T>(values.len())?;
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
        let buffer = pool.acquire(&self.device, self.unified_memory);
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
                buffer.metal_buffer.did_modify_range(NSRange {
                    location: 0,
                    length: *written as NSUInteger,
                });
            }
            if offset > 0 {
                current.metal_buffer.did_modify_range(NSRange {
                    location: 0,
                    length: offset as NSUInteger,
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
    pub fn new() -> Self {
        let instance_buffer_pool = Arc::new(Mutex::new(InstanceBufferPool::default()));
        let renderer = MetalRenderer::new_headless(instance_buffer_pool);
        Self { renderer }
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
