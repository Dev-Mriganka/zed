//! Low-resolution, GPU-rendered glass backdrops shared by the Linux window backends.
//! The scene is composited over this once, preserving independent sidebar/work-area tint.
use gpui::{Bounds, LiveBackground, Pixels};
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[derive(Clone)]
pub struct GlassImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

#[derive(Clone, Default)]
pub struct GlassSource {
    pub image: Option<Arc<GlassImage>>,
    pub live: Option<LiveBackground>,
    pub phase: f32,
}

#[derive(Clone)]
pub struct GlassFrame {
    pub current: GlassSource,
    pub fade: f32,
    pub transition: u64,
    pub view_size: [f32; 2],
    pub cover: Bounds<Pixels>,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LiveUniforms {
    resolution: [f32; 2],
    view_size: [f32; 2],
    cover_origin: [f32; 2],
    cover_size: [f32; 2],
    phase: f32,
    period: f32,
    brightness: f32,
    style: f32,
    colors: [[f32; 4]; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    current: LiveUniforms,
    fade: [f32; 4],
}

pub(crate) struct GlassRenderer {
    pipeline: wgpu::RenderPipeline,
    copy_pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    picture: Option<(Arc<GlassImage>, wgpu::Texture)>,
    empty: wgpu::Texture,
    canvas: wgpu::Texture,
    snapshot: wgpu::Texture,
    capture_snapshot: bool,
    frame: GlassFrame,
}

impl GlassRenderer {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, frame: GlassFrame) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("glass"),
            source: wgpu::ShaderSource::Wgsl(include_str!("glass.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("glass"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                texture_binding(2),
                texture_binding(3),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("glass"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry, target| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("glass_vertex"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        Self {
            pipeline: pipeline("glass_fragment", wgpu::TextureFormat::Rgba8Unorm),
            copy_pipeline: pipeline("glass_copy", format),
            layout,
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            uniform: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("glass uniforms"),
                contents: bytemuck::bytes_of(&<Uniforms as bytemuck::Zeroable>::zeroed()),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }),
            picture: None,
            empty: texture(device, 1, 1),
            canvas: texture(device, 320, 200),
            snapshot: texture(device, 320, 200),
            capture_snapshot: false,
            frame,
        }
    }

    pub fn set_frame(&mut self, frame: GlassFrame) {
        self.capture_snapshot |= self.frame.transition != frame.transition && frame.fade < 1.0;
        self.frame = frame;
    }

    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        size: [u32; 2],
    ) {
        if self.capture_snapshot {
            encoder.copy_texture_to_texture(
                self.canvas.as_image_copy(),
                self.snapshot.as_image_copy(),
                self.canvas.size(),
            );
            self.capture_snapshot = false;
        }
        {
            let slot = &mut self.picture;
            match &self.frame.current.image {
                Some(image)
                    if !slot
                        .as_ref()
                        .is_some_and(|(old, _)| Arc::ptr_eq(old, image)) =>
                {
                    let picture = slot
                        .take()
                        .filter(|(_, picture)| {
                            picture.width() == image.width && picture.height() == image.height
                        })
                        .map(|(_, picture)| picture)
                        .unwrap_or_else(|| texture(device, image.width, image.height));
                    queue.write_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: &picture,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        &image.rgba,
                        wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(image.width * 4),
                            rows_per_image: Some(image.height),
                        },
                        picture.size(),
                    );
                    *slot = Some((image.clone(), picture));
                }
                None => *slot = None,
                _ => {}
            }
        }
        let resolution = [320.0, 200.0];
        let uniforms = Uniforms {
            current: uniforms(&self.frame.current, &self.frame, resolution),
            fade: [self.frame.fade, size[0] as f32, size[1] as f32, 1.0],
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniforms));
        let previous = self.snapshot.create_view(&Default::default());
        let current = self
            .picture
            .as_ref()
            .map_or(&self.empty, |(_, t)| t)
            .create_view(&Default::default());
        let canvas = self.canvas.create_view(&Default::default());
        let group = self.bind_group(device, &previous, &current);
        render(encoder, &canvas, &self.pipeline, &group);
        let group = self.bind_group(device, &previous, &canvas);
        render(encoder, target, &self.copy_pipeline, &group);
    }

    fn bind_group(
        &self,
        device: &wgpu::Device,
        previous: &wgpu::TextureView,
        current: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("glass"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(previous),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(current),
                },
            ],
        })
    }
}

fn texture_binding(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("glass"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    })
}

fn render(
    encoder: &mut wgpu::CommandEncoder,
    target: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
    group: &wgpu::BindGroup,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("glass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })],
        ..Default::default()
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, group, &[]);
    pass.draw(0..3, 0..1);
}

fn uniforms(source: &GlassSource, frame: &GlassFrame, resolution: [f32; 2]) -> LiveUniforms {
    let style = source
        .live
        .as_ref()
        .and_then(|live| {
            [
                "aurora", "ink", "drift", "nebula", "silk", "bokeh", "waves", "mesh",
            ]
            .iter()
            .position(|id| *id == live.style.as_ref())
        })
        .map_or(if source.image.is_some() { 9.0 } else { 0.0 }, |index| {
            (index + 1) as f32
        });
    LiveUniforms {
        resolution,
        view_size: frame.view_size,
        cover_origin: [frame.cover.origin.x.into(), frame.cover.origin.y.into()],
        cover_size: [
            frame.cover.size.width.into(),
            frame.cover.size.height.into(),
        ],
        phase: source.phase,
        period: 120.0,
        brightness: source.live.as_ref().map_or(1.0, |live| live.brightness),
        style,
        colors: source.live.as_ref().map_or([[0.0; 4]; 3], |live| {
            live.colors.map(|c| [c[0], c[1], c[2], 1.0])
        }),
    }
}
