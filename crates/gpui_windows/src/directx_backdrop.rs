//! Ghostex: the backdrop of a blurred wallpaper-mode window on Windows, the counterpart of
//! gpui_macos's window_wallpaper.rs and window_live.rs: the blurred desktop wallpaper of the
//! window's monitor (or a picture the app chose), or one of the live styles (live_backdrop.hlsl),
//! drawn behind the window's content in place of the system's live blur.
//!
//! CDXC:Theming 2026-09-27 DECISION:
//! User answered "Build now" to building the Wallpaper, Picture and Live glass backgrounds for Windows. They draw the same as on macOS: the same eight styles with the same math, two-minute loops that never jump, cross-fades on every change, the same pause rules, and the blurred wallpaper of the window's own monitor or a chosen picture, either covering the window or held still against the screen. The user's own video stays macOS-only (Media Foundation playback behind the glass is not built), so Settings does not offer it on Windows.
//!
//! It is a DirectComposition visual of its own, placed under the visual GPUI presents into, with
//! a small swap chain the compositor scales up: the backdrop never makes GPUI redraw its scene,
//! and GPUI's frames never redraw the backdrop.

use std::{
    cell::RefCell,
    collections::HashMap,
    path::{Path, PathBuf},
    rc::{Rc, Weak},
    slice,
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime},
};

use anyhow::{Context as _, Result};
use gpui::{Bounds, LiveBackground, Pixels};
use gpui_util::ResultExt;
use windows::{
    Win32::{
        Foundation::{COLORREF, HWND, POINT, RECT},
        Graphics::{
            Direct3D::{D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, Fxc::*, ID3DBlob, ID3DInclude},
            Direct3D11::*,
            DirectComposition::*,
            Dwm::{DWMWA_CLOAKED, DwmGetWindowAttribute},
            Dxgi::{Common::*, *},
            Gdi::{
                ClientToScreen, GetMonitorInfoW, MONITOR_DEFAULTTONEAREST, MONITORINFO,
                MonitorFromWindow,
            },
        },
        System::{
            Com::{CLSCTX_ALL, CoCreateInstance, CoTaskMemFree},
            Power::{GetSystemPowerStatus, SYSTEM_POWER_STATUS},
            Threading::GetCurrentProcessId,
        },
        UI::{
            HiDpi::GetDpiForWindow,
            Shell::{DWPOS_FIT, DWPOS_STRETCH, DesktopWallpaper, IDesktopWallpaper},
            WindowsAndMessaging::{
                GetClientRect, GetForegroundWindow, GetWindowThreadProcessId, IsIconic,
                IsWindowVisible, KillTimer, SPI_GETCLIENTAREAANIMATION,
                SYSTEM_PARAMETERS_INFO_UPDATE_FLAGS, SetTimer, SystemParametersInfoW,
            },
        },
    },
    core::{BOOL, PCSTR, PCWSTR},
};

use crate::{DirectComposition, DirectXRendererDevices, WindowsWindowInner};

const SHADER_SOURCE: &str = include_str!("live_backdrop.hlsl");

/// The styles live_backdrop.hlsl draws, by the id Settings saves (`live_<id>` in the shader).
const LIVE_STYLES: [&str; 8] = [
    "aurora", "ink", "drift", "nebula", "silk", "bokeh", "waves", "mesh",
];

/// One loop of every style, in seconds of the clock at speed 1; gpui_macos's window_live.rs
/// `LIVE_PERIOD` holds the user's decision ("can just be 2 mins only as long as it loops").
const LIVE_PERIOD: f64 = 120.0;

/// CDXC:Theming 2026-09-27 WHY:
/// The same budget the macOS live backdrop measured: an eighth of the window's logical size (at most 320x200 pixels) at 24 frames a second, scaled up by the compositor. The styles are soft and sit under a tint, so a bigger or faster target would cost more without showing more.
const LIVE_DOWNSCALE: f32 = 8.0;
const LIVE_MAX_TARGET: (f32, f32) = (320.0, 200.0);

/// A blurred picture shows no detail finer than a few points, so it is drawn at a quarter of the
/// window's logical size and scaled up, like gpui_macos's `CANVAS_SCALE`. A lighter blur keeps
/// finer detail and is drawn larger (`picture_scale`), up to full size for a sharp picture.
const PICTURE_SCALE: f32 = 0.25;

/// The pictures' blur radius, in points, for a window that never set one with
/// `set_background_blur_style`.
const PICTURE_BLUR_RADIUS: f32 = 60.0;

const FRAME_INTERVAL_MS: u32 = 1000 / 24;

/// While every live backdrop is paused, how often the timer checks whether one may move again
/// (the app coming back to the front, the charger plugged in).
const PAUSED_POLL_MS: u32 = 500;

/// A pause longer than this does not advance the clock, so motion resumes where it stopped.
const LIVE_MAX_STEP: f64 = 0.1;

/// How long a change of style, colours, brightness or picture cross-fades.
const FADE: Duration = Duration::from_millis(750);

/// Pictures blurred so far, by file, modification time, layout and canvas size.
const PICTURE_CACHE_LIMIT: usize = 6;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct BackdropUniforms {
    resolution: [f32; 2],
    view_size: [f32; 2],
    cover_origin: [f32; 2],
    cover_size: [f32; 2],
    phase: f32,
    period: f32,
    brightness: f32,
    amount: f32,
    c0: [f32; 4],
    c1: [f32; 4],
    c2: [f32; 4],
}

/// What a window asked its backdrop to show, from the `set_background_*` calls.
#[derive(Clone, Default, PartialEq)]
pub(crate) struct BackdropRequest {
    pub(crate) blurred: bool,
    pub(crate) wallpaper: bool,
    pub(crate) image: Option<PathBuf>,
    pub(crate) follows_screen: bool,
    pub(crate) cover: Option<Bounds<Pixels>>,
    pub(crate) live: Option<LiveBackground>,
    /// The pictures' blur radius in points from `set_background_blur_style`; `None` keeps
    /// `PICTURE_BLUR_RADIUS`.
    pub(crate) blur_radius: Option<f32>,
}

impl BackdropRequest {
    fn picture_blur_radius(&self) -> f32 {
        self.blur_radius.unwrap_or(PICTURE_BLUR_RADIUS)
    }
}

/// How much of the window's logical size a picture blurred by `radius` points is drawn at: detail
/// finer than about a quarter of the radius does not survive the blur.
fn picture_scale(radius: f32) -> f32 {
    (4.0 / radius.max(0.0)).clamp(PICTURE_SCALE, 1.0)
}

/// Where the window sits, in device pixels, for laying a picture out against its monitor.
#[derive(Clone, Copy, PartialEq)]
pub(crate) struct BackdropPlacement {
    client_origin: (i32, i32),
    client_size: (u32, u32),
    monitor: (i32, i32, i32, i32),
    scale: f32,
}

impl BackdropPlacement {
    fn logical_size(&self) -> (f32, f32) {
        (
            self.client_size.0 as f32 / self.scale,
            self.client_size.1 as f32 / self.scale,
        )
    }

    fn monitor_size(&self) -> (f32, f32) {
        (
            (self.monitor.2 - self.monitor.0) as f32,
            (self.monitor.3 - self.monitor.1) as f32,
        )
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum PictureLayout {
    Cover,
    Contain,
    Stretch,
}

struct BlurredPicture {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

struct PictureSource {
    path: PathBuf,
    layout: PictureLayout,
    background: [u8; 3],
    canvas: (u32, u32),
    /// Blur radius in canvas pixels; 0 leaves the picture sharp.
    blur: usize,
    key: String,
}

type PictureSlot = Arc<Mutex<Option<Option<Arc<BlurredPicture>>>>>;

enum Content {
    Picture {
        key: String,
        picture: Arc<BlurredPicture>,
    },
    Live(LiveBackground),
}

struct RenderTexture {
    texture: ID3D11Texture2D,
    target: ID3D11RenderTargetView,
    view: ID3D11ShaderResourceView,
    size: (u32, u32),
}

struct BackdropGpu {
    visual: IDCompositionVisual,
    swap_chain: IDXGISwapChain1,
    swap_size: (u32, u32),
    transform: Option<((u32, u32), (u32, u32))>,
    display: Option<RenderTexture>,
    frame: Option<RenderTexture>,
    fading_from: Option<RenderTexture>,
    picture: Option<(String, ID3D11ShaderResourceView)>,
    vertex: ID3D11VertexShader,
    pixel: HashMap<&'static str, ID3D11PixelShader>,
    uniforms: ID3D11Buffer,
    sampler: ID3D11SamplerState,
}

/// The backdrop of one window, kept by its renderer.
#[derive(Default)]
pub(crate) struct Backdrop {
    request: BackdropRequest,
    placement: Option<BackdropPlacement>,
    content: Option<Content>,
    loading: Option<(String, PictureSlot)>,
    gpu: Option<BackdropGpu>,
    fade_started: Option<Instant>,
    /// Set when what is on screen no longer matches the content, placement or size.
    dirty: bool,
}

impl Backdrop {
    pub(crate) fn request(&self) -> &BackdropRequest {
        &self.request
    }

    /// Whether this backdrop shows (or is about to show) something of its own.
    pub(crate) fn is_active(&self) -> bool {
        self.request.blurred
            && self.request.wallpaper
            && (self
                .request
                .live
                .as_ref()
                .is_some_and(|live| draws_style(&live.style))
                || self.loading.is_some()
                || matches!(self.content, Some(Content::Picture { .. })))
    }

    fn is_live(&self) -> bool {
        matches!(self.content, Some(Content::Live(_)))
    }

    fn only_on_power(&self) -> bool {
        match &self.content {
            Some(Content::Live(live)) => live.only_on_power,
            _ => false,
        }
    }

    /// Takes a new request and placement; resolves what to show and redraws what changed.
    pub(crate) fn update(
        &mut self,
        request: BackdropRequest,
        placement: Option<BackdropPlacement>,
        devices: Option<&DirectXRendererDevices>,
        composition: Option<&DirectComposition>,
        phase: f64,
    ) {
        let request_changed = self.request != request;
        let placement_changed = self.placement != placement;
        let monitor_changed = match (self.placement, placement) {
            (Some(old), Some(new)) => old.monitor != new.monitor || old.scale != new.scale,
            _ => placement_changed,
        };
        self.request = request;
        self.placement = placement;
        if request_changed || monitor_changed {
            self.resolve();
        }
        if request_changed || placement_changed {
            self.dirty = true;
        }
        self.render(devices, composition, phase);
    }

    /// Drops everything that belongs to the lost device; the next frame builds it again.
    pub(crate) fn release_gpu(&mut self) {
        self.gpu = None;
        self.fade_started = None;
        self.dirty = true;
    }

    /// Forgets the desktop wallpaper it read, after Windows reported a new one.
    pub(crate) fn wallpaper_changed(&mut self) {
        if let Ok(mut cache) = PICTURE_CACHE.lock() {
            cache.clear();
        }
        self.resolve();
        self.dirty = true;
    }

    fn resolve(&mut self) {
        let request = &self.request;
        if !request.blurred || !request.wallpaper {
            self.set_content(None);
            self.loading = None;
            return;
        }
        if let Some(live) = request.live.clone().filter(|live| draws_style(&live.style)) {
            self.loading = None;
            self.set_content(Some(Content::Live(live)));
            return;
        }
        let Some(placement) = self.placement else {
            return;
        };
        let Some(source) = picture_source(
            request.image.as_deref(),
            &placement,
            request.picture_blur_radius(),
        ) else {
            // A picture that cannot be read leaves the system's live blur, as on macOS.
            self.loading = None;
            self.set_content(None);
            return;
        };
        if let Some(Content::Picture { key, .. }) = &self.content
            && *key == source.key
        {
            self.loading = None;
            return;
        }
        if let Some(picture) = cached_picture(&source.key) {
            self.loading = None;
            self.set_content(Some(Content::Picture {
                key: source.key,
                picture,
            }));
            return;
        }
        if self
            .loading
            .as_ref()
            .is_some_and(|(key, _)| *key == source.key)
        {
            return;
        }
        let slot: PictureSlot = Arc::new(Mutex::new(None));
        self.loading = Some((source.key.clone(), slot.clone()));
        std::thread::spawn(move || {
            let picture = blur_picture(&source).log_err().map(Arc::new);
            if let Some(picture) = &picture
                && let Ok(mut cache) = PICTURE_CACHE.lock()
            {
                cache.retain(|(key, _)| *key != source.key);
                cache.push((source.key.clone(), picture.clone()));
                if cache.len() > PICTURE_CACHE_LIMIT {
                    cache.remove(0);
                }
            }
            if let Ok(mut slot) = slot.lock() {
                *slot = Some(picture);
            }
        });
    }

    /// Picks up a picture a worker finished blurring.
    fn poll_loading(&mut self) {
        let Some((key, slot)) = &self.loading else {
            return;
        };
        let finished = slot.lock().ok().and_then(|mut slot| slot.take());
        let Some(picture) = finished else {
            return;
        };
        let key = key.clone();
        self.loading = None;
        self.set_content(picture.map(|picture| Content::Picture { key, picture }));
    }

    fn set_content(&mut self, content: Option<Content>) {
        let changes = match (&self.content, &content) {
            (None, None) => false,
            (Some(Content::Live(old)), Some(Content::Live(new))) => {
                old.style != new.style
                    || old.colors != new.colors
                    || old.brightness != new.brightness
            }
            (Some(Content::Picture { key: old, .. }), Some(Content::Picture { key: new, .. })) => {
                old != new
            }
            _ => true,
        };
        if changes
            && self.content.is_some()
            && content.is_some()
            && !reduce_motion()
            && let Some(gpu) = self.gpu.as_mut()
            && let Some(display) = gpu.display.take()
        {
            // Whatever is on screen now, a fade in progress included, is where this one starts.
            gpu.fading_from = Some(display);
            self.fade_started = Some(Instant::now());
        }
        // A speed or battery-rule change is taken as is: it changes how the clock runs, not what
        // is drawn.
        self.content = content;
        self.dirty = true;
    }

    /// One tick of the shared timer: picks up loaded pictures and redraws a moving or fading
    /// backdrop. Returns whether this backdrop wants the fast timer.
    fn tick(
        &mut self,
        devices: Option<&DirectXRendererDevices>,
        composition: Option<&DirectComposition>,
        phase: f64,
        moving: bool,
    ) -> bool {
        self.poll_loading();
        let fading = self.fade_started.is_some();
        if moving || fading || self.dirty {
            self.render(devices, composition, phase);
        }
        self.loading.is_some() || self.fade_started.is_some() || (moving && self.is_live())
    }

    fn render(
        &mut self,
        devices: Option<&DirectXRendererDevices>,
        composition: Option<&DirectComposition>,
        phase: f64,
    ) {
        if self.content.is_none() || !self.request.blurred || !self.request.wallpaper {
            if let (Some(gpu), Some(composition)) = (self.gpu.take(), composition) {
                composition.remove_backdrop(&gpu.visual).log_err();
            }
            self.fade_started = None;
            self.dirty = false;
            return;
        }
        let (Some(devices), Some(composition), Some(placement)) =
            (devices, composition, self.placement)
        else {
            return;
        };
        if self.gpu.is_none() {
            match BackdropGpu::new(devices, composition) {
                Ok(gpu) => self.gpu = Some(gpu),
                Err(error) => {
                    log::error!("Creating the window backdrop failed: {error:#}");
                    return;
                }
            }
        }
        if let Err(error) = self.draw_frame(devices, composition, &placement, phase) {
            log::error!("Drawing the window backdrop failed: {error:#}");
        }
        self.dirty = false;
    }

    fn draw_frame(
        &mut self,
        devices: &DirectXRendererDevices,
        composition: &DirectComposition,
        placement: &BackdropPlacement,
        phase: f64,
    ) -> Result<()> {
        let Some(content) = &self.content else {
            return Ok(());
        };
        let gpu = self.gpu.as_mut().context("backdrop resources missing")?;
        let (logical_width, logical_height) = placement.logical_size();
        let size = match content {
            Content::Live(_) => {
                let factor = (1.0 / LIVE_DOWNSCALE)
                    .min(LIVE_MAX_TARGET.0 / logical_width.max(1.0))
                    .min(LIVE_MAX_TARGET.1 / logical_height.max(1.0));
                (
                    (logical_width * factor).round().max(16.0) as u32,
                    (logical_height * factor).round().max(16.0) as u32,
                )
            }
            _ => {
                let scale = picture_scale(self.request.picture_blur_radius());
                (
                    (logical_width * scale).round().max(16.0) as u32,
                    (logical_height * scale).round().max(16.0) as u32,
                )
            }
        };
        gpu.fit(composition, size, placement.client_size)?;
        let fade = fade_amount(self.fade_started);
        if fade.is_none() {
            self.fade_started = None;
            gpu.fading_from = None;
        }
        let display = gpu.ensure_display(&devices.device, size)?;
        let draw_into = match fade {
            Some(_) => gpu.ensure_frame(&devices.device, size)?,
            None => display.clone(),
        };
        let context = &devices.device_context;
        match content {
            Content::Live(live) => {
                let style = style_id(&live.style).context("live style")?;
                let cover = self.request.cover.map_or(
                    ([0.0, 0.0], [logical_width, logical_height]),
                    |cover| {
                        (
                            [f32::from(cover.origin.x), f32::from(cover.origin.y)],
                            [f32::from(cover.size.width), f32::from(cover.size.height)],
                        )
                    },
                );
                let color = |index: usize| {
                    let [red, green, blue] = live.colors[index];
                    [red, green, blue, 1.0]
                };
                let uniforms = BackdropUniforms {
                    resolution: [size.0 as f32, size.1 as f32],
                    view_size: [logical_width, logical_height],
                    cover_origin: cover.0,
                    cover_size: cover.1,
                    phase: phase as f32,
                    period: LIVE_PERIOD as f32,
                    brightness: live.brightness.clamp(0.0, 1.0),
                    amount: 0.0,
                    c0: color(0),
                    c1: color(1),
                    c2: color(2),
                };
                gpu.draw_pass(context, style, &uniforms, &draw_into, &[])?;
            }
            Content::Picture { key, picture } => {
                let view = gpu.picture_view(&devices.device, key, picture)?;
                let rect = picture_rect(
                    &self.request,
                    placement,
                    (picture.width as f32, picture.height as f32),
                );
                // The rectangle, from client device pixels to the backdrop's own pixels.
                let to_target_x = size.0 as f32 / placement.client_size.0.max(1) as f32;
                let to_target_y = size.1 as f32 / placement.client_size.1.max(1) as f32;
                let uniforms = BackdropUniforms {
                    resolution: [size.0 as f32, size.1 as f32],
                    cover_origin: [rect.0 * to_target_x, rect.1 * to_target_y],
                    cover_size: [rect.2 * to_target_x, rect.3 * to_target_y],
                    ..Default::default()
                };
                gpu.draw_pass(
                    context,
                    "backdrop_picture",
                    &uniforms,
                    &draw_into,
                    &[None, Some(view)],
                )?;
            }
        }
        if let (Some(amount), Some(from)) = (fade, gpu.fading_from.as_ref()) {
            let uniforms = BackdropUniforms {
                amount,
                ..Default::default()
            };
            let from_view = from.view.clone();
            let to_view = draw_into.view.clone();
            gpu.draw_pass(
                context,
                "backdrop_composite",
                &uniforms,
                &display,
                &[Some(from_view), Some(to_view)],
            )?;
        }
        gpu.present(context, &display)
    }
}

impl BackdropGpu {
    fn new(devices: &DirectXRendererDevices, composition: &DirectComposition) -> Result<Self> {
        let device = &devices.device;
        let visual = unsafe { composition.device().CreateVisual() }?;
        unsafe {
            visual.SetBitmapInterpolationMode(DCOMPOSITION_BITMAP_INTERPOLATION_MODE_LINEAR)?;
        }
        let swap_chain = create_swap_chain(devices, 16, 16)?;
        unsafe { visual.SetContent(&swap_chain) }?;
        composition.add_backdrop(&visual)?;

        let vertex_bytes = shader_bytes("live_vertex", "vs_4_0")?;
        let vertex = unsafe {
            let mut shader = None;
            device.CreateVertexShader(&vertex_bytes, None, Some(&mut shader))?;
            shader.context("vertex shader")?
        };
        let mut pixel = HashMap::new();
        for entry in LIVE_STYLES
            .iter()
            .map(|style| style_entry(style))
            .chain(["backdrop_picture", "backdrop_composite"])
        {
            let bytes = shader_bytes(entry, "ps_4_0")?;
            let shader = unsafe {
                let mut shader = None;
                device.CreatePixelShader(&bytes, None, Some(&mut shader))?;
                shader.context("pixel shader")?
            };
            pixel.insert(entry, shader);
        }
        let uniforms = unsafe {
            let desc = D3D11_BUFFER_DESC {
                ByteWidth: std::mem::size_of::<BackdropUniforms>() as u32,
                Usage: D3D11_USAGE_DYNAMIC,
                BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
                CPUAccessFlags: D3D11_CPU_ACCESS_WRITE.0 as u32,
                MiscFlags: 0,
                StructureByteStride: 0,
            };
            let mut buffer = None;
            device.CreateBuffer(&desc, None, Some(&mut buffer))?;
            buffer.context("uniform buffer")?
        };
        let sampler = unsafe {
            let desc = D3D11_SAMPLER_DESC {
                Filter: D3D11_FILTER_MIN_MAG_MIP_LINEAR,
                AddressU: D3D11_TEXTURE_ADDRESS_CLAMP,
                AddressV: D3D11_TEXTURE_ADDRESS_CLAMP,
                AddressW: D3D11_TEXTURE_ADDRESS_CLAMP,
                MipLODBias: 0.0,
                MaxAnisotropy: 1,
                ComparisonFunc: D3D11_COMPARISON_ALWAYS,
                BorderColor: [0.0; 4],
                MinLOD: 0.0,
                MaxLOD: D3D11_FLOAT32_MAX,
            };
            let mut sampler = None;
            device.CreateSamplerState(&desc, Some(&mut sampler))?;
            sampler.context("sampler")?
        };
        Ok(Self {
            visual,
            swap_chain,
            swap_size: (16, 16),
            transform: None,
            display: None,
            frame: None,
            fading_from: None,
            picture: None,
            vertex,
            pixel,
            uniforms,
            sampler,
        })
    }

    /// Sizes the swap chain to the backdrop and scales the visual up to the window.
    fn fit(
        &mut self,
        composition: &DirectComposition,
        size: (u32, u32),
        client_size: (u32, u32),
    ) -> Result<()> {
        if self.swap_size != size {
            unsafe {
                self.swap_chain.ResizeBuffers(
                    2,
                    size.0,
                    size.1,
                    DXGI_FORMAT_B8G8R8A8_UNORM,
                    DXGI_SWAP_CHAIN_FLAG(0),
                )
            }
            .context("resizing the backdrop swap chain")?;
            self.swap_size = size;
        }
        if self.transform != Some((size, client_size)) {
            let matrix = windows_numerics::Matrix3x2 {
                M11: client_size.0 as f32 / size.0.max(1) as f32,
                M12: 0.0,
                M21: 0.0,
                M22: client_size.1 as f32 / size.1.max(1) as f32,
                M31: 0.0,
                M32: 0.0,
            };
            unsafe {
                self.visual.SetTransform2(&matrix)?;
                composition.device().Commit()?;
            }
            self.transform = Some((size, client_size));
        }
        Ok(())
    }

    fn ensure_display(&mut self, device: &ID3D11Device, size: (u32, u32)) -> Result<RenderTexture> {
        if self
            .display
            .as_ref()
            .is_none_or(|display| display.size != size)
        {
            self.display = Some(RenderTexture::new(device, size)?);
        }
        self.display.clone().context("backdrop display")
    }

    fn ensure_frame(&mut self, device: &ID3D11Device, size: (u32, u32)) -> Result<RenderTexture> {
        if self.frame.as_ref().is_none_or(|frame| frame.size != size) {
            self.frame = Some(RenderTexture::new(device, size)?);
        }
        self.frame.clone().context("backdrop frame")
    }

    fn picture_view(
        &mut self,
        device: &ID3D11Device,
        key: &str,
        picture: &BlurredPicture,
    ) -> Result<ID3D11ShaderResourceView> {
        if let Some((current, view)) = &self.picture
            && current == key
        {
            return Ok(view.clone());
        }
        let desc = D3D11_TEXTURE2D_DESC {
            Width: picture.width,
            Height: picture.height,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_IMMUTABLE,
            BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        let data = D3D11_SUBRESOURCE_DATA {
            pSysMem: picture.rgba.as_ptr() as *const _,
            SysMemPitch: picture.width * 4,
            SysMemSlicePitch: 0,
        };
        let view = unsafe {
            let mut texture = None;
            device.CreateTexture2D(&desc, Some(&data), Some(&mut texture))?;
            let texture = texture.context("picture texture")?;
            let mut view = None;
            device.CreateShaderResourceView(&texture, None, Some(&mut view))?;
            view.context("picture view")?
        };
        self.picture = Some((key.to_string(), view.clone()));
        Ok(view)
    }

    fn draw_pass(
        &self,
        context: &ID3D11DeviceContext,
        entry: &'static str,
        uniforms: &BackdropUniforms,
        target: &RenderTexture,
        textures: &[Option<ID3D11ShaderResourceView>],
    ) -> Result<()> {
        let pixel = self.pixel.get(entry).context("backdrop pixel shader")?;
        unsafe {
            let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
            context.Map(
                &self.uniforms,
                0,
                D3D11_MAP_WRITE_DISCARD,
                0,
                Some(&mut mapped),
            )?;
            std::ptr::copy_nonoverlapping(
                uniforms as *const BackdropUniforms,
                mapped.pData as *mut BackdropUniforms,
                1,
            );
            context.Unmap(&self.uniforms, 0);

            let viewport = D3D11_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: target.size.0 as f32,
                Height: target.size.1 as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            };
            // GPUI sets every piece of state it draws with at the start of its own frames
            // (targets, viewport, shaders, buffers, blend), so drawing here in between is safe.
            context.OMSetRenderTargets(Some(slice::from_ref(&Some(target.target.clone()))), None);
            context.RSSetViewports(Some(slice::from_ref(&viewport)));
            context.OMSetBlendState(None, None, 0xFFFFFFFF);
            context.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            context.VSSetShader(&self.vertex, None);
            context.PSSetShader(pixel, None);
            context.PSSetConstantBuffers(0, Some(slice::from_ref(&Some(self.uniforms.clone()))));
            context.PSSetSamplers(0, Some(slice::from_ref(&Some(self.sampler.clone()))));
            if !textures.is_empty() {
                context.PSSetShaderResources(0, Some(textures));
            }
            context.Draw(3, 0);
            // A texture drawn from here may be the next pass's target.
            context.PSSetShaderResources(0, Some(&[None, None]));
            context.OMSetRenderTargets(None, None);
        }
        Ok(())
    }

    fn present(&self, context: &ID3D11DeviceContext, display: &RenderTexture) -> Result<()> {
        unsafe {
            let back_buffer: ID3D11Texture2D = self.swap_chain.GetBuffer(0)?;
            context.CopyResource(&back_buffer, &display.texture);
            self.swap_chain.Present(0, DXGI_PRESENT(0)).ok()?;
        }
        Ok(())
    }
}

impl Clone for RenderTexture {
    fn clone(&self) -> Self {
        Self {
            texture: self.texture.clone(),
            target: self.target.clone(),
            view: self.view.clone(),
            size: self.size,
        }
    }
}

impl RenderTexture {
    fn new(device: &ID3D11Device, size: (u32, u32)) -> Result<Self> {
        let desc = D3D11_TEXTURE2D_DESC {
            Width: size.0,
            Height: size.1,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: (D3D11_BIND_RENDER_TARGET.0 | D3D11_BIND_SHADER_RESOURCE.0) as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        unsafe {
            let mut texture = None;
            device.CreateTexture2D(&desc, None, Some(&mut texture))?;
            let texture = texture.context("backdrop texture")?;
            let mut target = None;
            device.CreateRenderTargetView(&texture, None, Some(&mut target))?;
            let mut view = None;
            device.CreateShaderResourceView(&texture, None, Some(&mut view))?;
            Ok(Self {
                texture,
                target: target.context("backdrop target")?,
                view: view.context("backdrop view")?,
                size,
            })
        }
    }
}

fn create_swap_chain(
    devices: &DirectXRendererDevices,
    width: u32,
    height: u32,
) -> Result<IDXGISwapChain1> {
    let desc = DXGI_SWAP_CHAIN_DESC1 {
        Width: width,
        Height: height,
        Format: DXGI_FORMAT_B8G8R8A8_UNORM,
        Stereo: false.into(),
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
        BufferCount: 2,
        Scaling: DXGI_SCALING_STRETCH,
        SwapEffect: DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL,
        AlphaMode: DXGI_ALPHA_MODE_PREMULTIPLIED,
        Flags: 0,
    };
    Ok(unsafe {
        devices
            .dxgi_factory
            .CreateSwapChainForComposition(&devices.device, &desc, None)?
    })
}

fn style_entry(style: &str) -> &'static str {
    match style {
        "aurora" => "live_aurora",
        "ink" => "live_ink",
        "drift" => "live_drift",
        "nebula" => "live_nebula",
        "silk" => "live_silk",
        "bokeh" => "live_bokeh",
        "waves" => "live_waves",
        _ => "live_mesh",
    }
}

fn style_id(style: &str) -> Option<&'static str> {
    LIVE_STYLES
        .iter()
        .find(|known| **known == style)
        .map(|style| style_entry(style))
}

/// Whether this platform draws `style`.
pub(crate) fn draws_style(style: &str) -> bool {
    LIVE_STYLES.contains(&style)
}

thread_local! {
    /// Compiled shader bytecode by entry point; the HLSL is compiled once per process.
    static BYTECODE: RefCell<HashMap<&'static str, Vec<u8>>> = RefCell::new(HashMap::new());
}

/// CDXC:Theming 2026-09-27 WHY:
/// GPUI's own shaders are compiled by fxc in its build script; the backdrop's are compiled at run time with D3DCompile (d3dcompiler_47, part of every supported Windows) the first time a backdrop is shown, so building Ghostex needs no extra tool and a window that never shows a backdrop never pays for it.
fn shader_bytes(entry: &'static str, target: &'static str) -> Result<Vec<u8>> {
    if let Some(bytes) = BYTECODE.with_borrow(|cache| cache.get(entry).cloned()) {
        return Ok(bytes);
    }
    let entry_name = format!("{entry}\0");
    let target_name = format!("{target}\0");
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    let result = unsafe {
        D3DCompile(
            SHADER_SOURCE.as_ptr() as *const _,
            SHADER_SOURCE.len(),
            PCSTR(c"live_backdrop.hlsl".as_ptr() as *const u8),
            None,
            None::<&ID3DInclude>,
            PCSTR(entry_name.as_ptr()),
            PCSTR(target_name.as_ptr()),
            D3DCOMPILE_OPTIMIZATION_LEVEL3,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(error) = result {
        let message = errors.map(|errors| unsafe {
            String::from_utf8_lossy(slice::from_raw_parts(
                errors.GetBufferPointer() as *const u8,
                errors.GetBufferSize(),
            ))
            .into_owned()
        });
        anyhow::bail!("compiling {entry}: {error} {}", message.unwrap_or_default());
    }
    let code = code.context("compiled shader")?;
    let bytes = unsafe {
        slice::from_raw_parts(code.GetBufferPointer() as *const u8, code.GetBufferSize()).to_vec()
    };
    BYTECODE.with_borrow_mut(|cache| cache.insert(entry, bytes.clone()));
    Ok(bytes)
}

fn fade_amount(started: Option<Instant>) -> Option<f32> {
    let elapsed = started?.elapsed().as_secs_f32() / FADE.as_secs_f32();
    if elapsed >= 1.0 {
        return None;
    }
    let t = elapsed.max(0.0);
    Some(t * t * (3.0 - 2.0 * t))
}

/// The rectangle, in client device pixels, the picture is laid out over: the monitor when it stays
/// still against the screen, otherwise the cover rectangle (or the window) filled edge to edge.
fn picture_rect(
    request: &BackdropRequest,
    placement: &BackdropPlacement,
    picture: (f32, f32),
) -> (f32, f32, f32, f32) {
    if request.follows_screen {
        let (width, height) = placement.monitor_size();
        return (
            (placement.monitor.0 - placement.client_origin.0) as f32,
            (placement.monitor.1 - placement.client_origin.1) as f32,
            width,
            height,
        );
    }
    let (x, y, width, height) = match request.cover {
        Some(cover) => (
            f32::from(cover.origin.x) * placement.scale,
            f32::from(cover.origin.y) * placement.scale,
            f32::from(cover.size.width) * placement.scale,
            f32::from(cover.size.height) * placement.scale,
        ),
        None => (
            0.0,
            0.0,
            placement.client_size.0 as f32,
            placement.client_size.1 as f32,
        ),
    };
    let fill = (width / picture.0.max(1.0)).max(height / picture.1.max(1.0));
    let fill_width = picture.0 * fill;
    let fill_height = picture.1 * fill;
    (
        x + (width - fill_width) / 2.0,
        y + (height - fill_height) / 2.0,
        fill_width,
        fill_height,
    )
}

static PICTURE_CACHE: Mutex<Vec<(String, Arc<BlurredPicture>)>> = Mutex::new(Vec::new());

fn cached_picture(key: &str) -> Option<Arc<BlurredPicture>> {
    PICTURE_CACHE.lock().ok().and_then(|cache| {
        cache
            .iter()
            .find(|(cached, _)| cached == key)
            .map(|(_, picture)| picture.clone())
    })
}

/// The file the backdrop blurs by `radius` points and how it sits on the monitor: the app's
/// picture filling the monitor, or the monitor's desktop wallpaper laid out the way Windows places
/// it.
fn picture_source(
    image: Option<&Path>,
    placement: &BackdropPlacement,
    radius: f32,
) -> Option<PictureSource> {
    let (monitor_width, monitor_height) = placement.monitor_size();
    let scale = picture_scale(radius);
    let canvas = (
        ((monitor_width / placement.scale) * scale)
            .round()
            .max(16.0) as u32,
        ((monitor_height / placement.scale) * scale)
            .round()
            .max(16.0) as u32,
    );
    let blur = if radius > 0.0 {
        (radius * scale).round().max(1.0) as usize
    } else {
        0
    };
    let (path, layout, background) = match image {
        Some(image) => (image.to_path_buf(), PictureLayout::Cover, [0, 0, 0]),
        None => desktop_wallpaper(placement.monitor)?,
    };
    let modified = std::fs::metadata(&path)
        .and_then(|metadata| metadata.modified())
        .ok()?
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default();
    let key = format!(
        "{}|{modified}|{layout:?}|{background:?}|{}x{}|{blur}",
        path.display(),
        canvas.0,
        canvas.1
    );
    Some(PictureSource {
        path,
        layout,
        background,
        canvas,
        blur,
        key,
    })
}

/// The desktop wallpaper Windows shows on the monitor at `monitor`, its placement and the colour
/// around it. `None` for a solid colour, a slideshow between pictures, or when it cannot be read.
fn desktop_wallpaper(monitor: (i32, i32, i32, i32)) -> Option<(PathBuf, PictureLayout, [u8; 3])> {
    unsafe {
        let wallpaper: IDesktopWallpaper =
            CoCreateInstance(&DesktopWallpaper, None, CLSCTX_ALL).log_err()?;
        let count = wallpaper.GetMonitorDevicePathCount().log_err()?;
        let mut path = None;
        for index in 0..count {
            let Ok(monitor_id) = wallpaper.GetMonitorDevicePathAt(index) else {
                continue;
            };
            let matches = wallpaper
                .GetMonitorRECT(PCWSTR(monitor_id.0))
                .is_ok_and(|rect| (rect.left, rect.top, rect.right, rect.bottom) == monitor);
            if matches && let Ok(file) = wallpaper.GetWallpaper(PCWSTR(monitor_id.0)) {
                if !file.is_null() {
                    path = file.to_string().ok();
                }
                CoTaskMemFree(Some(file.0 as *const _));
            }
            CoTaskMemFree(Some(monitor_id.0 as *const _));
            if matches {
                break;
            }
        }
        let path = PathBuf::from(path.filter(|path| !path.is_empty())?);
        let layout = match wallpaper.GetPosition() {
            Ok(position) if position == DWPOS_FIT => PictureLayout::Contain,
            Ok(position) if position == DWPOS_STRETCH => PictureLayout::Stretch,
            // Fill and Span cover the monitor; Center and Tile are drawn as Fill, which is what
            // they look like once blurred unless the picture is much smaller than the screen.
            _ => PictureLayout::Cover,
        };
        let COLORREF(color) = wallpaper.GetBackgroundColor().unwrap_or(COLORREF(0));
        let background = [
            (color & 0xff) as u8,
            ((color >> 8) & 0xff) as u8,
            ((color >> 16) & 0xff) as u8,
        ];
        Some((path, layout, background))
    }
}

/// Decodes the picture, lays it out on a small canvas the size of the monitor and blurs it the
/// way the macOS backdrop does (not at all for a radius of 0).
fn blur_picture(source: &PictureSource) -> Result<BlurredPicture> {
    let image = image::ImageReader::open(&source.path)
        .with_context(|| format!("opening {}", source.path.display()))?
        .with_guessed_format()?
        .decode()
        .with_context(|| format!("decoding {}", source.path.display()))?
        .to_rgba8();
    let (canvas_width, canvas_height) = source.canvas;
    let (image_width, image_height) = (image.width().max(1) as f32, image.height().max(1) as f32);
    let (placed_width, placed_height) = match source.layout {
        PictureLayout::Stretch => (canvas_width as f32, canvas_height as f32),
        PictureLayout::Cover | PictureLayout::Contain => {
            let scale_x = canvas_width as f32 / image_width;
            let scale_y = canvas_height as f32 / image_height;
            let scale = if source.layout == PictureLayout::Cover {
                scale_x.max(scale_y)
            } else {
                scale_x.min(scale_y)
            };
            (image_width * scale, image_height * scale)
        }
    };
    let placed = image::imageops::resize(
        &image,
        placed_width.round().max(1.0) as u32,
        placed_height.round().max(1.0) as u32,
        image::imageops::FilterType::Triangle,
    );
    let [red, green, blue] = source.background;
    let mut canvas = image::RgbaImage::from_pixel(
        canvas_width,
        canvas_height,
        image::Rgba([red, green, blue, 255]),
    );
    let offset_x = (canvas_width as i64 - placed.width() as i64) / 2;
    let offset_y = (canvas_height as i64 - placed.height() as i64) / 2;
    image::imageops::overlay(&mut canvas, &placed, offset_x, offset_y);
    let mut rgba = canvas.into_raw();
    if source.blur > 0 {
        for _ in 0..3 {
            box_blur(
                &mut rgba,
                canvas_width as usize,
                canvas_height as usize,
                source.blur / 2 + 1,
            );
        }
    }
    for pixel in rgba.chunks_exact_mut(4) {
        pixel[3] = 255;
    }
    Ok(BlurredPicture {
        width: canvas_width,
        height: canvas_height,
        rgba,
    })
}

/// One horizontal and one vertical box pass; three of them come close to a Gaussian.
fn box_blur(rgba: &mut [u8], width: usize, height: usize, radius: usize) {
    let mut scratch = vec![0u8; rgba.len()];
    box_pass(rgba, &mut scratch, width, height, radius, true);
    box_pass(&scratch, rgba, width, height, radius, false);
}

fn box_pass(
    source: &[u8],
    target: &mut [u8],
    width: usize,
    height: usize,
    radius: usize,
    horizontal: bool,
) {
    let (lines, length) = if horizontal {
        (height, width)
    } else {
        (width, height)
    };
    let index = |line: usize, position: usize| {
        if horizontal {
            (line * width + position) * 4
        } else {
            (position * width + line) * 4
        }
    };
    let window = (radius * 2 + 1) as u32;
    for line in 0..lines {
        for channel in 0..3 {
            let sample = |position: isize| {
                let position = position.clamp(0, length as isize - 1) as usize;
                u32::from(source[index(line, position) + channel])
            };
            let mut sum: u32 = (-(radius as isize)..=radius as isize).map(sample).sum();
            for position in 0..length {
                target[index(line, position) + channel] = (sum / window) as u8;
                sum += sample(position as isize + radius as isize + 1);
                sum -= sample(position as isize - radius as isize);
            }
        }
    }
}

/// Where `hwnd`'s client area and monitor are, in device pixels.
pub(crate) fn placement_for(hwnd: HWND) -> Option<BackdropPlacement> {
    unsafe {
        let mut client = RECT::default();
        GetClientRect(hwnd, &mut client).ok()?;
        let mut origin = POINT { x: 0, y: 0 };
        if !ClientToScreen(hwnd, &mut origin).as_bool() {
            return None;
        }
        let monitor = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);
        let mut info = MONITORINFO {
            cbSize: std::mem::size_of::<MONITORINFO>() as u32,
            ..Default::default()
        };
        if !GetMonitorInfoW(monitor, &mut info).as_bool() {
            return None;
        }
        let dpi = GetDpiForWindow(hwnd).max(96);
        Some(BackdropPlacement {
            client_origin: (origin.x, origin.y),
            client_size: (
                (client.right - client.left).max(1) as u32,
                (client.bottom - client.top).max(1) as u32,
            ),
            monitor: (
                info.rcMonitor.left,
                info.rcMonitor.top,
                info.rcMonitor.right,
                info.rcMonitor.bottom,
            ),
            scale: dpi as f32 / 96.0,
        })
    }
}

/// Windows' "Show animations in Windows" off: backdrops hold a still frame and changes do not fade,
/// like Reduce Motion on macOS.
fn reduce_motion() -> bool {
    let mut animations = BOOL(1);
    let read = unsafe {
        SystemParametersInfoW(
            SPI_GETCLIENTAREAANIMATION,
            0,
            Some(&mut animations as *mut BOOL as *mut _),
            SYSTEM_PARAMETERS_INFO_UPDATE_FLAGS(0),
        )
    };
    read.is_ok() && !animations.as_bool()
}

/// Whether a live backdrop in `hwnd` should move right now: the rules gpui_macos's
/// `window_may_animate` applies (the app in front, the window visible and not minimised, no
/// battery saver, animations on, and the charger rule).
fn window_may_animate(hwnd: HWND, only_on_power: bool) -> bool {
    unsafe {
        let foreground = GetForegroundWindow();
        let mut process = 0u32;
        GetWindowThreadProcessId(foreground, Some(&mut process));
        if foreground.is_invalid() || process != GetCurrentProcessId() {
            return false;
        }
        if !IsWindowVisible(hwnd).as_bool() || IsIconic(hwnd).as_bool() {
            return false;
        }
        let mut cloaked = 0u32;
        if DwmGetWindowAttribute(
            hwnd,
            DWMWA_CLOAKED,
            &mut cloaked as *mut u32 as *mut _,
            std::mem::size_of::<u32>() as u32,
        )
        .is_ok()
            && cloaked != 0
        {
            return false;
        }
        let mut power = SYSTEM_POWER_STATUS::default();
        if GetSystemPowerStatus(&mut power).is_ok() {
            // SystemStatusFlag 1 is battery saver; ACLineStatus 0 is running on the battery.
            if power.SystemStatusFlag == 1 {
                return false;
            }
            if only_on_power && power.ACLineStatus == 0 {
                return false;
            }
        }
        !reduce_motion()
    }
}

struct Clock {
    phase: f64,
    last_tick: Option<Instant>,
    timer: usize,
    interval: u32,
}

thread_local! {
    /// Every window with a backdrop, for the shared timer.
    static WINDOWS: RefCell<Vec<(HWND, Weak<WindowsWindowInner>)>> = const { RefCell::new(Vec::new()) };
    /// One clock for every live backdrop, so a window laid over another draws the same frame.
    static CLOCK: RefCell<Clock> = const {
        RefCell::new(Clock { phase: 0.0, last_tick: None, timer: 0, interval: 0 })
    };
}

/// The live clock's current phase, for a backdrop drawn outside a tick.
pub(crate) fn current_phase() -> f64 {
    CLOCK.with_borrow(|clock| clock.phase)
}

/// Adds or removes `inner` from the shared timer after its backdrop changed.
pub(crate) fn track_window(hwnd: HWND, inner: Weak<WindowsWindowInner>, active: bool) {
    WINDOWS.with_borrow_mut(|windows| {
        windows.retain(|(tracked, window)| *tracked != hwnd && window.strong_count() > 0);
        if active {
            windows.push((hwnd, inner));
        }
    });
    schedule(true);
}

/// Runs the timer fast while something moves, fades or loads, slowly while a live backdrop is
/// paused (to notice it may move again), and stops it when no window has a backdrop.
fn schedule(wants_fast: bool) {
    // A still picture needs no ticks; a paused live backdrop needs the slow poll to resume.
    let any = WINDOWS.with_borrow(|windows| {
        windows.iter().any(|(_, window)| {
            window.upgrade().is_some_and(|window| {
                window
                    .state
                    .renderer
                    .try_borrow()
                    .map_or(true, |renderer| renderer.backdrop().is_live())
            })
        })
    });
    let interval = match (any || wants_fast, wants_fast) {
        (false, _) => 0,
        (true, true) => FRAME_INTERVAL_MS,
        (true, false) => PAUSED_POLL_MS,
    };
    CLOCK.with_borrow_mut(|clock| unsafe {
        if clock.interval == interval {
            return;
        }
        if clock.timer != 0 {
            KillTimer(None, clock.timer).log_err();
            clock.timer = 0;
        }
        clock.interval = interval;
        clock.last_tick = None;
        if interval != 0 {
            clock.timer = SetTimer(None, 0, interval, Some(tick));
        }
    });
}

/// CDXC:Theming 2026-09-27 SEE-ALSO:
/// gpui_macos/src/window_live.rs `LIVE_PERIOD` holds the user's never-jump decision; this clock keeps it on Windows: it only accumulates time times speed while a backdrop may move, a pause or the first tick after one adds nothing, and it wraps at the period where every style's last frame is its first.
unsafe extern "system" fn tick(_: HWND, _: u32, _: usize, _: u32) {
    let windows: Vec<(HWND, Rc<WindowsWindowInner>)> = WINDOWS.with_borrow_mut(|windows| {
        windows.retain(|(_, window)| window.strong_count() > 0);
        windows
            .iter()
            .filter_map(|(hwnd, window)| Some((*hwnd, window.upgrade()?)))
            .collect()
    });
    let mut moving: Vec<bool> = Vec::with_capacity(windows.len());
    let mut speed = None;
    for (hwnd, window) in &windows {
        let (live, only_on_power) = window
            .state
            .renderer
            .try_borrow()
            .map(|renderer| {
                let backdrop = renderer.backdrop();
                (backdrop.is_live(), backdrop.only_on_power())
            })
            .unwrap_or((false, false));
        let may_move = live && window_may_animate(*hwnd, only_on_power);
        if may_move && speed.is_none() {
            speed = window
                .state
                .renderer
                .try_borrow()
                .ok()
                .and_then(|renderer| match &renderer.backdrop().content {
                    Some(Content::Live(live)) => Some(f64::from(live.speed.clamp(0.05, 4.0))),
                    _ => None,
                });
        }
        moving.push(may_move);
    }
    let phase = CLOCK.with_borrow_mut(|clock| {
        let now = Instant::now();
        match speed {
            Some(speed) => {
                let step = clock
                    .last_tick
                    .map(|last| now.duration_since(last).as_secs_f64())
                    .unwrap_or(0.0)
                    .min(LIVE_MAX_STEP);
                clock.phase = (clock.phase + step * speed).rem_euclid(LIVE_PERIOD);
                clock.last_tick = Some(now);
            }
            None => clock.last_tick = None,
        }
        clock.phase
    });
    let mut wants_fast = false;
    for ((_, window), moving) in windows.iter().zip(moving) {
        if let Ok(mut renderer) = window.state.renderer.try_borrow_mut() {
            wants_fast |= renderer.tick_backdrop(phase, moving);
        } else {
            wants_fast = true;
        }
    }
    schedule(wants_fast);
}

impl crate::DirectXRenderer {
    pub(crate) fn backdrop(&self) -> &Backdrop {
        &self.backdrop
    }

    /// Applies a new backdrop request for this window and returns whether it now has a backdrop.
    pub(crate) fn update_backdrop(
        &mut self,
        request: BackdropRequest,
        placement: Option<BackdropPlacement>,
    ) -> bool {
        let phase = current_phase();
        let (devices, composition) = (self.devices.as_ref(), self.direct_composition.as_ref());
        self.backdrop
            .update(request, placement, devices, composition, phase);
        self.backdrop.is_active()
    }

    /// The desktop wallpaper changed; re-reads it.
    pub(crate) fn backdrop_wallpaper_changed(&mut self) {
        self.backdrop.wallpaper_changed();
        self.redraw_backdrop();
    }

    /// Draws the backdrop again, after its device was rebuilt.
    pub(crate) fn redraw_backdrop(&mut self) {
        let (devices, composition) = (self.devices.as_ref(), self.direct_composition.as_ref());
        self.backdrop.render(devices, composition, current_phase());
    }

    fn tick_backdrop(&mut self, phase: f64, moving: bool) -> bool {
        let (devices, composition) = (self.devices.as_ref(), self.direct_composition.as_ref());
        self.backdrop.tick(devices, composition, phase, moving)
    }
}
