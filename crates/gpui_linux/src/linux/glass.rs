//! Linux glass asset loading and playback. Both display servers use the same clock,
//! image preparation and power policy; only desktop/window discovery is platform-specific.
//!
//! CDXC:Theming 2026-09-27 SEE-ALSO:
//! apps/desktop/src/app/helpers/window_glass.rs selects the same sources and tint rules on macOS and Linux. gpui_wgpu/src/glass.wgsl ports gpui_macos/src/live_backdrop.metal, including its eight gains and two-minute loops. Both Linux backends must keep these background setters and focus/visibility notifications in step.
use gpui::{Bounds, LiveBackground, Pixels, Point, Size, px};
use gpui_wgpu::{GlassFrame, GlassImage, GlassSource};
use std::{
    path::PathBuf,
    sync::{Arc, mpsc},
    time::{Duration, Instant},
};

mod system;
mod video;

/// The blur, in points, of the pictures and videos a window's glass shows when the app never set
/// one (`set_background_blur_style`); matches the macOS backdrop's.
pub(crate) const DEFAULT_BLUR_RADIUS: f32 = 60.0;

#[derive(Default)]
pub(crate) struct Glass {
    pub enabled: bool,
    pub image: Option<PathBuf>,
    pub video: Option<PathBuf>,
    pub live: Option<LiveBackground>,
    pub only_on_power: bool,
    pub follows_screen: bool,
    pub cover: Option<Bounds<Pixels>>,
    /// The blur radius in points; `None` is `DEFAULT_BLUR_RADIUS` and 0 shows the picture sharp.
    blur_radius: Option<f32>,
    pub title: String,
    pub wake: Option<calloop::ping::Ping>,
    worker: Option<Worker>,
    observed_revision: u64,
    current: GlassSource,
    selection: Option<Selection>,
    fade_started: Option<Instant>,
    phase: f32,
    transition: u64,
    dirty: bool,
}

#[derive(Clone, PartialEq)]
enum Selection {
    Image(Option<PathBuf>),
    Video(PathBuf),
    Live(LiveBackground),
}

static LIVE_CLOCK: std::sync::Mutex<(Option<Instant>, f32)> = std::sync::Mutex::new((None, 0.0));

impl Glass {
    pub fn with_wake(wake: calloop::ping::Ping) -> Self {
        Self {
            wake: Some(wake),
            ..Default::default()
        }
    }

    pub fn frame(
        &mut self,
        active: bool,
        bounds: Bounds<Pixels>,
        screen: Bounds<Pixels>,
        monitor: Option<String>,
        light: bool,
        wayland: bool,
    ) -> Option<GlassFrame> {
        if !self.enabled {
            self.worker = None;
            self.current = GlassSource::default();
            self.selection = None;
            self.fade_started = None;
            return None;
        }
        let selection = if let Some(live) = &self.live {
            Selection::Live(live.clone())
        } else if let Some(video) = &self.video {
            Selection::Video(video.clone())
        } else {
            Selection::Image(self.image.clone())
        };
        let wake = self.wake.clone();
        let blur_radius = self.blur_radius();
        let worker = self.worker.get_or_insert_with(|| Worker::new(wake));
        let request = Request {
            selection: selection.clone(),
            active,
            only_on_power: self
                .live
                .as_ref()
                .map_or(self.only_on_power, |live| live.only_on_power),
            title: self.title.clone(),
            monitor,
            light,
            follows_screen: self.follows_screen
                && (wayland || std::env::var_os("HYPRLAND_INSTANCE_SIGNATURE").is_some()),
            cover_width: f32::from(
                self.cover
                    .unwrap_or(if self.follows_screen { screen } else { bounds })
                    .size
                    .width,
            )
            .max(1.0),
            blur_radius,
        };
        worker.request(request);
        let output = worker.output.lock().unwrap().clone();
        self.observed_revision = output.revision;
        let now = Instant::now();
        let running = active && output.may_animate;
        if self.live.is_some() {
            let mut clock = LIVE_CLOCK.lock().unwrap();
            if running {
                let delta = clock
                    .0
                    .replace(now)
                    .map_or(0.0, |last| now.duration_since(last).as_secs_f32());
                if delta < 0.1 {
                    clock.1 =
                        (clock.1 + delta * self.live.as_ref().unwrap().speed).rem_euclid(120.0);
                }
            }
            self.phase = clock.1;
        }
        let image = if output.selection.as_ref() == Some(&selection) {
            output.image.clone()
        } else {
            None
        };
        let pending = output.selection.as_ref() != Some(&selection) || output.loading;
        let ready = self.live.is_some() || image.is_some();
        if ready && self.selection.as_ref() != Some(&selection) {
            self.transition = self.transition.wrapping_add(1);
            self.fade_started = (!output.reduce_motion
                && (self.current.image.is_some() || self.current.live.is_some()))
            .then_some(now);
            self.selection = Some(selection.clone());
        }
        if ready {
            self.current = GlassSource {
                image,
                live: self.live.clone(),
                phase: if output.reduce_motion {
                    0.0
                } else {
                    self.phase
                },
            };
        } else if !pending {
            self.current = GlassSource::default();
        }
        let fade = self.fade_started.map_or(1.0, |start| {
            (now.duration_since(start).as_secs_f32() / 0.75).min(1.0)
        });
        if fade == 1.0 {
            self.fade_started = None;
        }
        self.dirty = fade < 1.0;
        if self.current.image.is_none() && self.current.live.is_none() {
            return None;
        }
        let cover = if self.follows_screen {
            if let Some(cover) = output.desktop_cover {
                cover
            } else if wayland {
                Bounds::new(Point::default(), bounds.size)
            } else {
                Bounds::new(screen.origin - bounds.origin, screen.size)
            }
        } else {
            self.cover
                .unwrap_or(Bounds::new(Point::default(), bounds.size))
        };
        Some(GlassFrame {
            current: self.current.clone(),
            fade,
            transition: self.transition,
            view_size: [bounds.size.width.into(), bounds.size.height.into()],
            cover,
        })
    }

    /// Sets the blur of the pictures and videos this window's glass shows; a change re-blurs them.
    pub fn set_blur_radius(&mut self, radius: Pixels) {
        let radius = f32::from(radius);
        self.blur_radius = Some(if radius.is_finite() {
            radius.max(0.0)
        } else {
            DEFAULT_BLUR_RADIUS
        });
    }

    fn blur_radius(&self) -> f32 {
        self.blur_radius.unwrap_or(DEFAULT_BLUR_RADIUS)
    }

    pub fn needs_frame(&self) -> bool {
        self.enabled
            && (self.dirty
                || self.worker.as_ref().is_some_and(|worker| {
                    worker.output.lock().unwrap().revision != self.observed_revision
                }))
    }

    pub fn set_active(&mut self, active: bool) {
        if let Some(worker) = &mut self.worker {
            if let Some(mut request) = worker.last_request.clone() {
                request.active = active;
                worker.request(request);
            }
        }
    }
}

#[derive(Clone, PartialEq)]
struct Request {
    selection: Selection,
    active: bool,
    only_on_power: bool,
    title: String,
    monitor: Option<String>,
    light: bool,
    follows_screen: bool,
    cover_width: f32,
    blur_radius: f32,
}

#[derive(Clone, Default)]
struct Output {
    revision: u64,
    selection: Option<Selection>,
    image: Option<Arc<GlassImage>>,
    loading: bool,
    may_animate: bool,
    reduce_motion: bool,
    desktop_cover: Option<Bounds<Pixels>>,
}

struct Worker {
    sender: Option<mpsc::Sender<Request>>,
    output: Arc<std::sync::Mutex<Output>>,
    last_request: Option<Request>,
}

impl Worker {
    fn new(wake: Option<calloop::ping::Ping>) -> Self {
        let (sender, receiver) = mpsc::channel();
        let output = Arc::new(std::sync::Mutex::new(Output {
            loading: true,
            ..Default::default()
        }));
        let shared = output.clone();
        std::thread::Builder::new()
            .name("glass-backdrop".into())
            .spawn(move || run(receiver, shared, wake))
            .expect("glass worker");
        Self {
            sender: Some(sender),
            output,
            last_request: None,
        }
    }

    fn request(&mut self, request: Request) {
        if self.last_request.as_ref() != Some(&request) {
            if let Some(sender) = &self.sender {
                let _ = sender.send(request.clone());
            }
            self.last_request = Some(request);
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.sender.take();
    }
}

fn run(
    receiver: mpsc::Receiver<Request>,
    output: Arc<std::sync::Mutex<Output>>,
    wake: Option<calloop::ping::Ping>,
) {
    let Ok(mut request) = receiver.recv() else {
        return;
    };
    let mut system = system::System::default();
    let mut applied = None;
    let mut image_key = None;
    let mut player: Option<video::Video> = None;
    let mut video_key = None;
    loop {
        let before = output.lock().unwrap().clone();
        let changed = applied.as_ref() != Some(&request.selection);
        system.refresh(&request);
        let may_animate = system.may_animate(request.only_on_power);
        let playing = request.active && may_animate;
        if changed {
            player = None;
            image_key = None;
            let mut result = output.lock().unwrap();
            result.loading = true;
            result.selection = Some(request.selection.clone());
            result.image = None;
        }
        match &request.selection {
            Selection::Live(_) => {}
            Selection::Image(custom) => {
                let path = custom.clone().or_else(|| system.wallpaper.clone());
                let key = path.as_ref().map(|path| {
                    (
                        path.clone(),
                        std::fs::metadata(path).and_then(|m| m.modified()).ok(),
                        request.cover_width.to_bits(),
                        request.blur_radius.to_bits(),
                    )
                });
                if changed || key != image_key {
                    let image = path.as_ref().and_then(|path| {
                        load_image(path, request.cover_width, request.blur_radius)
                    });
                    output.lock().unwrap().image = image.map(Arc::new);
                    image_key = key;
                }
            }
            Selection::Video(path) => {
                let key = Some(request.blur_radius.to_bits());
                if changed || key != video_key || (!before.reduce_motion && system.reduce_motion) {
                    video_key = key;
                    player = video::Video::open(path, request.cover_width, request.blur_radius);
                    if player.is_none() {
                        log::warn!(
                            "Cannot open glass video; FFmpeg and ffprobe must be installed and the file must be readable"
                        );
                    }
                    output.lock().unwrap().image = None;
                }
                if let Some(video) = &mut player {
                    video.set_playing(playing || output.lock().unwrap().image.is_none());
                    if let Some(frame) = video.frame() {
                        output.lock().unwrap().image = Some(Arc::new(frame));
                    }
                }
            }
        }
        {
            let mut result = output.lock().unwrap();
            result.loading = player.as_ref().is_some_and(|player| player.loading());
            result.may_animate = may_animate;
            result.reduce_motion = system.reduce_motion;
            result.desktop_cover = system.desktop_cover;
        }
        {
            let mut result = output.lock().unwrap();
            let image_changed = match (&result.image, &before.image) {
                (Some(a), Some(b)) => !Arc::ptr_eq(a, b),
                (None, None) => false,
                _ => true,
            };
            if changed
                || image_changed
                || before.loading != result.loading
                || before.may_animate != result.may_animate
                || before.desktop_cover != result.desktop_cover
                || (playing && matches!(request.selection, Selection::Live(_)))
            {
                result.revision += 1;
                if let Some(wake) = &wake {
                    wake.ping();
                }
            }
        }
        applied = Some(request.selection.clone());
        let first_frame_pending = player.as_ref().is_some_and(|player| player.loading());
        let interval = if (playing || first_frame_pending)
            && !matches!(request.selection, Selection::Image(_))
        {
            Duration::from_millis(42)
        } else if request.follows_screen && request.active {
            Duration::from_millis(40)
        } else {
            Duration::from_millis(500)
        };
        match receiver.recv_timeout(interval) {
            Ok(next) => {
                request = next;
                while let Ok(next) = receiver.try_recv() {
                    request = next;
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
}

fn load_image(path: &PathBuf, cover_width: f32, blur_radius: f32) -> Option<GlassImage> {
    let reader = image::ImageReader::open(path)
        .ok()?
        .with_guessed_format()
        .ok()?;
    let mut image = reader.decode().ok()?;
    // Preserve EXIF orientation before fitting and blurring, like Core Image on macOS.
    if let Ok(mut decoder) = image::ImageReader::open(path)
        .ok()?
        .with_guessed_format()
        .ok()?
        .into_decoder()
    {
        use image::ImageDecoder;
        if let Ok(orientation) = decoder.orientation() {
            image.apply_orientation(orientation);
        }
    }
    let mut image = image.thumbnail(640, 640).to_rgba8();
    for pixel in image.pixels_mut() {
        let alpha = u16::from(pixel[3]);
        for channel in &mut pixel.0[..3] {
            *channel = (u16::from(*channel) * alpha / 255) as u8;
        }
        pixel[3] = 255;
    }
    let sigma = blur_radius * image.width() as f32 / cover_width;
    // A radius of 0 shows the picture sharp; the blur itself rejects a zero sigma.
    let blurred = if sigma > 0.0 {
        image::imageops::blur(&image, sigma)
    } else {
        image
    };
    Some(GlassImage {
        width: blurred.width(),
        height: blurred.height(),
        rgba: blurred.into_raw(),
    })
}
