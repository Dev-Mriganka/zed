//! The live backdrop of a wallpaper-mode window: an animated style drawn with Metal
//! (live_backdrop.metal) behind the window's content, in place of the picture or video. It moves
//! only while someone can see it, under exactly the rules the video backdrop follows
//! (`window_video::window_may_animate`).
//!
//! Every live view shares one clock, so a window laid over another (a floating panel) that covers
//! the same rectangle draws exactly the same frame as the window under it.

use gpui::LiveBackground;
use objc2::{
    class, msg_send,
    rc::Retained,
    runtime::{AnyClass, AnyObject, Bool, ClassBuilder, ProtocolObject, Sel},
    sel,
};
use objc2_app_kit::{NSAutoresizingMaskOptions, NSWindowOrderingMode};
use objc2_foundation::{NSPoint, NSRect, NSSize, NSString};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLLoadAction, MTLOrigin, MTLPixelFormat,
    MTLPrimitiveType, MTLRenderCommandEncoder, MTLRenderPassDescriptor,
    MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLSize, MTLStorageMode, MTLStoreAction,
    MTLTexture, MTLTextureDescriptor, MTLTextureType, MTLTextureUsage,
};
use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};
use std::{
    cell::RefCell,
    collections::HashMap,
    ptr::{self, NonNull},
    sync::OnceLock,
    time::{Duration, Instant},
};

type Texture = Retained<ProtocolObject<dyn MTLTexture>>;
type Pipeline = Retained<ProtocolObject<dyn MTLRenderPipelineState>>;

use crate::window_video;

const SHADER_SOURCE: &str = include_str!("live_backdrop.metal");

/// The styles live_backdrop.metal draws, by the id Settings saves (`live_<id>` in the shader).
const LIVE_STYLES: [&str; 8] = [
    "aurora", "ink", "drift", "nebula", "silk", "bokeh", "waves", "mesh",
];

/// CDXC:Theming 2026-09-26 DECISION:
/// User: "the animations can't jump at all they need to loop and never break", then "can just be 2 mins only as long as it loops". Every style is periodic by construction: it moves only through whole turns of an angle that goes once round every `LIVE_PERIOD` seconds of the clock (two minutes at speed 1), so the clock wraps with the last frame of a loop equal to its first. The clock only ever accumulates time times speed: a speed change changes the rate, a pause holds it, and it is shared and kept for the whole app session, so a view created again (another window, the glass turned off and on) carries on from the same moment. A change of style, colours or brightness never snaps either: the picture on screen is kept and cross-fades into the new one (`LIVE_FADE`).
const LIVE_PERIOD: f64 = 120.0;

/// CDXC:Theming 2026-09-26 WHY:
/// The backdrop sits under a tint that hides most of it and every style is soft, so it is drawn small (an eighth of the window's points, at most 320x200 pixels) and scaled up by the compositor, and at 24 frames a second: slow calm motion looks the same at that rate. Measured on an M-series Mac: 0.1 to 0.9 ms of GPU time and about 70 microseconds of main-thread time per frame, which puts it at the video backdrop's level; do not raise the size or the rate without measuring again.
const LIVE_DOWNSCALE: f64 = 8.0;
const LIVE_MAX_DRAWABLE: (f64, f64) = (320.0, 200.0);
const LIVE_FRAME_INTERVAL: f64 = 1.0 / 24.0;

/// A pause longer than this does not advance the clock, so motion resumes where it stopped.
const LIVE_MAX_STEP: f64 = 0.1;

/// How long a change of style, colours or brightness cross-fades.
const LIVE_FADE: Duration = Duration::from_millis(750);

#[repr(C)]
struct LiveUniforms {
    resolution: [f32; 2],
    view_size: [f32; 2],
    cover_origin: [f32; 2],
    cover_size: [f32; 2],
    phase: f32,
    period: f32,
    brightness: f32,
    pad: f32,
    c0: [f32; 4],
    c1: [f32; 4],
    c2: [f32; 4],
}

struct LiveView {
    view: *mut AnyObject,
    layer: Retained<CAMetalLayer>,
    live: LiveBackground,
    /// The rectangle, in the view's top-left coordinates, the style is laid out over; `None` is the
    /// view itself.
    cover: Option<NSRect>,
    wants_motion: bool,
    /// The picture last put on screen; it is also what a change fades out from.
    display: Option<Texture>,
    /// The new picture on its own while a fade composites it over `fading_from`.
    style_frame: Option<Texture>,
    fading_from: Option<Texture>,
    fade_started: Option<Instant>,
}

struct LiveRenderer {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    pipelines: HashMap<&'static str, Pipeline>,
}

struct LiveClock {
    phase: f64,
    last_tick: Option<Instant>,
    timer: *mut AnyObject,
}

thread_local! {
    static VIEWS: RefCell<Vec<LiveView>> = const { RefCell::new(Vec::new()) };
    static RENDERER: RefCell<Option<LiveRenderer>> = const { RefCell::new(None) };
    static CLOCK: RefCell<LiveClock> = const {
        RefCell::new(LiveClock { phase: 0.0, last_tick: None, timer: ptr::null_mut() })
    };
}

static VIEW_CLASS: OnceLock<usize> = OnceLock::new();
static TICKER_CLASS: OnceLock<usize> = OnceLock::new();

/// Whether this platform draws `style`.
pub(crate) fn draws_style(style: &str) -> bool {
    LIVE_STYLES.contains(&style)
}

/// Adds a live backdrop drawing `live` below everything in `content_view`, or `None` when the
/// style is not one this platform draws or Metal is unavailable.
pub(crate) unsafe fn create_view(
    content_view: *mut AnyObject,
    live: LiveBackground,
) -> Option<*mut AnyObject> {
    unsafe {
        let style = style_id(&live.style)?;
        with_renderer(|renderer| pipeline(renderer, style).map(|_| ()))??;
        let layer = CAMetalLayer::new();
        with_renderer(|renderer| layer.setDevice(Some(&renderer.device)))?;
        layer.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        layer.setOpaque(true);
        // Frames are drawn off screen and copied in, so the drawable is a copy target.
        layer.setFramebufferOnly(false);
        layer.setMaximumDrawableCount(2);
        layer.setNeedsDisplayOnBoundsChange(false);
        let layer_id = Retained::as_ptr(&layer).cast_mut().cast::<AnyObject>();

        let class = view_class();
        let view: *mut AnyObject = msg_send![class, alloc];
        let content_bounds: NSRect = msg_send![content_view, bounds];
        let view: *mut AnyObject = msg_send![view, initWithFrame: content_bounds];
        let _: () = msg_send![
            view,
            setAutoresizingMask: NSAutoresizingMaskOptions::ViewWidthSizable
                | NSAutoresizingMaskOptions::ViewHeightSizable
        ];
        // A layer-hosting view: the Metal layer is the view's own layer and follows its size.
        let _: () = msg_send![view, setLayer: layer_id];
        let _: () = msg_send![view, setWantsLayer: Bool::YES];
        let _: () = msg_send![
            content_view,
            addSubview: view,
            positioned: NSWindowOrderingMode::Below,
            relativeTo: ptr::null_mut::<AnyObject>()
        ];
        VIEWS.with_borrow_mut(|views| {
            views.push(LiveView {
                view,
                layer,
                live,
                cover: None,
                wants_motion: false,
                display: None,
                style_frame: None,
                fading_from: None,
                fade_started: None,
            })
        });
        window_video::observe_conditions(view);
        window_video::observe_power_source();
        draw(view);
        refresh(view);
        let _: *mut AnyObject = msg_send![view, autorelease];
        Some(view)
    }
}

pub(crate) unsafe fn remove_view(view: *mut AnyObject) {
    unsafe {
        detach(view);
        let _: () = msg_send![view, removeFromSuperview];
    }
}

/// Stops `view` drawing and observing. Runs once: from `remove_view`, or from the view's dealloc
/// when its window went away without removing it.
unsafe fn detach(view: *mut AnyObject) {
    let removed = VIEWS.with_borrow_mut(|views| {
        let before = views.len();
        views.retain(|entry| entry.view != view);
        views.len() != before
    });
    if removed {
        unsafe { window_video::stop_observing_conditions(view) };
        update_clock();
    }
}

/// The style, colours, brightness, speed or battery rule `view` draws with. A new picture fades in
/// from the one on screen (at once under Reduce Motion). Returns false when the new style is not
/// one this platform draws, so the window can leave the live backdrop.
pub(crate) unsafe fn update(view: *mut AnyObject, live: LiveBackground) -> bool {
    let Some(style) = style_id(&live.style) else {
        return false;
    };
    if with_renderer(|renderer| pipeline(renderer, style).is_some()) != Some(true) {
        return false;
    }
    let fade = unsafe { !window_video::reduce_motion() };
    VIEWS.with_borrow_mut(|views| {
        let Some(entry) = views.iter_mut().find(|entry| entry.view == view) else {
            return;
        };
        let picture_changes = entry.live.style != live.style
            || entry.live.colors != live.colors
            || entry.live.brightness != live.brightness;
        if picture_changes && fade && entry.display.is_some() {
            // Whatever is on screen right now, a fade in progress included, is where this one
            // starts, so a change during a change never jumps either.
            entry.fading_from = entry.display.take();
            entry.fade_started = Some(Instant::now());
        }
        entry.live = live;
    });
    unsafe {
        draw(view);
        refresh(view);
    }
    true
}

/// The rectangle, in window coordinates with the origin at the top left, the style is laid out
/// over, for a window laid over part of another that must line up with it; `None` covers the view.
pub(crate) unsafe fn set_cover(view: *mut AnyObject, cover: Option<NSRect>) {
    let changed = VIEWS.with_borrow_mut(|views| {
        views
            .iter_mut()
            .find(|entry| entry.view == view)
            .is_some_and(|entry| {
                let changed = !same_rect(entry.cover, cover);
                entry.cover = cover;
                changed
            })
    });
    if changed {
        unsafe { draw(view) };
    }
}

fn same_rect(a: Option<NSRect>, b: Option<NSRect>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a.origin.x == b.origin.x
                && a.origin.y == b.origin.y
                && a.size.width == b.size.width
                && a.size.height == b.size.height
        }
        _ => false,
    }
}

/// Re-reads whether every live view may move, after the charger was plugged in or out.
pub(crate) fn refresh_all() {
    let views: Vec<*mut AnyObject> =
        VIEWS.with_borrow(|views| views.iter().map(|entry| entry.view).collect());
    for view in views {
        unsafe { refresh(view) };
    }
}

fn style_id(style: &str) -> Option<&'static str> {
    LIVE_STYLES.iter().copied().find(|known| *known == style)
}

fn with_renderer<R>(action: impl FnOnce(&mut LiveRenderer) -> R) -> Option<R> {
    RENDERER.with_borrow_mut(|slot| {
        if slot.is_none() {
            let device = MTLCreateSystemDefaultDevice()?;
            let library = device
                .newLibraryWithSource_options_error(
                    &NSString::from_str(SHADER_SOURCE),
                    Some(&MTLCompileOptions::new()),
                )
                .map_err(|error| {
                    log::error!(
                        "live backdrop shaders do not compile: {}",
                        error.localizedDescription()
                    )
                })
                .ok()?;
            let queue = device.newCommandQueue()?;
            *slot = Some(LiveRenderer {
                device,
                queue,
                library,
                pipelines: HashMap::new(),
            });
        }
        slot.as_mut().map(action)
    })
}

/// The pipeline drawing `name`: a style id, or "composite" for the fade.
fn pipeline(renderer: &mut LiveRenderer, name: &'static str) -> Option<Pipeline> {
    if let Some(pipeline) = renderer.pipelines.get(name) {
        return Some(pipeline.clone());
    }
    let vertex = renderer
        .library
        .newFunctionWithName(&NSString::from_str("live_vertex"))?;
    let Some(fragment) = renderer
        .library
        .newFunctionWithName(&NSString::from_str(&format!("live_{name}")))
    else {
        log::error!("live backdrop function {name} is missing");
        return None;
    };
    let descriptor = MTLRenderPipelineDescriptor::new();
    descriptor.setVertexFunction(Some(&vertex));
    descriptor.setFragmentFunction(Some(&fragment));
    // SAFETY: Index zero is within the pipeline's color attachment array.
    unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) }
        .setPixelFormat(MTLPixelFormat::BGRA8Unorm);
    let pipeline = renderer
        .device
        .newRenderPipelineStateWithDescriptor_error(&descriptor)
        .map_err(|error| {
            log::error!(
                "live backdrop {name} has no pipeline: {}",
                error.localizedDescription()
            )
        })
        .ok()?;
    renderer.pipelines.insert(name, pipeline.clone());
    Some(pipeline)
}

fn offscreen_texture(
    device: &ProtocolObject<dyn MTLDevice>,
    width: usize,
    height: usize,
) -> Option<Texture> {
    let descriptor = MTLTextureDescriptor::new();
    descriptor.setTextureType(MTLTextureType::Type2D);
    descriptor.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
    // SAFETY: Both sizes are at least 16 and this descriptor creates a 2D texture.
    unsafe {
        descriptor.setWidth(width);
        descriptor.setHeight(height);
    }
    descriptor.setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
    descriptor.setStorageMode(MTLStorageMode::Private);
    device.newTextureWithDescriptor(&descriptor)
}

fn texture_matches(texture: &Option<Texture>, width: usize, height: usize) -> bool {
    texture
        .as_ref()
        .is_some_and(|texture| texture.width() == width && texture.height() == height)
}

/// Draws one frame of `view` at the shared clock's phase.
unsafe fn draw(view: *mut AnyObject) {
    let phase = CLOCK.with_borrow(|clock| clock.phase);
    VIEWS.with_borrow_mut(|views| {
        let Some(entry) = views.iter_mut().find(|entry| entry.view == view) else {
            return;
        };
        unsafe { draw_entry(entry, phase) };
    });
}

/// How far the fade has come, eased in and out; `None` once it is over.
fn fade_amount(started: Option<Instant>) -> Option<f32> {
    let elapsed = started?.elapsed().as_secs_f32() / LIVE_FADE.as_secs_f32();
    if elapsed >= 1.0 {
        return None;
    }
    let t = elapsed.max(0.0);
    Some(t * t * (3.0 - 2.0 * t))
}

fn begin_pass(
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    target: &ProtocolObject<dyn MTLTexture>,
) -> Option<Retained<ProtocolObject<dyn MTLRenderCommandEncoder>>> {
    let pass = MTLRenderPassDescriptor::new();
    // SAFETY: Index zero is within the render pass's color attachment array.
    let attachment = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
    attachment.setTexture(Some(target));
    // Every pass covers its whole target with one triangle.
    attachment.setLoadAction(MTLLoadAction::DontCare);
    attachment.setStoreAction(MTLStoreAction::Store);
    command_buffer.renderCommandEncoderWithDescriptor(&pass)
}

unsafe fn draw_entry(entry: &mut LiveView, phase: f64) {
    unsafe {
        let bounds: NSRect = msg_send![entry.view, bounds];
        if bounds.size.width < 1.0 || bounds.size.height < 1.0 {
            return;
        }
        let Some(style) = style_id(&entry.live.style) else {
            return;
        };
        let factor = (1.0 / LIVE_DOWNSCALE)
            .min(LIVE_MAX_DRAWABLE.0 / bounds.size.width)
            .min(LIVE_MAX_DRAWABLE.1 / bounds.size.height);
        let width = (bounds.size.width * factor).round().max(16.0);
        let height = (bounds.size.height * factor).round().max(16.0);
        let current = entry.layer.drawableSize();
        if current.width != width || current.height != height {
            entry.layer.setDrawableSize(NSSize::new(width, height));
        }
        let fade = fade_amount(entry.fade_started);
        if fade.is_none() {
            entry.fading_from = None;
            entry.fade_started = None;
            entry.style_frame = None;
        }
        let cover = entry
            .cover
            .unwrap_or(NSRect::new(NSPoint::new(0.0, 0.0), bounds.size));
        let color = |index: usize| {
            let [red, green, blue] = entry.live.colors[index];
            [red, green, blue, 1.0]
        };
        let uniforms = LiveUniforms {
            resolution: [width as f32, height as f32],
            view_size: [bounds.size.width as f32, bounds.size.height as f32],
            cover_origin: [cover.origin.x as f32, cover.origin.y as f32],
            cover_size: [cover.size.width as f32, cover.size.height as f32],
            phase: phase as f32,
            period: LIVE_PERIOD as f32,
            brightness: entry.live.brightness.clamp(0.0, 1.0),
            pad: 0.0,
            c0: color(0),
            c1: color(1),
            c2: color(2),
        };
        let (texture_width, texture_height) = (width as usize, height as usize);
        with_renderer(|renderer| {
            let style_pipeline = pipeline(renderer, style)?;
            if !texture_matches(&entry.display, texture_width, texture_height) {
                entry.display = offscreen_texture(&renderer.device, texture_width, texture_height);
            }
            let display = entry.display.clone()?;
            let composite = match fade {
                Some(_) => Some(pipeline(renderer, "composite")?),
                None => None,
            };
            let command_buffer = renderer.queue.commandBuffer()?;
            let draw_style = |target: &ProtocolObject<dyn MTLTexture>| -> Option<()> {
                let encoder = begin_pass(&command_buffer, target)?;
                encoder.setRenderPipelineState(&style_pipeline);
                // SAFETY: The uniforms are copied from a live value into the slot the style's
                // fragment shader reads them from, and the triangle is generated from the vertex id.
                encoder.setFragmentBytes_length_atIndex(
                    NonNull::from(&uniforms).cast(),
                    std::mem::size_of::<LiveUniforms>(),
                    0,
                );
                encoder.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3);
                encoder.endEncoding();
                Some(())
            };
            match (fade, entry.fading_from.clone(), composite) {
                (Some(amount), Some(from), Some(composite)) => {
                    if !texture_matches(&entry.style_frame, texture_width, texture_height) {
                        entry.style_frame =
                            offscreen_texture(&renderer.device, texture_width, texture_height);
                    }
                    let style_frame = entry.style_frame.clone()?;
                    draw_style(&style_frame)?;
                    let encoder = begin_pass(&command_buffer, &display)?;
                    encoder.setRenderPipelineState(&composite);
                    // SAFETY: Both textures are retained by Metal for the pass, the amount is
                    // copied from a live value, and the triangle is generated from the vertex id.
                    encoder.setFragmentTexture_atIndex(Some(&from), 0);
                    encoder.setFragmentTexture_atIndex(Some(&style_frame), 1);
                    encoder.setFragmentBytes_length_atIndex(
                        NonNull::from(&amount).cast(),
                        std::mem::size_of::<f32>(),
                        0,
                    );
                    encoder.drawPrimitives_vertexStart_vertexCount(
                        MTLPrimitiveType::Triangle,
                        0,
                        3,
                    );
                    encoder.endEncoding();
                }
                _ => draw_style(&display)?,
            }
            let drawable = entry.layer.nextDrawable()?;
            let blit = command_buffer.blitCommandEncoder()?;
            // SAFETY: Both textures are BGRA8 and at least `texture_width` x `texture_height`:
            // the drawable was sized to them above.
            blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                &display,
                0,
                0,
                MTLOrigin { x: 0, y: 0, z: 0 },
                MTLSize {
                    width: texture_width,
                    height: texture_height,
                    depth: 1,
                },
                &drawable.texture(),
                0,
                0,
                MTLOrigin { x: 0, y: 0, z: 0 },
            );
            blit.endEncoding();
            command_buffer.presentDrawable(ProtocolObject::from_ref(&*drawable));
            command_buffer.commit();
            Some(())
        });
    }
}

/// Works out whether `view` can be seen and should move, then starts or stops the shared clock.
unsafe fn refresh(view: *mut AnyObject) {
    unsafe {
        let window: *mut AnyObject = msg_send![view, window];
        let only_on_power = VIEWS.with_borrow(|views| {
            views
                .iter()
                .find(|entry| entry.view == view)
                .map(|entry| entry.live.only_on_power)
        });
        let Some(only_on_power) = only_on_power else {
            return;
        };
        let wants = window_video::window_may_animate(window, only_on_power);
        VIEWS.with_borrow_mut(|views| {
            if let Some(entry) = views.iter_mut().find(|entry| entry.view == view) {
                entry.wants_motion = wants;
            }
        });
        update_clock();
    }
}

/// Runs the shared timer while any live view may move or is fading, and stops it otherwise.
fn update_clock() {
    let any = VIEWS.with_borrow(|views| {
        views
            .iter()
            .any(|entry| entry.wants_motion || entry.fade_started.is_some())
    });
    CLOCK.with_borrow_mut(|clock| unsafe {
        if any && clock.timer.is_null() {
            let ticker: *mut AnyObject = msg_send![ticker_class(), new];
            let timer: *mut AnyObject = msg_send![
                class!(NSTimer),
                timerWithTimeInterval: LIVE_FRAME_INTERVAL,
                target: ticker,
                selector: sel!(tick:),
                userInfo: ptr::null_mut::<AnyObject>(),
                repeats: Bool::YES
            ];
            // The timer keeps the ticker; the common modes keep it running through a window drag.
            let _: () = msg_send![ticker, release];
            let run_loop: *mut AnyObject = msg_send![class!(NSRunLoop), mainRunLoop];
            let _: () = msg_send![
                run_loop,
                addTimer: timer,
                forMode: crate::ns_string("kCFRunLoopCommonModes")
            ];
            let _: *mut AnyObject = msg_send![timer, retain];
            clock.timer = timer;
            clock.last_tick = None;
        } else if !any && !clock.timer.is_null() {
            let _: () = msg_send![clock.timer, invalidate];
            let _: () = msg_send![clock.timer, release];
            clock.timer = ptr::null_mut();
            clock.last_tick = None;
        }
    });
}

extern "C" fn tick(_: &AnyObject, _: Sel, _: *mut AnyObject) {
    let speed = VIEWS.with_borrow(|views| {
        views
            .iter()
            .find(|entry| entry.wants_motion)
            .map(|entry| f64::from(entry.live.speed.clamp(0.05, 4.0)))
    });
    let phase = CLOCK.with_borrow_mut(|clock| {
        let now = Instant::now();
        match speed {
            // Only time someone could see the motion in moves the clock; a paused stretch, or
            // the first tick after one, adds nothing, so motion resumes exactly where it stopped.
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
    VIEWS.with_borrow_mut(|views| {
        for entry in views
            .iter_mut()
            .filter(|entry| entry.wants_motion || entry.fade_started.is_some())
        {
            unsafe { draw_entry(entry, phase) };
        }
    });
    update_clock();
}

fn ticker_class() -> *const AnyClass {
    *TICKER_CLASS.get_or_init(|| {
        let mut decl = ClassBuilder::new(c"GPUILiveBackdropTicker", class!(NSObject)).unwrap();
        unsafe {
            decl.add_method(sel!(tick:), tick as extern "C" fn(_, Sel, *mut AnyObject));
        }
        decl.register() as *const AnyClass as usize
    }) as *const AnyClass
}

fn view_class() -> *const AnyClass {
    *VIEW_CLASS.get_or_init(|| {
        let mut decl = ClassBuilder::new(c"GPUILiveBackdropView", class!(NSView)).unwrap();
        unsafe {
            // `window_video::observe_conditions` registers this selector.
            decl.add_method(
                sel!(videoConditionsDidChange:),
                live_conditions_did_change as extern "C" fn(_, Sel, *mut AnyObject),
            );
            decl.add_method(
                sel!(refreshLiveBackdrop),
                refresh_live_backdrop as extern "C" fn(_, Sel),
            );
            decl.add_method(
                sel!(viewDidMoveToWindow),
                view_did_move_to_window as extern "C" fn(_, Sel),
            );
            decl.add_method(
                sel!(setFrameSize:),
                set_frame_size as extern "C" fn(_, Sel, NSSize),
            );
            decl.add_method(sel!(dealloc), dealloc_view as extern "C" fn(_, Sel));
        }
        decl.register() as *const AnyClass as usize
    }) as *const AnyClass
}

extern "C" fn live_conditions_did_change(this: &AnyObject, _: Sel, notification: *mut AnyObject) {
    unsafe {
        let this_id = this as *const AnyObject as *mut AnyObject;
        window_video::note_condition_notification(this_id, notification);
        // Low Power Mode changes arrive on a background queue.
        let main: Bool = msg_send![class!(NSThread), isMainThread];
        if main.as_bool() {
            refresh(this_id);
        } else {
            let _: () = msg_send![
                this_id,
                performSelectorOnMainThread: sel!(refreshLiveBackdrop),
                withObject: ptr::null_mut::<AnyObject>(),
                waitUntilDone: Bool::NO
            ];
        }
    }
}

extern "C" fn refresh_live_backdrop(this: &AnyObject, _: Sel) {
    unsafe { refresh(this as *const AnyObject as *mut AnyObject) };
}

extern "C" fn view_did_move_to_window(this: &AnyObject, _: Sel) {
    unsafe { refresh(this as *const AnyObject as *mut AnyObject) };
}

/// A resize while the motion is paused (Reduce Motion, the app in the background) still redraws,
/// so the still frame is never stretched.
extern "C" fn set_frame_size(this: &AnyObject, _: Sel, size: NSSize) {
    unsafe {
        let _: () = msg_send![super(this, class!(NSView)), setFrameSize: size];
        draw(this as *const AnyObject as *mut AnyObject);
    }
}

extern "C" fn dealloc_view(this: &AnyObject, _: Sel) {
    unsafe {
        detach(this as *const AnyObject as *mut AnyObject);
        let _: () = msg_send![super(this, class!(NSView)), dealloc];
    }
}
