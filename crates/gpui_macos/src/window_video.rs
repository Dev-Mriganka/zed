//! The video backdrop of a wallpaper-mode window: a looping, muted video blurred like the
//! wallpaper picture and placed the same way, which plays only while someone can see it.
//!
//! Windows showing the same file share one player, so a window laid over another (a floating
//! panel) shows exactly the same frame as the window under it, and the video is decoded once.

use core_foundation::{
    base::{CFRelease, CFTypeRef, TCFType},
    string::{CFString, CFStringRef},
};
use core_foundation_sys::runloop::{
    CFRunLoopAddSource, CFRunLoopGetMain, CFRunLoopSourceRef, kCFRunLoopDefaultMode,
};
use objc2::{
    Encode, Encoding, class, msg_send,
    runtime::{AnyClass, AnyObject, Bool, ClassBuilder, Sel},
    sel,
};
use objc2_app_kit::{NSAutoresizingMaskOptions, NSWindowOrderingMode};
use objc2_core_graphics::CGColor;
use objc2_foundation::{NSPoint, NSRect, NSSize};
use parking_lot::Mutex;
use std::{
    ffi::{CStr, c_void},
    path::Path,
    ptr,
    sync::{
        OnceLock,
        atomic::{AtomicBool, Ordering},
    },
};

use crate::{ns_string, window_wallpaper};

type Id = *mut AnyObject;
const NIL: Id = ptr::null_mut();

/// The value of `object`'s ivar `name`.
unsafe fn ivar<T: objc2::Encode + Copy>(object: &AnyObject, name: &CStr) -> T {
    unsafe {
        *object
            .class()
            .instance_variable(name)
            .expect("GPUI backdrop ivar")
            .load::<T>(object)
    }
}

/// Sets `object`'s ivar `name`.
unsafe fn set_ivar<T: objc2::Encode>(object: &AnyObject, name: &CStr, value: T) {
    unsafe {
        object
            .class()
            .instance_variable(name)
            .expect("GPUI backdrop ivar")
            .load_ptr::<T>(object)
            .write(value);
    }
}

#[link(name = "AVFoundation", kind = "framework")]
unsafe extern "C" {
    static AVLayerVideoGravityResizeAspectFill: Id;
}

#[link(name = "CoreMedia", kind = "framework")]
unsafe extern "C" {
    static kCMTimeZero: CMTime;
}

#[link(name = "AppKit", kind = "framework")]
unsafe extern "C" {
    static NSApplicationDidBecomeActiveNotification: Id;
    static NSApplicationDidResignActiveNotification: Id;
    static NSApplicationDidChangeScreenParametersNotification: Id;
    static NSWindowDidChangeOcclusionStateNotification: Id;
    static NSWindowDidMiniaturizeNotification: Id;
    static NSWindowDidDeminiaturizeNotification: Id;
    static NSWorkspaceScreensDidSleepNotification: Id;
    static NSWorkspaceScreensDidWakeNotification: Id;
    static NSWorkspaceAccessibilityDisplayOptionsDidChangeNotification: Id;
    static NSWorkspaceActiveSpaceDidChangeNotification: Id;
}

#[link(name = "Foundation", kind = "framework")]
unsafe extern "C" {
    static NSProcessInfoPowerStateDidChangeNotification: Id;
}

#[link(name = "IOKit", kind = "framework")]
unsafe extern "C" {
    fn IOPSCopyPowerSourcesInfo() -> CFTypeRef;
    fn IOPSGetProvidingPowerSourceType(snapshot: CFTypeRef) -> CFStringRef;
    fn IOPSNotificationCreateRunLoopSource(
        callback: extern "C" fn(*mut c_void),
        context: *mut c_void,
    ) -> CFRunLoopSourceRef;
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CMTime {
    value: i64,
    timescale: i32,
    flags: u32,
    epoch: i64,
}

// SAFETY: matches CoreMedia's `CMTime`, an unnamed struct of a CMTimeValue (int64_t), a
// CMTimeScale (int32_t), CMTimeFlags (uint32_t) and a CMTimeEpoch (int64_t).
unsafe impl Encode for CMTime {
    const ENCODING: Encoding = Encoding::Struct(
        "?",
        &[i64::ENCODING, i32::ENCODING, u32::ENCODING, i64::ENCODING],
    );
}

/// How far the video layer reaches past every edge of the area it covers under a blur of `radius`
/// points, so the blur's transparent fringe falls outside the window instead of darkening its
/// edges.
fn video_bleed(radius: f64) -> f64 {
    radius.max(0.0) * 2.0
}

static VIEW_CLASS: OnceLock<usize> = OnceLock::new();

/// Every live video backdrop view.
static VIEWS: Mutex<Vec<usize>> = Mutex::new(Vec::new());

/// One player per file, shared by every view showing it: (path, AVQueuePlayer, AVPlayerLooper,
/// views using it). The player and looper each hold one retain.
static PLAYERS: Mutex<Vec<(String, usize, usize, usize)>> = Mutex::new(Vec::new());

/// Whether the displays are asleep, from the workspace's sleep and wake notifications.
static SCREENS_ASLEEP: AtomicBool = AtomicBool::new(false);

static POWER_SOURCE_OBSERVED: OnceLock<()> = OnceLock::new();

/// Adds a video backdrop playing `path` below everything in `content_view`, blurred by
/// `blur_radius` points, or `None` when the file does not exist.
pub(crate) unsafe fn create_view(
    content_view: Id,
    path: &Path,
    only_on_power: bool,
    blur_radius: f64,
) -> Option<Id> {
    unsafe {
        let path_string = path.to_str()?.to_string();
        if !path.is_file() {
            return None;
        }
        let player = acquire_player(&path_string)?;
        let class = view_class();
        let view: Id = msg_send![class, alloc];
        let content_bounds: NSRect = msg_send![content_view, bounds];
        let view: Id = msg_send![view, initWithFrame: content_bounds];
        let _: () = msg_send![
            view,
            setAutoresizingMask: NSAutoresizingMaskOptions::ViewWidthSizable
                | NSAutoresizingMaskOptions::ViewHeightSizable
        ];
        let _: () = msg_send![view, setWantsLayer: Bool::YES];
        // Layer filters only run on a view that asks for them.
        let _: () = msg_send![view, setLayerUsesCoreImageFilters: Bool::YES];
        let layer: Id = msg_send![view, layer];
        let black: Id = msg_send![class!(NSColor), blackColor];
        let black: *mut CGColor = msg_send![black, CGColor];
        let _: () = msg_send![layer, setBackgroundColor: black];
        // The blur's bleed reaches past the view; keep it inside.
        let _: () = msg_send![layer, setMasksToBounds: Bool::YES];

        let video_layer: Id = msg_send![class!(AVPlayerLayer), playerLayerWithPlayer: player];
        let _: () = msg_send![video_layer, setVideoGravity: AVLayerVideoGravityResizeAspectFill];
        // `window_wallpaper::layout` places the sublayer with this name.
        let _: () = msg_send![video_layer, setName: ns_string("wallpaper")];
        apply_blur(video_layer, blur_radius);
        let _: () = msg_send![layer, addSublayer: video_layer];

        let object = &*view;
        set_ivar::<f64>(object, c"gpuiBlurRadius", blur_radius);
        set_ivar::<Id>(object, c"gpuiPlayer", player);
        set_ivar::<Bool>(
            object,
            c"gpuiOnlyOnPower",
            if only_on_power { Bool::YES } else { Bool::NO },
        );
        set_ivar::<Bool>(object, c"gpuiWantsPlay", Bool::NO);
        let path_ns: Id = msg_send![ns_string(&path_string), retain];
        set_ivar::<Id>(object, c"gpuiPath", path_ns);

        let _: () = msg_send![
            content_view,
            addSubview: view,
            positioned: NSWindowOrderingMode::Below,
            relativeTo: NIL
        ];
        VIEWS.lock().push(view as usize);
        observe_conditions(view);
        observe_power_source();
        refresh(view);
        let _: Id = msg_send![view, autorelease];
        Some(view)
    }
}

pub(crate) unsafe fn remove_view(view: Id) {
    unsafe {
        detach(view);
        let _: () = msg_send![view, removeFromSuperview];
    }
}

/// Stops `view` observing and hands back its share of the player. Runs once: from `remove_view`,
/// or from the view's dealloc when its window went away without removing it.
unsafe fn detach(view: Id) {
    unsafe {
        let registered = {
            let mut views = VIEWS.lock();
            let before = views.len();
            views.retain(|existing| *existing != view as usize);
            views.len() != before
        };
        if !registered {
            return;
        }
        stop_observing_conditions(view);
        let object = &*view;
        set_ivar::<Bool>(object, c"gpuiWantsPlay", Bool::NO);
        let path: Id = ivar::<Id>(object, c"gpuiPath");
        let player: Id = ivar::<Id>(object, c"gpuiPlayer");
        if let Some(path) = ns_to_string(path) {
            update_player(player);
            release_player(&path);
        }
        let _: () = msg_send![path, release];
        set_ivar::<Id>(object, c"gpuiPath", NIL);
    }
}

extern "C" fn dealloc_view(this: &AnyObject, _: Sel) {
    unsafe {
        let this_id = this as *const AnyObject as Id;
        detach(this_id);
        let _: () = msg_send![super(this, class!(NSView)), dealloc];
    }
}

/// The file `view` plays.
pub(crate) unsafe fn view_path(view: Id) -> Option<String> {
    unsafe {
        let object = &*view;
        ns_to_string(ivar::<Id>(object, c"gpuiPath"))
    }
}

pub(crate) unsafe fn set_only_on_power(view: Id, only_on_power: bool) {
    unsafe {
        let object = &*view;
        set_ivar::<Bool>(
            object,
            c"gpuiOnlyOnPower",
            if only_on_power { Bool::YES } else { Bool::NO },
        );
        refresh(view);
    }
}

/// Sets the video layer's blur filter: `radius` points, or none at 0.
unsafe fn apply_blur(video_layer: Id, radius: f64) {
    unsafe {
        let filters: Id = if radius > 0.0 {
            let blur: Id = msg_send![class!(CIFilter), filterWithName: ns_string("CIGaussianBlur")];
            let _: () = msg_send![blur, setDefaults];
            let radius: Id = msg_send![class!(NSNumber), numberWithDouble: radius];
            let _: () = msg_send![blur, setValue: radius, forKey: ns_string("inputRadius")];
            msg_send![class!(NSArray), arrayWithObject: blur]
        } else {
            NIL
        };
        let _: () = msg_send![video_layer, setFilters: filters];
    }
}

/// Changes the blur of a video backdrop. The caller lays it out again, since the bleed follows the
/// radius.
pub(crate) unsafe fn set_blur_radius(view: Id, radius: f64) {
    unsafe {
        let object = &*view;
        set_ivar::<f64>(object, c"gpuiBlurRadius", radius);
        let layer: Id = msg_send![view, layer];
        let sublayers: Id = msg_send![layer, sublayers];
        if sublayers.is_null() {
            return;
        }
        let video_layer: Id = msg_send![sublayers, firstObject];
        if !video_layer.is_null() {
            apply_blur(video_layer, radius);
        }
    }
}

/// Places the video like the wallpaper picture, then lets it reach past every edge by the blur's
/// bleed so the edges stay as bright as the middle.
pub(crate) unsafe fn layout(
    view: Id,
    window: Id,
    screen_frame: NSRect,
    follows_screen: bool,
    cover: Option<NSRect>,
) {
    unsafe {
        window_wallpaper::layout(view, window, screen_frame, follows_screen, cover);
        let layer: Id = msg_send![view, layer];
        let sublayers: Id = msg_send![layer, sublayers];
        if sublayers.is_null() {
            return;
        }
        let video_layer: Id = msg_send![sublayers, firstObject];
        if video_layer.is_null() {
            return;
        }
        let frame: NSRect = msg_send![video_layer, frame];
        let object = &*view;
        let bleed = video_bleed(ivar::<f64>(object, c"gpuiBlurRadius"));
        let bled = NSRect::new(
            NSPoint::new(frame.origin.x - bleed, frame.origin.y - bleed),
            NSSize::new(
                frame.size.width + bleed * 2.0,
                frame.size.height + bleed * 2.0,
            ),
        );
        let _: () = msg_send![class!(CATransaction), begin];
        let _: () = msg_send![class!(CATransaction), setDisableActions: Bool::YES];
        // The picture layout resizes with the view on its own; the bled frame is set by hand, so
        // it is redone on every layout (`window_did_resize` calls back here).
        let _: () = msg_send![video_layer, setAutoresizingMask: 0u32];
        let _: () = msg_send![video_layer, setFrame: bled];
        let _: () = msg_send![class!(CATransaction), commit];
    }
}

unsafe fn acquire_player(path: &str) -> Option<Id> {
    unsafe {
        let mut players = PLAYERS.lock();
        if let Some(entry) = players.iter_mut().find(|(existing, ..)| existing == path) {
            entry.3 += 1;
            return Some(entry.1 as Id);
        }
        let url: Id = msg_send![class!(NSURL), fileURLWithPath: ns_string(path)];
        let item: Id = msg_send![class!(AVPlayerItem), playerItemWithURL: url];
        if item.is_null() {
            return None;
        }
        let player: Id = msg_send![class!(AVQueuePlayer), alloc];
        let player: Id = msg_send![player, init];
        if player.is_null() {
            return None;
        }
        let _: () = msg_send![player, setMuted: Bool::YES];
        // A background must never keep the display awake or show up as media playing.
        let _: () = msg_send![player, setPreventsDisplaySleepDuringVideoPlayback: Bool::NO];
        let looper: Id = msg_send![
            class!(AVPlayerLooper),
            playerLooperWithPlayer: player,
            templateItem: item
        ];
        let looper: Id = msg_send![looper, retain];
        let _: () = msg_send![player, pause];
        players.push((path.to_string(), player as usize, looper as usize, 1));
        Some(player)
    }
}

unsafe fn release_player(path: &str) {
    unsafe {
        let mut players = PLAYERS.lock();
        let Some(index) = players.iter().position(|(existing, ..)| existing == path) else {
            return;
        };
        players[index].3 -= 1;
        if players[index].3 > 0 {
            return;
        }
        let (_, player, looper, _) = players.remove(index);
        let player = player as Id;
        let looper = looper as Id;
        let _: () = msg_send![player, pause];
        let _: () = msg_send![looper, disableLooping];
        let _: () = msg_send![looper, release];
        let _: () = msg_send![player, release];
    }
}

/// Works out whether `view` can be seen and should move, then plays or pauses its player.
unsafe fn refresh(view: Id) {
    unsafe {
        let wants = view_wants_play(view);
        let object = &*view;
        set_ivar::<Bool>(
            object,
            c"gpuiWantsPlay",
            if wants { Bool::YES } else { Bool::NO },
        );
        let player: Id = ivar::<Id>(object, c"gpuiPlayer");
        update_player(player);
    }
}

/// Plays `player` while any view showing it wants to, and pauses it otherwise. Under Reduce Motion
/// the paused video goes back to its first frame, so the window shows a still picture.
unsafe fn update_player(player: Id) {
    unsafe {
        let wants = VIEWS.lock().iter().any(|view| {
            let object = &*(*view as *const AnyObject);
            ivar::<Id>(object, c"gpuiPlayer") == player
                && ivar::<Bool>(object, c"gpuiWantsPlay").as_bool()
        });
        let rate: f32 = msg_send![player, rate];
        if wants {
            if rate == 0.0 {
                let _: () = msg_send![player, play];
            }
        } else {
            if rate != 0.0 {
                let _: () = msg_send![player, pause];
            }
            if reduce_motion() {
                let _: () = msg_send![player, seekToTime: kCMTimeZero];
            }
        }
    }
}

unsafe fn view_wants_play(view: Id) -> bool {
    unsafe {
        let window: Id = msg_send![view, window];
        let object = &*view;
        let only_on_power = ivar::<Bool>(object, c"gpuiOnlyOnPower").as_bool();
        window_may_animate(window, only_on_power)
    }
}

/// Whether a moving backdrop in `window` may move right now: someone can see it, and nothing asks
/// it to hold still. The live backdrop (window_live.rs) moves under exactly the same rules.
pub(crate) unsafe fn window_may_animate(window: Id, only_on_power: bool) -> bool {
    unsafe {
        if window.is_null() {
            return false;
        }
        let app: Id = msg_send![class!(NSApplication), sharedApplication];
        let active: Bool = msg_send![app, isActive];
        let miniaturized: Bool = msg_send![window, isMiniaturized];
        // NSWindowOcclusionStateVisible
        let occlusion: u64 = msg_send![window, occlusionState];
        let visible = occlusion & (1 << 1) != 0;
        let process: Id = msg_send![class!(NSProcessInfo), processInfo];
        let low_power: Bool = msg_send![process, isLowPowerModeEnabled];
        active.as_bool()
            && !miniaturized.as_bool()
            && visible
            && !SCREENS_ASLEEP.load(Ordering::Relaxed)
            && !low_power.as_bool()
            && !reduce_motion()
            && (!only_on_power || on_external_power())
    }
}

pub(crate) unsafe fn reduce_motion() -> bool {
    unsafe {
        let workspace: Id = msg_send![class!(NSWorkspace), sharedWorkspace];
        let reduce: Bool = msg_send![workspace, accessibilityDisplayShouldReduceMotion];
        reduce.as_bool()
    }
}

/// Whether the computer is running on a charger rather than its battery. A desktop always is.
fn on_external_power() -> bool {
    unsafe {
        let snapshot = IOPSCopyPowerSourcesInfo();
        if snapshot.is_null() {
            return true;
        }
        let source = IOPSGetProvidingPowerSourceType(snapshot);
        let on_power = source.is_null()
            || CFString::wrap_under_get_rule(source).to_string() != "Battery Power";
        CFRelease(snapshot);
        on_power
    }
}

fn view_class() -> *const AnyClass {
    *VIEW_CLASS.get_or_init(|| {
        let mut decl = ClassBuilder::new(c"GPUIVideoBackdropView", class!(NSView)).unwrap();
        decl.add_ivar::<Id>(c"gpuiPlayer");
        decl.add_ivar::<Id>(c"gpuiPath");
        decl.add_ivar::<Bool>(c"gpuiOnlyOnPower");
        decl.add_ivar::<Bool>(c"gpuiWantsPlay");
        decl.add_ivar::<f64>(c"gpuiBlurRadius");
        unsafe {
            decl.add_method(
                sel!(videoConditionsDidChange:),
                video_conditions_did_change as extern "C" fn(_, Sel, Id),
            );
            decl.add_method(
                sel!(refreshVideoPlayback),
                refresh_video_playback as extern "C" fn(_, Sel),
            );
            decl.add_method(
                sel!(viewDidMoveToWindow),
                view_did_move_to_window as extern "C" fn(_, Sel),
            );
            decl.add_method(sel!(dealloc), dealloc_view as extern "C" fn(_, Sel));
        }
        decl.register() as *const AnyClass as usize
    }) as *const AnyClass
}

/// Everything that decides whether the video can be seen or should move: the app coming to or
/// leaving the front, a window hidden, covered or minimised, the displays sleeping, Low Power
/// Mode, Reduce Motion. Screens and Spaces changing re-place the video like the picture.
pub(crate) unsafe fn observe_conditions(view: Id) {
    unsafe {
        let center: Id = msg_send![class!(NSNotificationCenter), defaultCenter];
        for name in [
            NSApplicationDidBecomeActiveNotification,
            NSApplicationDidResignActiveNotification,
            NSApplicationDidChangeScreenParametersNotification,
            NSWindowDidChangeOcclusionStateNotification,
            NSWindowDidMiniaturizeNotification,
            NSWindowDidDeminiaturizeNotification,
            NSProcessInfoPowerStateDidChangeNotification,
        ] {
            let _: () = msg_send![
                center,
                addObserver: view,
                selector: sel!(videoConditionsDidChange:),
                name: name,
                object: NIL
            ];
        }
        let workspace: Id = msg_send![class!(NSWorkspace), sharedWorkspace];
        let workspace_center: Id = msg_send![workspace, notificationCenter];
        for name in [
            NSWorkspaceScreensDidSleepNotification,
            NSWorkspaceScreensDidWakeNotification,
            NSWorkspaceAccessibilityDisplayOptionsDidChangeNotification,
            NSWorkspaceActiveSpaceDidChangeNotification,
        ] {
            let _: () = msg_send![
                workspace_center,
                addObserver: view,
                selector: sel!(videoConditionsDidChange:),
                name: name,
                object: NIL
            ];
        }
    }
}

pub(crate) unsafe fn stop_observing_conditions(view: Id) {
    unsafe {
        let center: Id = msg_send![class!(NSNotificationCenter), defaultCenter];
        let _: () = msg_send![center, removeObserver: view];
        let workspace: Id = msg_send![class!(NSWorkspace), sharedWorkspace];
        let workspace_center: Id = msg_send![workspace, notificationCenter];
        let _: () = msg_send![workspace_center, removeObserver: view];
    }
}

/// Plugging in or unplugging the charger, through IOKit's power source notification on the main
/// run loop. Registered once for the whole app.
pub(crate) fn observe_power_source() {
    POWER_SOURCE_OBSERVED.get_or_init(|| unsafe {
        let source =
            IOPSNotificationCreateRunLoopSource(power_source_did_change, std::ptr::null_mut());
        if !source.is_null() {
            CFRunLoopAddSource(CFRunLoopGetMain(), source, kCFRunLoopDefaultMode);
        }
    });
}

extern "C" fn power_source_did_change(_: *mut c_void) {
    let views: Vec<usize> = VIEWS.lock().clone();
    for view in views {
        unsafe { refresh(view as Id) };
    }
    crate::window_live::refresh_all();
}

extern "C" fn video_conditions_did_change(this: &AnyObject, _: Sel, notification: Id) {
    unsafe {
        let this_id = this as *const AnyObject as Id;
        note_condition_notification(this_id, notification);
        // Low Power Mode changes arrive on a background queue.
        let main: Bool = msg_send![class!(NSThread), isMainThread];
        if main.as_bool() {
            refresh(this_id);
        } else {
            let _: () = msg_send![
                this_id,
                performSelectorOnMainThread: sel!(refreshVideoPlayback),
                withObject: NIL,
                waitUntilDone: Bool::NO
            ];
        }
    }
}

/// What every moving backdrop does with one of the notifications `observe_conditions` registers:
/// tracks the displays sleeping, and re-places its window's backdrop when screens or Spaces change.
pub(crate) unsafe fn note_condition_notification(view: Id, notification: Id) {
    unsafe {
        let name: Id = if notification.is_null() {
            NIL
        } else {
            msg_send![notification, name]
        };
        if !name.is_null() {
            let is = |other: Id| -> bool {
                let equal: Bool = msg_send![name, isEqualToString: other];
                equal.as_bool()
            };
            if is(NSWorkspaceScreensDidSleepNotification) {
                SCREENS_ASLEEP.store(true, Ordering::Relaxed);
            } else if is(NSWorkspaceScreensDidWakeNotification) {
                SCREENS_ASLEEP.store(false, Ordering::Relaxed);
            }
            if is(NSApplicationDidChangeScreenParametersNotification)
                || is(NSWorkspaceActiveSpaceDidChangeNotification)
            {
                let _: Id = msg_send![view, retain];
                let _: Id = msg_send![view, autorelease];
                let window: Id = msg_send![view, window];
                if !window.is_null() {
                    crate::window::refresh_window_backdrop(&*window);
                }
            }
        }
    }
}

extern "C" fn refresh_video_playback(this: &AnyObject, _: Sel) {
    unsafe {
        let this_id = this as *const AnyObject as Id;
        if VIEWS.lock().contains(&(this_id as usize)) {
            refresh(this_id);
        }
    }
}

extern "C" fn view_did_move_to_window(this: &AnyObject, _: Sel) {
    unsafe {
        let this_id = this as *const AnyObject as Id;
        if VIEWS.lock().contains(&(this_id as usize)) {
            refresh(this_id);
        }
    }
}

unsafe fn ns_to_string(string: Id) -> Option<String> {
    unsafe {
        if string.is_null() {
            return None;
        }
        let utf8: *const std::os::raw::c_char = msg_send![string, UTF8String];
        if utf8.is_null() {
            return None;
        }
        Some(
            std::ffi::CStr::from_ptr(utf8)
                .to_string_lossy()
                .into_owned(),
        )
    }
}
