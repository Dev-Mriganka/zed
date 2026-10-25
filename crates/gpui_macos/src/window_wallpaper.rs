//! The wallpaper backdrop of a blurred window: the blurred desktop picture of the window's screen
//! (or a picture the app chose) in place of the live behind-window blur, either covering the window
//! and moving with it, or held still against the screen while the window moves.

use core_foundation::base::{CFRelease, CFTypeRef};
use objc2::{
    Encode, Encoding, RefEncode, class, msg_send,
    rc::autoreleasepool,
    runtime::{AnyClass, AnyObject, Bool, ClassBuilder, Sel},
    sel,
};
use objc2_app_kit::{NSAutoresizingMaskOptions, NSWindowOrderingMode};
use objc2_core_graphics::CGColor;
use objc2_foundation::{NSPoint, NSRect, NSSize};
use parking_lot::Mutex;
use std::{path::Path, ptr, sync::OnceLock};

use crate::{ns_string, window_wallpaper_system};

type Id = *mut AnyObject;
const NIL: Id = ptr::null_mut();

/// A Core Graphics image (`CGImageRef`), only ever handled by pointer.
#[repr(C)]
struct CGImage {
    _private: [u8; 0],
}

// SAFETY: `CGImageRef` is a pointer to the opaque `struct CGImage`.
unsafe impl RefEncode for CGImage {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Encoding::Struct("CGImage", &[]));
}

#[link(name = "AppKit", kind = "framework")]
unsafe extern "C" {
    static NSWorkspaceDesktopImageScalingKey: Id;
    static NSWorkspaceDesktopImageAllowClippingKey: Id;
    static NSWorkspaceDesktopImageFillColorKey: Id;
    static NSWorkspaceActiveSpaceDidChangeNotification: Id;
    static NSApplicationDidChangeScreenParametersNotification: Id;
    static NSApplicationDidBecomeActiveNotification: Id;
}

#[link(name = "CoreImage", kind = "framework")]
unsafe extern "C" {
    static kCIImageApplyOrientationProperty: Id;
}

/// The picture is blurred on a canvas this fraction of the screen's size in points, for a blur of
/// `radius` points. Under a 60pt blur no detail finer than a few points survives, so a quarter-size
/// image is indistinguishable once scaled up; a lighter blur keeps more detail and needs more of
/// the screen's size, up to all of it for an unblurred picture.
fn canvas_scale(radius: f64) -> f64 {
    if radius <= 0.0 {
        1.0
    } else {
        (15.0 / radius).clamp(0.25, 1.0)
    }
}

/// Pictures blurred so far, by wallpaper file, placement options and screen size. A handful covers
/// every display and Space; the oldest is dropped past that.
const CACHE_LIMIT: usize = 6;

/// CGImage pointers, kept as `usize` so the cache is `Send`; each holds one retain.
static CACHE: Mutex<Vec<(String, usize)>> = Mutex::new(Vec::new());

/// Created once; Core Image contexts are expensive and reusable.
static CONTEXT: OnceLock<usize> = OnceLock::new();

static VIEW_CLASS: OnceLock<usize> = OnceLock::new();

/// A blurred desktop picture ready to show, and the frame of the screen it belongs to.
pub(crate) struct Wallpaper {
    image: Id,
    pub(crate) screen_frame: NSRect,
}

/// The blurred picture behind `window`: `image` when the app chose one, otherwise the desktop
/// picture of the window's screen. `None` when that picture cannot be read.
///
/// macOS draws its built-in dynamic, aerial and colour wallpapers from extensions; for those
/// `desktopImageURLForScreen:` reports the system's default picture rather than what is on screen,
/// so that placeholder is replaced by a still of the built-in wallpaper
/// (`window_wallpaper_system`), and a wallpaper with no still keeps the live blur.
/// `blur_radius` is the window's backdrop blur in points (`set_background_blur_style`).
pub(crate) unsafe fn wallpaper_for_window(
    window: Id,
    image: Option<&Path>,
    blur_radius: f64,
) -> Option<Wallpaper> {
    unsafe {
        let screen: Id = msg_send![window, screen];
        if screen.is_null() {
            return None;
        }
        let screen_frame: NSRect = msg_send![screen, frame];
        let size = format!("{}x{}", screen_frame.size.width, screen_frame.size.height);
        let (url, placement, key) = match image {
            Some(image) => {
                let path = image.to_str()?;
                // A file replaced under the same name is a new picture.
                let modified = std::fs::metadata(image)
                    .and_then(|metadata| metadata.modified())
                    .ok()
                    .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                    .map_or(0, |since| since.as_millis());
                (
                    file_url(path),
                    Placement::fill(),
                    format!("image|{path}|{modified}|{size}"),
                )
            }
            None => desktop_picture(screen, &size)?,
        };
        let key = format!("{key}|blur{blur_radius}");
        if let Some(image) = cached(&key) {
            return Some(Wallpaper {
                image,
                screen_frame,
            });
        }
        let image = render_blurred(url, &placement, screen_frame.size, blur_radius)?;
        store(key, image);
        Some(Wallpaper {
            image,
            screen_frame,
        })
    }
}

/// The desktop picture of `screen`, how the desktop places it, and its cache key.
unsafe fn desktop_picture(screen: Id, size: &str) -> Option<(Id, Placement, String)> {
    unsafe {
        let workspace: Id = msg_send![class!(NSWorkspace), sharedWorkspace];
        let url: Id = msg_send![workspace, desktopImageURLForScreen: screen];
        if url.is_null() {
            return None;
        }
        let path = url_path(url)?;
        let resolved: Id = msg_send![url, URLByResolvingSymlinksInPath];
        let resolved_path = url_path(resolved).unwrap_or_default();
        if path == "/System/Library/CoreServices/DefaultDesktop.heic"
            || resolved_path.starts_with("/System/Library/Wallpapers/.default/")
        {
            let still = window_wallpaper_system::built_in_wallpaper_still(screen)?;
            let key = format!("built-in|{still}|{size}");
            return Some((file_url(&still), Placement::fill(), key));
        }
        let options: Id = msg_send![workspace, desktopImageOptionsForScreen: screen];
        let placement = Placement::from_options(options);
        let key = format!("{path}|{size}|{placement:?}");
        Some((url, placement, key))
    }
}

unsafe fn file_url(path: &str) -> Id {
    unsafe { msg_send![class!(NSURL), fileURLWithPath: ns_string(path)] }
}

/// Adds the backdrop view below everything in `content_view`.
pub(crate) unsafe fn create_view(content_view: Id) -> Id {
    unsafe {
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
        let layer: Id = msg_send![view, layer];
        // Window snapshots (Mission Control, the app switcher) and the moment before the picture
        // lands show this base, the same black the live backdrop keeps under its blur.
        let black: Id = msg_send![class!(NSColor), blackColor];
        let black: *mut CGColor = msg_send![black, CGColor];
        let _: () = msg_send![layer, setBackgroundColor: black];
        let picture: Id = msg_send![class!(CALayer), layer];
        let _: () = msg_send![picture, setName: ns_string("wallpaper")];
        let _: () = msg_send![layer, addSublayer: picture];
        let _: () = msg_send![
            content_view,
            addSubview: view,
            positioned: NSWindowOrderingMode::Below,
            relativeTo: NIL
        ];
        observe_environment(view);
        let _: Id = msg_send![view, autorelease];
        view
    }
}

pub(crate) unsafe fn remove_view(view: Id) {
    unsafe {
        stop_observing_environment(view);
        let _: () = msg_send![view, removeFromSuperview];
    }
}

/// Shows `wallpaper` in `view` and places it.
pub(crate) unsafe fn show(
    view: Id,
    wallpaper: &Wallpaper,
    window: Id,
    follows_screen: bool,
    cover: Option<NSRect>,
) {
    unsafe {
        let Some(picture) = picture_layer(view) else {
            return;
        };
        without_animation(|| {
            let _: () = msg_send![picture, setContents: wallpaper.image];
        });
        layout(view, window, wallpaper.screen_frame, follows_screen, cover);
    }
}

/// Places the picture. Held against the screen, it moves the other way inside the window whenever
/// the window moves, so it looks still. Attached to the window, it fills the view (cropping what
/// does not fit) and resizes with it on its own, so a moving window needs no update at all; given
/// a `cover` rectangle (window coordinates, top-left origin) it fills that rectangle instead.
pub(crate) unsafe fn layout(
    view: Id,
    window: Id,
    screen_frame: NSRect,
    follows_screen: bool,
    cover: Option<NSRect>,
) {
    unsafe {
        let Some(picture) = picture_layer(view) else {
            return;
        };
        if let Some(cover) = cover.filter(|_| !follows_screen) {
            let layer: Id = msg_send![view, layer];
            let bounds: NSRect = msg_send![layer, bounds];
            let placed = NSRect::new(
                NSPoint::new(
                    cover.origin.x,
                    bounds.size.height - cover.origin.y - cover.size.height,
                ),
                cover.size,
            );
            without_animation(|| {
                let _: () = msg_send![picture, setContentsGravity: ns_string("resizeAspectFill")];
                let _: () = msg_send![picture, setAutoresizingMask: 0u32];
                let _: () = msg_send![picture, setFrame: placed];
            });
            return;
        }
        if !follows_screen {
            let layer: Id = msg_send![view, layer];
            let bounds: NSRect = msg_send![layer, bounds];
            without_animation(|| {
                let _: () = msg_send![picture, setContentsGravity: ns_string("resizeAspectFill")];
                // kCALayerWidthSizable | kCALayerHeightSizable
                let _: () = msg_send![picture, setAutoresizingMask: 2u32 | 16u32];
                let _: () = msg_send![picture, setFrame: bounds];
            });
            return;
        }
        without_animation(|| {
            let _: () = msg_send![picture, setContentsGravity: ns_string("resize")];
            let _: () = msg_send![picture, setAutoresizingMask: 0u32];
        });
        let frame: NSRect = msg_send![window, frame];
        let content: NSRect = msg_send![window, contentRectForFrameRect: frame];
        let placed = NSRect::new(
            NSPoint::new(
                screen_frame.origin.x - content.origin.x,
                screen_frame.origin.y - content.origin.y,
            ),
            screen_frame.size,
        );
        without_animation(|| {
            let _: () = msg_send![picture, setFrame: placed];
        });
    }
}

pub(crate) unsafe fn set_corner_radius(view: Id, radius: f64) {
    unsafe {
        let layer: Id = msg_send![view, layer];
        if layer.is_null() {
            return;
        }
        let _: () = msg_send![layer, setCornerRadius: radius];
        let _: () =
            msg_send![layer, setMasksToBounds: if radius > 0.0 { Bool::YES } else { Bool::NO }];
    }
}

unsafe fn picture_layer(view: Id) -> Option<Id> {
    unsafe {
        let layer: Id = msg_send![view, layer];
        if layer.is_null() {
            return None;
        }
        let sublayers: Id = msg_send![layer, sublayers];
        if sublayers.is_null() {
            return None;
        }
        let count: usize = msg_send![sublayers, count];
        (0..count)
            .map(|index| -> Id { msg_send![sublayers, objectAtIndex: index] })
            .find(|sublayer| {
                let name: Id = msg_send![*sublayer, name];
                !name.is_null() && {
                    let equal: Bool = msg_send![name, isEqualToString: ns_string("wallpaper")];
                    equal.as_bool()
                }
            })
    }
}

fn without_animation(apply: impl FnOnce()) {
    unsafe {
        let _: () = msg_send![class!(CATransaction), begin];
        let _: () = msg_send![class!(CATransaction), setDisableActions: Bool::YES];
    }
    apply();
    unsafe {
        let _: () = msg_send![class!(CATransaction), commit];
    }
}

/// How the desktop places its picture, from System Settings' Fill, Fit, Stretch and Center.
#[derive(Debug, PartialEq)]
struct Placement {
    /// `NSImageScaling`: 0 proportionally down, 1 axes independently, 2 none, 3 up or down.
    scaling: i64,
    allow_clipping: bool,
    fill: Option<(u64, u64, u64)>,
}

impl Placement {
    /// Fills the screen, cropping what does not fit: System Settings' Fill.
    fn fill() -> Self {
        Placement {
            scaling: 3,
            allow_clipping: true,
            fill: None,
        }
    }

    unsafe fn from_options(options: Id) -> Self {
        unsafe {
            let mut placement = Placement::fill();
            if options.is_null() {
                return placement;
            }
            let scaling: Id = msg_send![options, objectForKey: NSWorkspaceDesktopImageScalingKey];
            if !scaling.is_null() {
                placement.scaling = msg_send![scaling, integerValue];
            }
            let clipping: Id =
                msg_send![options, objectForKey: NSWorkspaceDesktopImageAllowClippingKey];
            if !clipping.is_null() {
                let value: Bool = msg_send![clipping, boolValue];
                placement.allow_clipping = value.as_bool();
            }
            let fill: Id = msg_send![options, objectForKey: NSWorkspaceDesktopImageFillColorKey];
            if !fill.is_null() {
                let srgb: Id = msg_send![class!(NSColorSpace), sRGBColorSpace];
                let fill: Id = msg_send![fill, colorUsingColorSpace: srgb];
                if !fill.is_null() {
                    let red: f64 = msg_send![fill, redComponent];
                    let green: f64 = msg_send![fill, greenComponent];
                    let blue: f64 = msg_send![fill, blueComponent];
                    let quantize = |value: f64| (value.clamp(0.0, 1.0) * 255.0).round() as u64;
                    placement.fill = Some((quantize(red), quantize(green), quantize(blue)));
                }
            }
            placement
        }
    }

    /// Horizontal and vertical scale of an `image` sized picture on a `canvas`.
    fn scale(&self, image: NSSize, canvas: NSSize, canvas_scale: f64) -> (f64, f64) {
        let fit_x = canvas.width / image.width;
        let fit_y = canvas.height / image.height;
        match self.scaling {
            0 => {
                let scale = fit_x.min(fit_y).min(canvas_scale);
                (scale, scale)
            }
            1 => (fit_x, fit_y),
            2 => (canvas_scale, canvas_scale),
            _ => {
                let scale = if self.allow_clipping {
                    fit_x.max(fit_y)
                } else {
                    fit_x.min(fit_y)
                };
                (scale, scale)
            }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AffineTransform {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    tx: f64,
    ty: f64,
}

// SAFETY: laid out as Core Graphics' `CGAffineTransform`, six CGFloats.
unsafe impl Encode for AffineTransform {
    const ENCODING: Encoding = Encoding::Struct("CGAffineTransform", &[f64::ENCODING; 6]);
}

/// The desktop picture laid out on a small screen-shaped canvas the way the desktop places it,
/// then blurred like the live backdrop.
unsafe fn render_blurred(
    url: Id,
    placement: &Placement,
    screen: NSSize,
    blur_radius: f64,
) -> Option<Id> {
    autoreleasepool(|_| unsafe { render_blurred_in_pool(url, placement, screen, blur_radius) })
}

unsafe fn render_blurred_in_pool(
    url: Id,
    placement: &Placement,
    screen: NSSize,
    blur_radius: f64,
) -> Option<Id> {
    unsafe {
        let yes: Id = msg_send![class!(NSNumber), numberWithBool: Bool::YES];
        let options: Id = msg_send![
            class!(NSDictionary),
            dictionaryWithObject: yes,
            forKey: kCIImageApplyOrientationProperty
        ];
        let source: Id = msg_send![class!(CIImage), imageWithContentsOfURL: url, options: options];
        if source.is_null() {
            return None;
        }
        let extent: NSRect = msg_send![source, extent];
        if !(extent.size.width.is_finite() && extent.size.height.is_finite())
            || extent.size.width < 1.0
            || extent.size.height < 1.0
        {
            return None;
        }
        let canvas_scale = canvas_scale(blur_radius);
        let canvas = NSSize::new(
            (screen.width * canvas_scale).round().max(1.0),
            (screen.height * canvas_scale).round().max(1.0),
        );
        let (scale_x, scale_y) = placement.scale(extent.size, canvas, canvas_scale);
        let transform = AffineTransform {
            a: scale_x,
            b: 0.0,
            c: 0.0,
            d: scale_y,
            tx: (canvas.width - extent.size.width * scale_x) / 2.0 - extent.origin.x * scale_x,
            ty: (canvas.height - extent.size.height * scale_y) / 2.0 - extent.origin.y * scale_y,
        };
        let placed: Id = msg_send![source, imageByApplyingTransform: transform];
        let (red, green, blue) = placement.fill.unwrap_or((0, 0, 0));
        let fill: Id = msg_send![
            class!(CIColor),
            colorWithRed: red as f64 / 255.0,
            green: green as f64 / 255.0,
            blue: blue as f64 / 255.0
        ];
        let background: Id = msg_send![class!(CIImage), imageWithColor: fill];
        let composed: Id = msg_send![placed, imageByCompositingOverImage: background];
        let canvas_rect = NSRect::new(NSPoint::new(0.0, 0.0), canvas);
        let cropped: Id = msg_send![composed, imageByCroppingToRect: canvas_rect];
        // Clamped so the blur pulls in the picture's own edge colours rather than darkening the
        // screen's edges.
        let clamped: Id = msg_send![cropped, imageByClampingToExtent];
        let blurred: Id = if blur_radius > 0.0 {
            let radius: Id =
                msg_send![class!(NSNumber), numberWithDouble: blur_radius * canvas_scale];
            let parameters: Id = msg_send![
                class!(NSDictionary),
                dictionaryWithObject: radius,
                forKey: ns_string("inputRadius")
            ];
            msg_send![
                clamped,
                imageByApplyingFilter: ns_string("CIGaussianBlur"),
                withInputParameters: parameters
            ]
        } else {
            clamped
        };
        if blurred.is_null() {
            return None;
        }
        let context = *CONTEXT.get_or_init(|| {
            let context: Id = msg_send![class!(CIContext), contextWithOptions: NIL];
            let context: Id = msg_send![context, retain];
            context as usize
        }) as Id;
        let image: *mut CGImage = msg_send![context, createCGImage: blurred, fromRect: canvas_rect];
        (!image.is_null()).then_some(image.cast::<AnyObject>())
    }
}

fn cached(key: &str) -> Option<Id> {
    CACHE
        .lock()
        .iter()
        .find(|(cached, _)| cached == key)
        .map(|(_, image)| *image as Id)
}

fn store(key: String, image: Id) {
    let mut cache = CACHE.lock();
    if cache.len() >= CACHE_LIMIT {
        let (_, oldest) = cache.remove(0);
        // Layers still showing it hold their own retain.
        unsafe { CFRelease(oldest as CFTypeRef) };
    }
    cache.push((key, image as usize));
}

unsafe fn url_path(url: Id) -> Option<String> {
    unsafe {
        if url.is_null() {
            return None;
        }
        let path: Id = msg_send![url, path];
        if path.is_null() {
            return None;
        }
        let utf8: *const std::os::raw::c_char = msg_send![path, UTF8String];
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

fn view_class() -> *const AnyClass {
    *VIEW_CLASS.get_or_init(|| {
        let mut decl = ClassBuilder::new(c"GPUIWallpaperBackdropView", class!(NSView)).unwrap();
        unsafe {
            decl.add_method(
                sel!(wallpaperEnvironmentDidChange:),
                wallpaper_environment_did_change as extern "C" fn(_, Sel, Id),
            );
        }
        decl.register() as *const AnyClass as usize
    }) as *const AnyClass
}

/// The desktop picture can change without the window moving: another Space with its own picture,
/// a display rearranged or rescaled, the system switching between light and dark, or a new picture
/// chosen in System Settings (noticed when the app comes back to the front).
unsafe fn observe_environment(view: Id) {
    unsafe {
        let workspace: Id = msg_send![class!(NSWorkspace), sharedWorkspace];
        let workspace_center: Id = msg_send![workspace, notificationCenter];
        let _: () = msg_send![
            workspace_center,
            addObserver: view,
            selector: sel!(wallpaperEnvironmentDidChange:),
            name: NSWorkspaceActiveSpaceDidChangeNotification,
            object: NIL
        ];
        // A built-in wallpaper with light and dark looks switches with the system appearance.
        let distributed: Id = msg_send![class!(NSDistributedNotificationCenter), defaultCenter];
        let _: () = msg_send![
            distributed,
            addObserver: view,
            selector: sel!(wallpaperEnvironmentDidChange:),
            name: ns_string("AppleInterfaceThemeChangedNotification"),
            object: NIL
        ];
        let center: Id = msg_send![class!(NSNotificationCenter), defaultCenter];
        for name in [
            NSApplicationDidChangeScreenParametersNotification,
            NSApplicationDidBecomeActiveNotification,
        ] {
            let _: () = msg_send![
                center,
                addObserver: view,
                selector: sel!(wallpaperEnvironmentDidChange:),
                name: name,
                object: NIL
            ];
        }
    }
}

unsafe fn stop_observing_environment(view: Id) {
    unsafe {
        let workspace: Id = msg_send![class!(NSWorkspace), sharedWorkspace];
        let workspace_center: Id = msg_send![workspace, notificationCenter];
        let _: () = msg_send![workspace_center, removeObserver: view];
        let center: Id = msg_send![class!(NSNotificationCenter), defaultCenter];
        let _: () = msg_send![center, removeObserver: view];
        let distributed: Id = msg_send![class!(NSDistributedNotificationCenter), defaultCenter];
        let _: () = msg_send![distributed, removeObserver: view];
    }
}

extern "C" fn wallpaper_environment_did_change(this: &AnyObject, _: Sel, _: Id) {
    unsafe {
        // Refreshing may swap this view out for the live blur; keep it alive until this returns.
        let this_id = this as *const AnyObject as Id;
        let _: Id = msg_send![this_id, retain];
        let _: Id = msg_send![this_id, autorelease];
        let window: Id = msg_send![this_id, window];
        if !window.is_null() {
            crate::window::refresh_window_backdrop(&*window);
        }
    }
}
