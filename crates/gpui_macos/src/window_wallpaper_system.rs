//! A still picture of the built-in wallpaper macOS shows on a screen, for the wallpaper backdrop.
//!
//! macOS draws its built-in wallpapers (the dynamic landscapes, the aerials, the rendered ones)
//! from wallpaper extensions, and for those `desktopImageURLForScreen:` reports the system's
//! default picture instead of what is on screen. The real choice lives in the wallpaper agent's
//! store, `~/Library/Application Support/com.apple.wallpaper/Store/Index.plist`, which records a
//! provider and its configuration per Space and display; each provider ships a thumbnail of its
//! wallpaper. Under a 60pt blur a thumbnail is indistinguishable from the full picture.
//!
//! The store and the extensions' resources are private and undocumented, so any step that finds
//! something it does not recognise gives up, and the window keeps the live blur.

use objc2::{
    class, msg_send,
    runtime::{AnyObject, Bool},
};
use std::ffi::{CStr, c_char, c_void};
use std::path::Path;
use std::ptr;

type Id = *mut AnyObject;
const NIL: Id = ptr::null_mut();

use crate::ns_string;

const STORE_PATH: &str = "Library/Application Support/com.apple.wallpaper/Store/Index.plist";
const AERIAL_THUMBNAILS: &str =
    "Library/Application Support/com.apple.wallpaper/aerials/thumbnails";
const EXTENSIONS: &str = "/System/Library/ExtensionKit/Extensions";
const DESKTOP_PICTURES: &str = "/System/Library/Desktop Pictures";
const PROVIDER_PREFIX: &str = "com.apple.wallpaper.choice.";

/// The path of a still picture of the built-in wallpaper on `screen`, or `None` when that
/// wallpaper has no still (a solid colour, a provider this does not know) or the store cannot be
/// read.
pub(crate) unsafe fn built_in_wallpaper_still(screen: Id) -> Option<String> {
    unsafe {
        let display = display_uuid(screen)?;
        let space = current_space_uuid(&display, screen);
        let home = std::env::var("HOME").ok()?;
        let store: Id = msg_send![
            class!(NSDictionary),
            dictionaryWithContentsOfFile: ns_string(&format!("{home}/{STORE_PATH}"))
        ];
        if store.is_null() {
            return None;
        }
        let choice = desktop_choice(store, space.as_deref(), &display)?;
        let provider = string_value(msg_send![choice, objectForKey: ns_string("Provider")])?;
        let configuration = configuration(choice);
        still_for_provider(&provider, configuration, &home)
    }
}

/// The store's most specific desktop choice for this Space and display: the Space's entry for the
/// display, the Space's default, the display's own entry, the one for all Spaces and displays, then
/// the system default. Entries without a desktop choice (idle-only ones) are skipped.
unsafe fn desktop_choice(store: Id, space: Option<&str>, display: &str) -> Option<Id> {
    unsafe {
        let mut entries: Vec<Id> = Vec::new();
        if let Some(space) = space {
            let spaces = object(store, "Spaces");
            let space = object(spaces, space);
            entries.push(object(object(space, "Displays"), display));
            entries.push(object(space, "Default"));
        }
        entries.push(object(object(store, "Displays"), display));
        entries.push(object(store, "AllSpacesAndDisplays"));
        entries.push(object(store, "SystemDefault"));
        entries.into_iter().find_map(|entry| {
            let choices = object(object(object(entry, "Desktop"), "Content"), "Choices");
            if choices.is_null() {
                return None;
            }
            let is_array: Bool = msg_send![choices, isKindOfClass: class!(NSArray)];
            if !is_array.as_bool() {
                return None;
            }
            let count: usize = msg_send![choices, count];
            (count > 0).then(|| msg_send![choices, objectAtIndex: 0usize])
        })
    }
}

/// A choice's configuration, stored as a nested property list; `nil` when it has none.
unsafe fn configuration(choice: Id) -> Id {
    unsafe {
        let data: Id = msg_send![choice, objectForKey: ns_string("Configuration")];
        if data.is_null() {
            return NIL;
        }
        let is_data: Bool = msg_send![data, isKindOfClass: class!(NSData)];
        let length: usize = if is_data.as_bool() {
            msg_send![data, length]
        } else {
            0
        };
        if length == 0 {
            return NIL;
        }
        msg_send![
            class!(NSPropertyListSerialization),
            propertyListWithData: data,
            options: 0usize,
            format: std::ptr::null_mut::<usize>(),
            error: std::ptr::null_mut::<Id>()
        ]
    }
}

unsafe fn still_for_provider(provider: &str, configuration: Id, home: &str) -> Option<String> {
    unsafe {
        let name = provider.strip_prefix(PROVIDER_PREFIX)?;
        if name == "aerials" {
            let asset = string_value(object(configuration, "assetID"))?;
            if asset.is_empty() || asset.contains('/') {
                return None;
            }
            return first_existing([
                format!("{home}/{AERIAL_THUMBNAILS}/{asset}.png"),
                format!(
                    "{EXTENSIONS}/WallpaperAerialsExtension.appex/Contents/Resources/{asset}.png"
                ),
            ]);
        }
        if name == "color" || name.is_empty() || !name.chars().all(char::is_alphanumeric) {
            return None;
        }
        let title = title_case(name);
        let (variant, variant_title) = if system_uses_dark_appearance() {
            ("dark", "Dark")
        } else {
            ("light", "Light")
        };
        let resources = format!("{EXTENSIONS}/Wallpaper{title}Extension.appex/Contents/Resources");
        first_existing([
            format!("{resources}/thumbnail {variant}.heic"),
            format!("{resources}/thumbnail.heic"),
            format!("{DESKTOP_PICTURES}/.thumbnails/{title} {variant_title}.heic"),
            format!("{DESKTOP_PICTURES}/.thumbnails/{title}.heic"),
            format!("{DESKTOP_PICTURES}/{title}.heic"),
        ])
    }
}

/// Built-in wallpapers with light and dark looks follow the system appearance, not the app's.
unsafe fn system_uses_dark_appearance() -> bool {
    unsafe {
        let defaults: Id = msg_send![class!(NSUserDefaults), standardUserDefaults];
        let style: Id = msg_send![defaults, stringForKey: ns_string("AppleInterfaceStyle")];
        string_value(style).is_some_and(|style| style.eq_ignore_ascii_case("dark"))
    }
}

/// The screen's display UUID, the key the store files displays under.
unsafe fn display_uuid(screen: Id) -> Option<String> {
    type CreateUuid = unsafe extern "C" fn(u32) -> *const c_void;
    unsafe {
        let description: Id = msg_send![screen, deviceDescription];
        let number: Id = msg_send![description, objectForKey: ns_string("NSScreenNumber")];
        if number.is_null() {
            return None;
        }
        let display: u32 = msg_send![number, unsignedIntValue];
        let create: CreateUuid = std::mem::transmute(symbol(c"CGDisplayCreateUUIDFromDisplayID")?);
        let uuid = create(display);
        if uuid.is_null() {
            return None;
        }
        let string = core_foundation_sys::uuid::CFUUIDCreateString(std::ptr::null(), uuid as _);
        core_foundation_sys::base::CFRelease(uuid);
        if string.is_null() {
            return None;
        }
        let value = string_value(string as Id);
        core_foundation_sys::base::CFRelease(string as _);
        value
    }
}

/// The UUID of the Space showing on the display (the first Space's is empty), from the window
/// server's private Space list. `None` when that list is unavailable.
unsafe fn current_space_uuid(display: &str, screen: Id) -> Option<String> {
    type MainConnection = unsafe extern "C" fn() -> i32;
    type CopySpaces = unsafe extern "C" fn(i32) -> Id;
    unsafe {
        let main: MainConnection = std::mem::transmute(symbol(c"CGSMainConnectionID")?);
        let copy: CopySpaces = std::mem::transmute(symbol(c"CGSCopyManagedDisplaySpaces")?);
        let displays = copy(main());
        if displays.is_null() {
            return None;
        }
        let _: Id = msg_send![displays, autorelease];
        let main_screen: Id = msg_send![class!(NSScreen), screens];
        let main_screen: Id = msg_send![main_screen, firstObject];
        let is_main = main_screen == screen;
        let count: usize = msg_send![displays, count];
        (0..count).find_map(|index| {
            let entry: Id = msg_send![displays, objectAtIndex: index];
            let identifier = string_value(object(entry, "Display Identifier"))?;
            if !(identifier.eq_ignore_ascii_case(display) || (identifier == "Main" && is_main)) {
                return None;
            }
            string_value(object(object(entry, "Current Space"), "uuid"))
        })
    }
}

unsafe fn symbol(name: &CStr) -> Option<*mut c_void> {
    let symbol = unsafe { libc::dlsym(libc::RTLD_DEFAULT, name.as_ptr()) };
    (!symbol.is_null()).then_some(symbol)
}

/// `dictionary[key]` when `dictionary` is a dictionary, `nil` otherwise.
unsafe fn object(dictionary: Id, key: &str) -> Id {
    unsafe {
        if dictionary.is_null() {
            return NIL;
        }
        let is_dictionary: Bool = msg_send![dictionary, isKindOfClass: class!(NSDictionary)];
        if !is_dictionary.as_bool() {
            return NIL;
        }
        msg_send![dictionary, objectForKey: ns_string(key)]
    }
}

unsafe fn string_value(value: Id) -> Option<String> {
    unsafe {
        if value.is_null() {
            return None;
        }
        let is_string: Bool = msg_send![value, isKindOfClass: class!(NSString)];
        if !is_string.as_bool() {
            return None;
        }
        let utf8: *const c_char = msg_send![value, UTF8String];
        (!utf8.is_null()).then(|| CStr::from_ptr(utf8).to_string_lossy().into_owned())
    }
}

fn title_case(name: &str) -> String {
    let mut chars = name.chars();
    chars
        .next()
        .map(|first| first.to_uppercase().chain(chars).collect())
        .unwrap_or_default()
}

fn first_existing(paths: impl IntoIterator<Item = String>) -> Option<String> {
    paths.into_iter().find(|path| Path::new(path).is_file())
}
