//! Ghostex: the blur behind a frosted surface's rounded rects (a tooltip bubble, toast cards).
//!
//! CDXC:Theming 2026-10-04 WHY:
//! A window's acrylic accent fills its whole rectangle: DWM ignores the window region
//! (`SetWindowRgn`) when it draws the blur, so the region the frosted surface reported clipped
//! nothing and every tooltip and toast showed a square blurred box behind its rounded card. The
//! only shape DWM gives a blur is a window's own rectangle with its rounded corners, so each
//! reported rect gets a small window of its own that holds just the blur, rounded by DWM and
//! stacked right under the surface, whose own accent turns clear. The surface keeps its region,
//! which still decides where it takes the mouse. The blur windows take no input and follow the
//! surface when it moves, shows, hides or closes.

use std::{cell::RefCell, sync::Once};

use gpui::{Bounds, Pixels, px};
use gpui_util::ResultExt as _;
use windows::{
    Win32::{
        Foundation::*,
        Graphics::{Dwm::*, Gdi::ClientToScreen},
        UI::WindowsAndMessaging::*,
    },
    core::*,
};

use crate::{get_module_handle, set_window_composition_attribute};

const BACKDROP_CLASS_NAME: PCWSTR = w!("Zed::FrostedBackdrop");

/// The blur windows of one frosted surface and the rounded rects (logical, in the surface's
/// client coordinates) they cover.
#[derive(Default)]
pub(crate) struct FrostedBackdrops {
    region: RefCell<Vec<(Bounds<Pixels>, Pixels)>>,
    windows: RefCell<Vec<HWND>>,
}

impl FrostedBackdrops {
    /// Whether the surface's blur is drawn by blur windows rather than its own accent.
    pub(crate) fn active(&self) -> bool {
        !self.windows.borrow().is_empty()
    }

    /// Blurs `region` (the last one when `None`) behind `host`, or nothing of it when `blurred`
    /// is false or the region is empty. Returns whether blur windows now draw the host's blur.
    pub(crate) fn set(
        &self,
        host: HWND,
        region: Option<Vec<(Bounds<Pixels>, Pixels)>>,
        blurred: bool,
        scale: f32,
    ) -> bool {
        if let Some(region) = region {
            *self.region.borrow_mut() = region;
        }
        let wanted = if blurred {
            self.region.borrow().len()
        } else {
            0
        };
        {
            let mut windows = self.windows.borrow_mut();
            while windows.len() > wanted {
                if let Some(window) = windows.pop() {
                    unsafe { DestroyWindow(window) }.log_err();
                }
            }
            while windows.len() < wanted {
                let Some(window) = create_backdrop_window(host) else {
                    break;
                };
                windows.push(window);
            }
            let region = self.region.borrow();
            for (window, (_, radius)) in windows.iter().zip(region.iter()) {
                set_corner_preference(*window, *radius);
            }
        }
        self.sync(host, scale);
        self.active()
    }

    /// Lays the blur windows over their rects, right under `host`, shown while it is.
    pub(crate) fn sync(&self, host: HWND, scale: f32) {
        let windows = self.windows.borrow();
        if windows.is_empty() {
            return;
        }
        let mut origin = POINT::default();
        unsafe { ClientToScreen(host, &mut origin) }.ok().log_err();
        let visibility = if unsafe { IsWindowVisible(host) }.as_bool() {
            SWP_SHOWWINDOW
        } else {
            SWP_HIDEWINDOW
        };
        let region = self.region.borrow();
        for (window, (bounds, _)) in windows.iter().zip(region.iter()) {
            let device = bounds.to_device_pixels(scale);
            unsafe {
                SetWindowPos(
                    *window,
                    Some(host),
                    origin.x + device.origin.x.0,
                    origin.y + device.origin.y.0,
                    device.size.width.0.max(1),
                    device.size.height.0.max(1),
                    SWP_NOACTIVATE | SWP_NOOWNERZORDER | visibility,
                )
            }
            .log_err();
        }
    }
}

impl FrostedBackdrops {
    /// Closes the blur windows, for a surface that is closing.
    pub(crate) fn clear(&self) {
        self.region.borrow_mut().clear();
        for window in self.windows.borrow_mut().drain(..) {
            unsafe { DestroyWindow(window) }.log_err();
        }
    }
}

impl Drop for FrostedBackdrops {
    fn drop(&mut self) {
        self.clear();
    }
}

/// A click-through, never-activated window that shows only the acrylic blur, with the host's
/// owner and topmost band so it can sit right under the host.
fn create_backdrop_window(host: HWND) -> Option<HWND> {
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| {
        let class = WNDCLASSW {
            lpfnWndProc: Some(backdrop_window_procedure),
            hInstance: get_module_handle().into(),
            lpszClassName: BACKDROP_CLASS_NAME,
            ..Default::default()
        };
        unsafe { RegisterClassW(&class) };
    });
    let host_ex_style = WINDOW_EX_STYLE(unsafe { GetWindowLongPtrW(host, GWL_EXSTYLE) } as u32);
    let ex_style = WS_EX_TOOLWINDOW
        | WS_EX_NOACTIVATE
        | WS_EX_TRANSPARENT
        | WS_EX_NOREDIRECTIONBITMAP
        | (host_ex_style & WS_EX_TOPMOST);
    let owner = unsafe { GetWindow(host, GW_OWNER) }.ok();
    let window = unsafe {
        CreateWindowExW(
            ex_style,
            BACKDROP_CLASS_NAME,
            None,
            WS_POPUP,
            0,
            0,
            1,
            1,
            owner,
            None,
            Some(get_module_handle().into()),
            None,
        )
    }
    .log_err()?;
    // The surface draws its own border; DWM's would ring the blur.
    let border = DWMWA_COLOR_NONE;
    unsafe {
        DwmSetWindowAttribute(
            window,
            DWMWA_BORDER_COLOR,
            &border as *const _ as *const _,
            std::mem::size_of_val(&border) as u32,
        )
    }
    .log_err();
    set_window_composition_attribute(window, Some((0, 0, 0, 0)), 4);
    Some(window)
}

/// DWM rounds a window to one of two fixed radii (Windows 11 only), as
/// `set_background_corner_radius` picks them.
fn set_corner_preference(window: HWND, radius: Pixels) {
    let preference = if radius <= px(0.0) {
        DWMWCP_DONOTROUND
    } else if radius < px(6.0) {
        DWMWCP_ROUNDSMALL
    } else {
        DWMWCP_ROUND
    };
    unsafe {
        DwmSetWindowAttribute(
            window,
            DWMWA_WINDOW_CORNER_PREFERENCE,
            &preference as *const _ as *const _,
            std::mem::size_of_val(&preference) as u32,
        )
    }
    .log_err();
}

unsafe extern "system" fn backdrop_window_procedure(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    match msg {
        WM_NCHITTEST => LRESULT(HTTRANSPARENT as isize),
        WM_MOUSEACTIVATE => LRESULT(MA_NOACTIVATE as isize),
        _ => unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) },
    }
}
