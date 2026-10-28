//! Provides a [calloop] event source from [XDG Desktop Portal] events
//!
//! This module uses the [ashpd] crate

use ashpd::desktop::settings::{ColorScheme, Settings};
use calloop::channel::{Channel, Sender};
use calloop::{EventSource, Poll, PostAction, Readiness, Token, TokenFactory};
use futures::future::BoxFuture;
use smol::stream::StreamExt;

use gpui::{BackgroundExecutor, Task, WindowAppearance};

pub enum Event {
    WindowAppearance(WindowAppearance),
    #[cfg_attr(feature = "x11", allow(dead_code))]
    CursorTheme(String),
    #[cfg_attr(feature = "x11", allow(dead_code))]
    CursorSize(u32),
    ButtonLayout(String),
}

/// Forwards desktop settings from the portal to the event loop.
///
/// Owns the task that listens to the portal, so removing the source from the loop cancels it and
/// releases its D-Bus subscriptions.
pub struct XDPEventSource {
    channel: Channel<Event>,
    _listener: Task<()>,
}

impl XDPEventSource {
    pub fn new(executor: &BackgroundExecutor) -> Self {
        let (sender, channel) = calloop::channel::channel();
        let listener = executor.spawn(async move {
            if let Err(error) = listen(sender).await {
                log::debug!("stopped listening to desktop portal settings: {error:#}");
            }
        });
        Self {
            channel,
            _listener: listener,
        }
    }
}

async fn listen(sender: Sender<Event>) -> anyhow::Result<()> {
    let settings = Settings::new().await?;

    if let Ok(initial_appearance) = settings.color_scheme().await {
        sender.send(Event::WindowAppearance(
            window_appearance_from_color_scheme(initial_appearance),
        ))?;
    }
    if let Ok(initial_theme) = settings
        .read::<String>("org.gnome.desktop.interface", "cursor-theme")
        .await
    {
        sender.send(Event::CursorTheme(initial_theme))?;
    }

    // If u32 is used here, it throws invalid type error
    if let Ok(initial_size) = settings
        .read::<i32>("org.gnome.desktop.interface", "cursor-size")
        .await
    {
        sender.send(Event::CursorSize(initial_size as u32))?;
    }

    if let Ok(initial_layout) = settings
        .read::<String>("org.gnome.desktop.wm.preferences", "button-layout")
        .await
    {
        sender.send(Event::ButtonLayout(initial_layout))?;
    }

    // Each listener runs until its stream or the channel closes. They're joined rather than
    // spawned, so that cancelling this future cancels them all.
    let mut listeners: Vec<BoxFuture<'_, anyhow::Result<()>>> = Vec::new();

    if let Ok(mut cursor_theme_changed) = settings
        .receive_setting_changed_with_args("org.gnome.desktop.interface", "cursor-theme")
        .await
    {
        let sender = sender.clone();
        listeners.push(Box::pin(async move {
            while let Some(theme) = cursor_theme_changed.next().await {
                sender.send(Event::CursorTheme(theme?))?;
            }
            anyhow::Ok(())
        }));
    }

    if let Ok(mut cursor_size_changed) = settings
        .receive_setting_changed_with_args::<i32>("org.gnome.desktop.interface", "cursor-size")
        .await
    {
        let sender = sender.clone();
        listeners.push(Box::pin(async move {
            while let Some(size) = cursor_size_changed.next().await {
                sender.send(Event::CursorSize(size? as u32))?;
            }
            anyhow::Ok(())
        }));
    }

    if let Ok(mut button_layout_changed) = settings
        .receive_setting_changed_with_args("org.gnome.desktop.wm.preferences", "button-layout")
        .await
    {
        let sender = sender.clone();
        listeners.push(Box::pin(async move {
            while let Some(layout) = button_layout_changed.next().await {
                sender.send(Event::ButtonLayout(layout?))?;
            }
            anyhow::Ok(())
        }));
    }

    let mut appearance_changed = settings.receive_color_scheme_changed().await?;
    listeners.push(Box::pin(async move {
        while let Some(scheme) = appearance_changed.next().await {
            sender.send(Event::WindowAppearance(
                window_appearance_from_color_scheme(scheme),
            ))?;
        }
        anyhow::Ok(())
    }));

    for result in futures::future::join_all(listeners).await {
        result?;
    }
    Ok(())
}

impl EventSource for XDPEventSource {
    type Event = Event;
    type Metadata = ();
    type Ret = ();
    type Error = anyhow::Error;

    fn process_events<F>(
        &mut self,
        readiness: Readiness,
        token: Token,
        mut callback: F,
    ) -> Result<PostAction, Self::Error>
    where
        F: FnMut(Self::Event, &mut Self::Metadata) -> Self::Ret,
    {
        self.channel.process_events(readiness, token, |evt, _| {
            if let calloop::channel::Event::Msg(msg) = evt {
                (callback)(msg, &mut ())
            }
        })?;

        Ok(PostAction::Continue)
    }

    fn register(
        &mut self,
        poll: &mut Poll,
        token_factory: &mut TokenFactory,
    ) -> calloop::Result<()> {
        self.channel.register(poll, token_factory)?;

        Ok(())
    }

    fn reregister(
        &mut self,
        poll: &mut Poll,
        token_factory: &mut TokenFactory,
    ) -> calloop::Result<()> {
        self.channel.reregister(poll, token_factory)?;

        Ok(())
    }

    fn unregister(&mut self, poll: &mut Poll) -> calloop::Result<()> {
        self.channel.unregister(poll)?;

        Ok(())
    }
}

/// The desktop's colour scheme read once, waiting at most `timeout`, for the first frame.
///
/// CDXC:PlatformSupport 2026-10-28 WHY: the platform started out Light and learned the real scheme
/// only when the portal listener above answered, about two seconds into the launch on Omarchy, so a
/// dark desktop saw every window drawn light first and then switch. Reading it before the first
/// window opens costs one D-Bus round trip; the listener still follows later changes, and a portal
/// that does not answer in time leaves Light as before.
pub fn initial_window_appearance(timeout: std::time::Duration) -> Option<WindowAppearance> {
    let read = async {
        let settings = Settings::new().await.ok()?;
        settings
            .color_scheme()
            .await
            .ok()
            .map(window_appearance_from_color_scheme)
    };
    let deadline = async {
        smol::Timer::after(timeout).await;
        None
    };
    smol::block_on(smol::future::or(read, deadline))
}

fn window_appearance_from_color_scheme(cs: ColorScheme) -> WindowAppearance {
    match cs {
        ColorScheme::PreferDark => WindowAppearance::Dark,
        ColorScheme::PreferLight => WindowAppearance::Light,
        ColorScheme::NoPreference => WindowAppearance::Light,
    }
}
