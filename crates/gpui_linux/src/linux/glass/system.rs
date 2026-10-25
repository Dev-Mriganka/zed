//! Desktop discovery runs on the backdrop worker, never on the UI/render thread.
use super::*;
use std::{
    io::{Read, Write},
    os::unix::net::UnixStream,
    process::{Command, Stdio},
};

#[derive(Default)]
pub(super) struct System {
    pub wallpaper: Option<PathBuf>,
    pub desktop_cover: Option<Bounds<Pixels>>,
    pub reduce_motion: bool,
    on_battery: bool,
    low_power: bool,
    display_asleep: bool,
    last_settings: Option<Instant>,
    last_position: Option<Instant>,
    appearance: Option<bool>,
    monitor: Option<String>,
}

impl System {
    pub fn may_animate(&self, only_on_power: bool) -> bool {
        !self.reduce_motion
            && !self.low_power
            && !self.display_asleep
            && !(only_on_power && self.on_battery)
    }

    pub fn refresh(&mut self, request: &Request) {
        let now = Instant::now();
        if self
            .last_settings
            .is_none_or(|time| now.duration_since(time) > Duration::from_secs(2))
            || self.appearance != Some(request.light)
            || self.monitor != request.monitor
        {
            self.last_settings = Some(now);
            self.appearance = Some(request.light);
            self.monitor = request.monitor.clone();
            self.on_battery = on_battery();
            self.low_power =
                command("powerprofilesctl", &["get"]).is_some_and(|v| v.trim() == "power-saver");
            self.reduce_motion = command(
                "gsettings",
                &["get", "org.gnome.desktop.interface", "enable-animations"],
            )
            .is_some_and(|v| v.trim() == "false");
            if let Some(value) = hypr("j/getoption animations:enabled") {
                self.reduce_motion = value["bool"]
                    .as_bool()
                    .map(|enabled| !enabled)
                    .or_else(|| value["int"].as_i64().map(|enabled| enabled == 0))
                    .unwrap_or(self.reduce_motion);
            }
            if std::env::var("XDG_CURRENT_DESKTOP")
                .is_ok_and(|desktop| desktop.to_lowercase().contains("kde"))
            {
                self.reduce_motion |= command(
                    "kreadconfig6",
                    &[
                        "--file",
                        "kdeglobals",
                        "--group",
                        "KDE",
                        "--key",
                        "AnimationDurationFactor",
                    ],
                )
                .and_then(|value| value.trim().parse::<f32>().ok())
                    == Some(0.0);
            }
            self.display_asleep = display_asleep();
            let hypr_window = hypr_cover(&request.title);
            if let Some((_, asleep, _)) = &hypr_window {
                self.display_asleep = *asleep;
            }
            if matches!(request.selection, Selection::Image(None)) {
                let monitor = request.monitor.as_deref().or_else(|| {
                    hypr_window
                        .as_ref()
                        .and_then(|(_, _, name)| name.as_deref())
                });
                self.wallpaper = wallpaper(request.light, monitor);
            }
        }
        if request.follows_screen
            && request.active
            && self
                .last_position
                .is_none_or(|time| now.duration_since(time) > Duration::from_millis(40))
        {
            self.last_position = Some(now);
            self.desktop_cover = hypr_cover(&request.title).map(|(bounds, asleep, _)| {
                self.display_asleep = asleep;
                bounds
            });
        }
    }
}

fn on_battery() -> bool {
    let Ok(supplies) = std::fs::read_dir("/sys/class/power_supply") else {
        return false;
    };
    let mut battery = false;
    let mut plugged_in = false;
    for supply in supplies.flatten() {
        let path = supply.path();
        let kind = std::fs::read_to_string(path.join("type")).unwrap_or_default();
        if kind.trim() == "Battery" {
            battery |= std::fs::read_to_string(path.join("status"))
                .is_ok_and(|s| s.trim() == "Discharging");
        } else {
            plugged_in |=
                std::fs::read_to_string(path.join("online")).is_ok_and(|s| s.trim() == "1");
        }
    }
    battery && !plugged_in
}

/// Hyprland is the one supported Wayland backend that exposes global client positions.
/// Match our PID and exact title rather than reading the currently focused foreign window.
fn hypr_cover(title: &str) -> Option<(Bounds<Pixels>, bool, Option<String>)> {
    let clients = hypr("j/clients")?;
    let mut matches = clients.as_array()?.iter().filter(|client| {
        client["pid"].as_u64() == Some(std::process::id() as u64)
            && client["title"].as_str() == Some(title)
    });
    let client = matches.next()?;
    if matches.next().is_some() {
        return None;
    }
    let monitors = hypr("j/monitors")?;
    let monitor = monitors
        .as_array()?
        .iter()
        .find(|monitor| monitor["id"] == client["monitor"])?;
    let number = |value: &serde_json::Value| value.as_f64().map(|v| v as f32);
    let scale = number(&monitor["scale"])?;
    let mut width = number(&monitor["width"])?;
    let mut height = number(&monitor["height"])?;
    if monitor["transform"].as_u64().unwrap_or(0) % 2 == 1 {
        std::mem::swap(&mut width, &mut height);
    }
    Some((
        Bounds::new(
            Point::new(
                px(number(&monitor["x"])? - number(&client["at"][0])?),
                px(number(&monitor["y"])? - number(&client["at"][1])?),
            ),
            Size::new(px(width / scale), px(height / scale)),
        ),
        monitor["dpmsStatus"].as_bool() == Some(false),
        monitor["name"].as_str().map(str::to_owned),
    ))
}

fn hypr(request: &str) -> Option<serde_json::Value> {
    let signature = std::env::var("HYPRLAND_INSTANCE_SIGNATURE").ok()?;
    let runtime = std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(format!("/run/user/{}", unsafe { libc::getuid() })));
    let mut socket =
        UnixStream::connect(runtime.join("hypr").join(signature).join(".socket.sock")).ok()?;
    socket
        .set_read_timeout(Some(Duration::from_millis(200)))
        .ok()?;
    socket
        .set_write_timeout(Some(Duration::from_millis(200)))
        .ok()?;
    socket.write_all(request.as_bytes()).ok()?;
    let mut bytes = Vec::new();
    socket.take(4 * 1024 * 1024).read_to_end(&mut bytes).ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn wallpaper(light: bool, monitor: Option<&str>) -> Option<PathBuf> {
    if std::env::var_os("HYPRLAND_INSTANCE_SIGNATURE").is_some() {
        if let Some(active) = command("hyprctl", &["hyprpaper", "listactive"]) {
            for line in active.lines() {
                if let Some((output, path)) = line.split_once(" = ") {
                    if monitor.is_none_or(|monitor| output.trim() == monitor) {
                        return file_path(path.trim());
                    }
                }
            }
        }
        for program in ["swww", "awww"] {
            if let Some(active) = command(program, &["query"]) {
                for line in active.lines() {
                    if monitor.is_none_or(|monitor| line.starts_with(&format!("{monitor}:"))) {
                        if let Some((_, path)) = line.split_once("image: ") {
                            return file_path(path.trim());
                        }
                    }
                }
            }
        }
        return None;
    }
    let desktop = std::env::var("XDG_CURRENT_DESKTOP")
        .unwrap_or_default()
        .to_lowercase();
    if desktop.contains("gnome")
        || desktop.contains("cinnamon")
        || desktop.contains("mate")
        || desktop.contains("unity")
    {
        let schema = if desktop.contains("cinnamon") {
            "org.cinnamon.desktop.background"
        } else if desktop.contains("mate") {
            "org.mate.background"
        } else {
            "org.gnome.desktop.background"
        };
        let key = if desktop.contains("mate") {
            "picture-filename"
        } else if light || desktop.contains("cinnamon") {
            "picture-uri"
        } else {
            "picture-uri-dark"
        };
        return command("gsettings", &["get", schema, key])
            .and_then(|value| file_path(value.trim().trim_matches('\'')));
    }
    if desktop.contains("kde") {
        // Ask Plasma for the active image on the matching output; this is read-only scripting.
        let script = "var r=[]; for (var d of desktops()) { d.currentConfigGroup=['Wallpaper','org.kde.image','General']; r.push({screen:d.screen,image:d.readConfig('Image','')}); } print(JSON.stringify(r));";
        let reply = command(
            "gdbus",
            &[
                "call",
                "--session",
                "--dest",
                "org.kde.plasmashell",
                "--object-path",
                "/PlasmaShell",
                "--method",
                "org.kde.PlasmaShell.evaluateScript",
                script,
            ],
        )?;
        let start = reply.find('[')?;
        let end = reply.rfind(']')?;
        let rows: serde_json::Value = serde_json::from_str(&reply[start..=end]).ok()?;
        // Without an output mapping only an unambiguous single desktop is usable.
        let rows = rows.as_array()?;
        if rows.len() == 1 {
            return rows[0]["image"].as_str().and_then(file_path);
        }
    }
    None
}

fn file_path(value: &str) -> Option<PathBuf> {
    let path = if value.starts_with("file:") {
        url::Url::parse(value).ok()?.to_file_path().ok()?
    } else {
        PathBuf::from(value)
    };
    path.is_file().then_some(path)
}

/// Bounded reads and a deadline keep an absent/hung desktop service from holding a worker alive.
pub(super) fn command(program: &str, args: &[&str]) -> Option<String> {
    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let stdout = child.stdout.take()?;
    use std::os::fd::AsRawFd;
    unsafe {
        libc::fcntl(stdout.as_raw_fd(), libc::F_SETFL, libc::O_NONBLOCK);
    }
    let mut stdout = stdout;
    let mut bytes = Vec::new();
    let deadline = Instant::now() + Duration::from_millis(800);
    loop {
        let mut buffer = [0; 8192];
        match stdout.read(&mut buffer) {
            Ok(n) if n > 0 => {
                bytes.extend_from_slice(&buffer[..n]);
                if bytes.len() > 4 * 1024 * 1024 {
                    break;
                }
            }
            Err(error) if error.kind() != std::io::ErrorKind::WouldBlock => break,
            _ => {}
        }
        if let Ok(Some(status)) = child.try_wait() {
            let _ = stdout.read_to_end(&mut bytes);
            return status
                .success()
                .then(|| String::from_utf8(bytes).ok())
                .flatten();
        }
        if Instant::now() >= deadline {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    let _ = child.kill();
    let _ = child.wait();
    None
}

#[cfg(feature = "x11")]
fn display_asleep() -> bool {
    use x11rb::protocol::dpms::{ConnectionExt, DPMSMode};
    let asleep = || -> Option<bool> {
        let (connection, _) = x11rb::connect(None).ok()?;
        let reply = connection.dpms_info().ok()?.reply().ok()?;
        Some(reply.state && reply.power_level != DPMSMode::ON)
    };
    asleep().unwrap_or(false)
}

#[cfg(not(feature = "x11"))]
fn display_asleep() -> bool {
    false
}
