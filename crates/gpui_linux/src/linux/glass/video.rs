//! Muted, looping video decoded by the distribution's FFmpeg, with a bounded one-frame pipe.
//! SIGSTOP pauses decoding too: a paused backdrop must not burn CPU in a hidden decoder.
use super::*;
use std::{
    io::Read,
    os::fd::AsRawFd,
    process::{Child, ChildStdout, Command, Stdio},
};

pub(super) struct Video {
    child: Child,
    stdout: ChildStdout,
    width: u32,
    height: u32,
    bytes: Vec<u8>,
    paused: bool,
    first_frame: bool,
    started: Instant,
}

impl Video {
    pub fn open(path: &std::path::Path, cover_width: f32, blur_radius: f32) -> Option<Self> {
        let path_text = path.to_str()?;
        let metadata = system::command(
            "ffprobe",
            &[
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height:stream_side_data=rotation",
                "-of",
                "json",
                path_text,
            ],
        )?;
        let metadata: serde_json::Value = serde_json::from_str(&metadata).ok()?;
        let stream = &metadata["streams"][0];
        let mut width = stream["width"].as_u64()? as f32;
        let mut height = stream["height"].as_u64()? as f32;
        let rotated = stream["side_data_list"].as_array().is_some_and(|rows| {
            rows.iter().any(|row| {
                row["rotation"]
                    .as_i64()
                    .is_some_and(|r| r.abs() % 180 == 90)
            })
        });
        if rotated {
            std::mem::swap(&mut width, &mut height);
        }
        if width < 1.0 || height < 1.0 {
            return None;
        }
        let scale = 512.0 / width.max(height);
        let width = (width * scale).round().max(1.0) as u32;
        let height = (height * scale).round().max(1.0) as u32;
        let sigma = (blur_radius * width as f32 / cover_width.max(1.0)).min(64.0);
        // A radius of 0 plays the frames sharp.
        let blur = if sigma > 0.0 {
            format!("gblur=sigma={sigma}:steps=3,")
        } else {
            String::new()
        };
        let filter = format!("fps=24,scale={width}:{height},{blur}format=rgba");
        let mut child = Command::new("ffmpeg")
            .args([
                "-nostdin",
                "-v",
                "error",
                "-threads",
                "1",
                "-filter_threads",
                "1",
                "-stream_loop",
                "-1",
                "-i",
            ])
            .arg(path)
            .args([
                "-an", "-sn", "-dn", "-vf", &filter, "-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;
        let stdout = child.stdout.take()?;
        unsafe {
            libc::fcntl(stdout.as_raw_fd(), libc::F_SETFL, libc::O_NONBLOCK);
        }
        Some(Self {
            child,
            stdout,
            width,
            height,
            bytes: Vec::new(),
            paused: false,
            first_frame: true,
            started: Instant::now(),
        })
    }

    pub fn set_playing(&mut self, playing: bool) {
        if self.paused == playing {
            // This PID belongs to the Child we retain; it cannot be reused until wait().
            unsafe {
                libc::kill(
                    self.child.id() as i32,
                    if playing {
                        libc::SIGCONT
                    } else {
                        libc::SIGSTOP
                    },
                );
            }
            self.paused = !playing;
        }
    }

    pub fn loading(&self) -> bool {
        self.first_frame && self.started.elapsed() < Duration::from_secs(5)
    }

    pub fn frame(&mut self) -> Option<GlassImage> {
        if self.paused {
            return None;
        }
        let frame_len = (self.width * self.height * 4) as usize;
        let mut buffer = [0; 16384];
        while self.bytes.len() < frame_len {
            let available = (frame_len - self.bytes.len()).min(buffer.len());
            match self.stdout.read(&mut buffer[..available]) {
                Ok(0) => return None,
                Ok(n) => self.bytes.extend_from_slice(&buffer[..n]),
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => return None,
                Err(_) => return None,
            }
        }
        self.first_frame = false;
        Some(GlassImage {
            width: self.width,
            height: self.height,
            rgba: std::mem::take(&mut self.bytes),
        })
    }
}

impl Drop for Video {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
