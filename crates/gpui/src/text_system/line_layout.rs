use crate::{FontId, GlyphId, Pixels, PlatformTextSystem, Point, SharedString, Size, point, px};
use collections::FxHashMap;
use smallvec::SmallVec;
use std::{
    borrow::Borrow,
    cell::RefCell,
    hash::{Hash, Hasher},
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use super::LineWrapper;

/// A laid out and styled line of text
#[derive(Default, Debug)]
pub struct LineLayout {
    /// The font size for this line
    pub font_size: Pixels,
    /// The width of the line
    pub width: Pixels,
    /// The ascent of the line
    pub ascent: Pixels,
    /// The descent of the line
    pub descent: Pixels,
    /// The shaped runs that make up this line
    pub runs: Vec<ShapedRun>,
    /// The length of the line in utf-8 bytes
    pub len: usize,
}

/// A run of text that has been shaped .
#[derive(Debug, Clone)]
pub struct ShapedRun {
    /// The font id for this run
    pub font_id: FontId,
    /// The glyphs that make up this run
    pub glyphs: Vec<ShapedGlyph>,
}

/// A single glyph, ready to paint.
#[derive(Clone, Debug)]
pub struct ShapedGlyph {
    /// The ID for this glyph, as determined by the text system.
    pub id: GlyphId,

    /// The position of this glyph in its containing line.
    pub position: Point<Pixels>,

    /// The index of this glyph in the original text.
    pub index: usize,

    /// Whether this glyph is an emoji
    pub is_emoji: bool,
}

impl LineLayout {
    /// The index for the character at the given x coordinate
    pub fn index_for_x(&self, x: Pixels) -> Option<usize> {
        if x >= self.width {
            None
        } else {
            for run in self.runs.iter().rev() {
                for glyph in run.glyphs.iter().rev() {
                    if glyph.position.x <= x {
                        return Some(glyph.index);
                    }
                }
            }
            Some(0)
        }
    }

    /// closest_index_for_x returns the character boundary closest to the given x coordinate
    /// (e.g. to handle aligning up/down arrow keys)
    pub fn closest_index_for_x(&self, x: Pixels) -> usize {
        let mut prev_index = 0;
        let mut prev_x = px(0.);

        for run in self.runs.iter() {
            for glyph in run.glyphs.iter() {
                if glyph.position.x >= x {
                    if glyph.position.x - x < x - prev_x {
                        return glyph.index;
                    } else {
                        return prev_index;
                    }
                }
                prev_index = glyph.index;
                prev_x = glyph.position.x;
            }
        }

        if self.len == 1 {
            if x > self.width / 2. {
                return 1;
            } else {
                return 0;
            }
        }

        self.len
    }

    /// The x position of the character at the given index
    pub fn x_for_index(&self, index: usize) -> Pixels {
        for run in &self.runs {
            for glyph in &run.glyphs {
                if glyph.index >= index {
                    return glyph.position.x;
                }
            }
        }
        self.width
    }

    /// The corresponding Font at the given index
    pub fn font_id_for_index(&self, index: usize) -> Option<FontId> {
        for run in &self.runs {
            for glyph in &run.glyphs {
                if glyph.index >= index {
                    return Some(run.font_id);
                }
            }
        }
        None
    }

    /// Split this layout at a byte index, returning `(prefix, suffix)`.
    ///
    /// - `prefix` contains glyphs for bytes `[0, byte_index)` with original positions.
    ///   Its width equals the x-advance up to the split point.
    /// - `suffix` contains glyphs for bytes `[byte_index, len)` with positions
    ///   shifted left so the first glyph starts at x=0, and byte indices rebased to 0.
    /// - `font_size`, `ascent`, and `descent` are copied to both halves.
    pub fn split_at(&self, byte_index: usize) -> (LineLayout, LineLayout) {
        let x_offset = self.x_for_index(byte_index);

        // Partition glyph runs. A single run may contribute glyphs to both halves.
        let mut left_runs = Vec::new();
        let mut right_runs = Vec::new();

        for run in &self.runs {
            let split_pos = run.glyphs.partition_point(|g| g.index < byte_index);

            if split_pos > 0 {
                left_runs.push(ShapedRun {
                    font_id: run.font_id,
                    glyphs: run.glyphs[..split_pos].to_vec(),
                });
            }

            if split_pos < run.glyphs.len() {
                let right_glyphs = run.glyphs[split_pos..]
                    .iter()
                    .map(|g| ShapedGlyph {
                        id: g.id,
                        position: point(g.position.x - x_offset, g.position.y),
                        index: g.index - byte_index,
                        is_emoji: g.is_emoji,
                    })
                    .collect();
                right_runs.push(ShapedRun {
                    font_id: run.font_id,
                    glyphs: right_glyphs,
                });
            }
        }

        let left = LineLayout {
            font_size: self.font_size,
            width: x_offset,
            ascent: self.ascent,
            descent: self.descent,
            runs: left_runs,
            len: byte_index,
        };

        let right = LineLayout {
            font_size: self.font_size,
            width: self.width - x_offset,
            ascent: self.ascent,
            descent: self.descent,
            runs: right_runs,
            len: self.len - byte_index,
        };

        (left, right)
    }

    pub(crate) fn compute_wrap_boundaries(
        &self,
        text: &str,
        wrap_width: Pixels,
        max_lines: Option<usize>,
    ) -> SmallVec<[WrapBoundary; 1]> {
        let mut boundaries = SmallVec::new();
        let mut first_non_whitespace_ix = None;
        let mut last_candidate_ix = None;
        let mut last_candidate_x = px(0.);
        let mut last_boundary = WrapBoundary {
            run_ix: 0,
            glyph_ix: 0,
        };
        let mut last_boundary_x = px(0.);
        let mut prev_ch = '\0';
        let mut glyphs = self
            .runs
            .iter()
            .enumerate()
            .flat_map(move |(run_ix, run)| {
                run.glyphs.iter().enumerate().map(move |(glyph_ix, glyph)| {
                    let character = text[glyph.index..].chars().next().unwrap();
                    (
                        WrapBoundary { run_ix, glyph_ix },
                        character,
                        glyph.position.x,
                    )
                })
            })
            .peekable();

        while let Some((boundary, ch, x)) = glyphs.next() {
            if ch == '\n' {
                continue;
            }

            // Here is very similar to `LineWrapper::wrap_line` to determine text wrapping,
            // but there are some differences, so we have to duplicate the code here.
            if LineWrapper::is_word_char(ch) {
                if prev_ch == ' ' && ch != ' ' && first_non_whitespace_ix.is_some() {
                    last_candidate_ix = Some(boundary);
                    last_candidate_x = x;
                }
            } else {
                // Not between CJK punctuation and the character it belongs
                // to, see `fast::line_break`.
                if ch != ' '
                    && first_non_whitespace_ix.is_some()
                    && crate::fast::line_break::allows_break_between(prev_ch, ch)
                {
                    last_candidate_ix = Some(boundary);
                    last_candidate_x = x;
                }
            }

            if ch != ' ' && first_non_whitespace_ix.is_none() {
                first_non_whitespace_ix = Some(boundary);
            }

            let next_x = glyphs.peek().map_or(self.width, |(_, _, x)| *x);
            let width = next_x - last_boundary_x;

            if width > wrap_width && boundary > last_boundary {
                // When used line_clamp, we should limit the number of lines.
                if let Some(max_lines) = max_lines
                    && boundaries.len() >= max_lines.saturating_sub(1)
                {
                    break;
                }

                if let Some(last_candidate_ix) = last_candidate_ix.take() {
                    last_boundary = last_candidate_ix;
                    last_boundary_x = last_candidate_x;
                } else {
                    last_boundary = boundary;
                    last_boundary_x = x;
                }
                boundaries.push(last_boundary);
            }
            prev_ch = ch;
        }

        boundaries
    }

    /// Each glyph cluster's first byte, left edge and drawn width, in byte order.
    ///
    /// Glyph order differs between text systems for right-to-left runs (byte order
    /// with falling x, or visual order), so this goes by position, not by order.
    fn clusters(&self) -> Vec<(usize, Pixels, Pixels)> {
        let mut lefts: Vec<(usize, Pixels)> = Vec::new();
        for glyph in self.runs.iter().flat_map(|run| run.glyphs.iter()) {
            match lefts.iter_mut().find(|(index, _)| *index == glyph.index) {
                Some((_, left)) => *left = (*left).min(glyph.position.x),
                None => lefts.push((glyph.index, glyph.position.x)),
            }
        }
        lefts.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut clusters: Vec<(usize, Pixels, Pixels)> = lefts
            .iter()
            .enumerate()
            .map(|(ix, (index, left))| {
                let right = lefts.get(ix + 1).map_or(self.width, |(_, next)| *next);
                (*index, *left, (right - *left).max(px(0.)))
            })
            .collect();
        clusters.sort_by_key(|(index, _, _)| *index);
        clusters
    }

    /// The byte ranges of the rows `text` wraps into, broken in byte (reading)
    /// order with the same rules as [`Self::compute_wrap_boundaries`].
    ///
    /// That method slices the glyphs in the order they are stored, which for
    /// right-to-left text does not follow the text: rows came out holding the
    /// paragraph's end, or never wrapped. A browser breaks a paragraph's lines in
    /// reading order and then lays each line out on its own, which is what this is
    /// for.
    fn logical_wrap_ranges(
        &self,
        text: &str,
        clusters: &[(usize, Pixels, Pixels)],
        wrap_width: Pixels,
        max_lines: Option<usize>,
    ) -> SmallVec<[Range<usize>; 2]> {
        let mut rows = SmallVec::new();
        let mut row_start = 0;
        let mut row_width = px(0.);
        let mut candidate: Option<(usize, Pixels)> = None;
        let mut seen_non_space = false;
        let mut prev_ch = '\0';
        for &(index, _, width) in clusters {
            let Some(ch) = text.get(index..).and_then(|rest| rest.chars().next()) else {
                continue;
            };
            if ch == '\n' {
                continue;
            }

            if index > row_start && seen_non_space {
                if LineWrapper::is_word_char(ch) {
                    if prev_ch == ' ' && ch != ' ' {
                        candidate = Some((index, row_width));
                    }
                } else if ch != ' ' {
                    candidate = Some((index, row_width));
                }
            }
            if ch != ' ' {
                seen_non_space = true;
            }

            if row_width + width > wrap_width && index > row_start {
                if max_lines.is_some_and(|max_lines| rows.len() + 1 >= max_lines) {
                    break;
                }
                let (break_ix, width_before) = candidate.take().unwrap_or((index, row_width));
                rows.push(row_start..break_ix);
                row_start = break_ix;
                row_width -= width_before;
            }
            row_width += width;
            prev_ch = ch;
        }
        rows.push(row_start..text.len());
        rows
    }
}

/// Whether `text` holds a character that can start a right-to-left run, whose
/// glyphs are drawn against byte order.
fn has_right_to_left(text: &str) -> bool {
    !text.is_ascii()
        && text.chars().any(|character| {
            matches!(
                unicode_bidi::bidi_class(character),
                unicode_bidi::BidiClass::R
                    | unicode_bidi::BidiClass::AL
                    | unicode_bidi::BidiClass::AN
                    | unicode_bidi::BidiClass::RLE
                    | unicode_bidi::BidiClass::RLO
                    | unicode_bidi::BidiClass::RLI
            )
        })
}

/// The font runs covering `range` of the text, rebased to start at its start.
fn font_runs_in_range(runs: &[FontRun], range: Range<usize>) -> SmallVec<[FontRun; 2]> {
    let mut clipped = SmallVec::new();
    let mut offset = 0;
    for run in runs {
        let run_start = offset;
        offset += run.len;
        let start = run_start.max(range.start);
        let end = offset.min(range.end);
        if start < end {
            clipped.push(FontRun {
                len: end - start,
                font_id: run.font_id,
            });
        }
    }
    clipped
}

/// A line of text that has been wrapped to fit a given width
#[derive(Default, Debug)]
pub struct WrappedLineLayout {
    /// The line layout, pre-wrapping.
    pub unwrapped_layout: Arc<LineLayout>,

    /// The boundaries at which the line was wrapped
    pub wrap_boundaries: SmallVec<[WrapBoundary; 1]>,

    /// The width of the line, if it was wrapped
    pub wrap_width: Option<Pixels>,
}

/// A boundary at which a line was wrapped
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct WrapBoundary {
    /// The index in the run just before the line was wrapped
    pub run_ix: usize,
    /// The index of the glyph just before the line was wrapped
    pub glyph_ix: usize,
}

impl WrappedLineLayout {
    /// The length of the underlying text, in utf8 bytes.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.unwrapped_layout.len
    }

    /// The width of this line, in pixels, whether or not it was wrapped.
    pub fn width(&self) -> Pixels {
        self.wrap_width
            .unwrap_or(Pixels::MAX)
            .min(self.unwrapped_layout.width)
    }

    /// The size of the whole wrapped text, for the given line_height.
    /// can span multiple lines if there are multiple wrap boundaries.
    pub fn size(&self, line_height: Pixels) -> Size<Pixels> {
        Size {
            width: self.width(),
            height: line_height * (self.wrap_boundaries.len() + 1),
        }
    }

    /// The ascent of a line in this layout
    pub fn ascent(&self) -> Pixels {
        self.unwrapped_layout.ascent
    }

    /// The descent of a line in this layout
    pub fn descent(&self) -> Pixels {
        self.unwrapped_layout.descent
    }

    /// The wrap boundaries in this layout
    pub fn wrap_boundaries(&self) -> &[WrapBoundary] {
        &self.wrap_boundaries
    }

    /// The font size of this layout
    pub fn font_size(&self) -> Pixels {
        self.unwrapped_layout.font_size
    }

    /// The runs in this layout, sans wrapping
    pub fn runs(&self) -> &[ShapedRun] {
        &self.unwrapped_layout.runs
    }

    /// The index corresponding to a given position in this layout for the given line height.
    ///
    /// See also [`Self::closest_index_for_position`].
    pub fn index_for_position(
        &self,
        position: Point<Pixels>,
        line_height: Pixels,
    ) -> Result<usize, usize> {
        self._index_for_position(position, line_height, false)
    }

    /// The closest index to a given position in this layout for the given line height.
    ///
    /// Closest means the character boundary closest to the given position.
    ///
    /// See also [`LineLayout::closest_index_for_x`].
    pub fn closest_index_for_position(
        &self,
        position: Point<Pixels>,
        line_height: Pixels,
    ) -> Result<usize, usize> {
        self._index_for_position(position, line_height, true)
    }

    fn _index_for_position(
        &self,
        mut position: Point<Pixels>,
        line_height: Pixels,
        closest: bool,
    ) -> Result<usize, usize> {
        let wrapped_line_ix = (position.y / line_height) as usize;

        let wrapped_line_start_index;
        let wrapped_line_start_x;
        if wrapped_line_ix > 0 {
            let Some(line_start_boundary) = self.wrap_boundaries.get(wrapped_line_ix - 1) else {
                return Err(0);
            };
            let run = &self.unwrapped_layout.runs[line_start_boundary.run_ix];
            let glyph = &run.glyphs[line_start_boundary.glyph_ix];
            wrapped_line_start_index = glyph.index;
            wrapped_line_start_x = glyph.position.x;
        } else {
            wrapped_line_start_index = 0;
            wrapped_line_start_x = Pixels::ZERO;
        };

        let wrapped_line_end_index;
        let wrapped_line_end_x;
        if wrapped_line_ix < self.wrap_boundaries.len() {
            let next_wrap_boundary_ix = wrapped_line_ix;
            let next_wrap_boundary = self.wrap_boundaries[next_wrap_boundary_ix];
            let run = &self.unwrapped_layout.runs[next_wrap_boundary.run_ix];
            let glyph = &run.glyphs[next_wrap_boundary.glyph_ix];
            wrapped_line_end_index = glyph.index;
            wrapped_line_end_x = glyph.position.x;
        } else {
            wrapped_line_end_index = self.unwrapped_layout.len;
            wrapped_line_end_x = self.unwrapped_layout.width;
        };

        let mut position_in_unwrapped_line = position;
        position_in_unwrapped_line.x += wrapped_line_start_x;
        if position_in_unwrapped_line.x < wrapped_line_start_x {
            Err(wrapped_line_start_index)
        } else if position_in_unwrapped_line.x >= wrapped_line_end_x {
            Err(wrapped_line_end_index)
        } else {
            if closest {
                Ok(self
                    .unwrapped_layout
                    .closest_index_for_x(position_in_unwrapped_line.x))
            } else {
                // The shaper can place a trailing zero-width wrap boundary glyph slightly past
                // the line's width, so the row can extend past where `index_for_x` has glyphs.
                self.unwrapped_layout
                    .index_for_x(position_in_unwrapped_line.x)
                    .ok_or(wrapped_line_end_index)
            }
        }
    }

    /// Returns the pixel position for the given byte index.
    pub fn position_for_index(&self, index: usize, line_height: Pixels) -> Option<Point<Pixels>> {
        let mut line_start_ix = 0;
        let mut line_end_indices = self
            .wrap_boundaries
            .iter()
            .map(|wrap_boundary| {
                let run = &self.unwrapped_layout.runs[wrap_boundary.run_ix];
                let glyph = &run.glyphs[wrap_boundary.glyph_ix];
                glyph.index
            })
            .chain([self.len()])
            .enumerate();
        for (ix, line_end_ix) in line_end_indices {
            let line_y = ix as f32 * line_height;
            if index < line_start_ix {
                break;
            } else if index > line_end_ix {
                line_start_ix = line_end_ix;
                continue;
            } else {
                let line_start_x = self.unwrapped_layout.x_for_index(line_start_ix);
                let x = self.unwrapped_layout.x_for_index(index) - line_start_x;
                return Some(point(x, line_y));
            }
        }

        None
    }
}

pub(crate) struct LineLayoutCache {
    state: RefCell<LineLayoutCacheState>,
    pub(crate) platform_text_system: Arc<dyn PlatformTextSystem>,
    /// Advances when [`TextSystem::add_fonts`] successfully changes the font database.
    font_generation: Arc<AtomicUsize>,
    /// Records the generation represented by both frame caches.
    cached_font_generation: AtomicUsize,
    pub(crate) shaping: crate::fast::text::LineShaping,
}

#[derive(Default)]
struct LineLayoutCacheState {
    previous_frame: FrameCache,
    current_frame: FrameCache,
}

#[derive(Default)]
pub(crate) struct FrameCache {
    pub(crate) lines: FxHashMap<Arc<CacheKey>, Arc<LineLayout>>,
    pub(crate) wrapped_lines: FxHashMap<Arc<CacheKey>, Arc<WrappedLineLayout>>,
    pub(crate) used_lines: Vec<Arc<CacheKey>>,
    pub(crate) used_wrapped_lines: Vec<Arc<CacheKey>>,
}

#[derive(Clone, Default)]
pub(crate) struct LineLayoutIndex {
    pub(crate) font_generation: usize,
    pub(crate) lines_index: usize,
    pub(crate) wrapped_lines_index: usize,
}

impl LineLayoutCache {
    pub fn new(
        platform_text_system: Arc<dyn PlatformTextSystem>,
        font_generation: Arc<AtomicUsize>,
    ) -> Self {
        let cached_font_generation = font_generation.load(Ordering::Acquire);
        Self {
            state: RefCell::default(),
            platform_text_system,
            font_generation,
            cached_font_generation: AtomicUsize::new(cached_font_generation),
            shaping: crate::fast::text::LineShaping::default(),
        }
    }

    pub fn layout_index(&self) -> LineLayoutIndex {
        let font_generation = self.clear_if_font_generation_changed();
        let state = self.state.borrow();
        let frame = &state.current_frame;
        LineLayoutIndex {
            font_generation,
            lines_index: frame.used_lines.len(),
            wrapped_lines_index: frame.used_wrapped_lines.len(),
        }
    }

    pub fn reuse_layouts(&self, range: Range<LineLayoutIndex>) {
        let font_generation = self.clear_if_font_generation_changed();
        if range.start.font_generation != font_generation
            || range.end.font_generation != font_generation
        {
            return;
        }
        let mut state = self.state.borrow_mut();
        let LineLayoutCacheState {
            previous_frame,
            current_frame,
        } = &mut *state;

        for key in &previous_frame.used_lines[range.start.lines_index..range.end.lines_index] {
            if let Some((key, line)) = previous_frame.lines.remove_entry(key) {
                current_frame.lines.insert(key, line);
            }
            current_frame.used_lines.push(key.clone());
        }

        for key in &previous_frame.used_wrapped_lines
            [range.start.wrapped_lines_index..range.end.wrapped_lines_index]
        {
            if let Some((key, line)) = previous_frame.wrapped_lines.remove_entry(key) {
                current_frame.wrapped_lines.insert(key, line);
            }
            current_frame.used_wrapped_lines.push(key.clone());
        }
    }

    pub fn truncate_layouts(&self, index: LineLayoutIndex) {
        let font_generation = self.clear_if_font_generation_changed();
        if index.font_generation != font_generation {
            return;
        }
        let mut state = self.state.borrow_mut();
        let current_frame = &mut state.current_frame;
        current_frame.used_lines.truncate(index.lines_index);
        current_frame
            .used_wrapped_lines
            .truncate(index.wrapped_lines_index);
    }

    pub fn finish_frame(&self) {
        let _font_generation = self.clear_if_font_generation_changed();
        let LineLayoutCacheState {
            previous_frame,
            current_frame,
        } = &mut *self.state.borrow_mut();
        crate::fast::text::carry_over_line_layouts(previous_frame, current_frame);
    }

    pub fn layout_wrapped_line<Text>(
        &self,
        text: Text,
        font_size: Pixels,
        runs: &[FontRun],
        wrap_width: Option<Pixels>,
        max_lines: Option<usize>,
    ) -> Arc<WrappedLineLayout>
    where
        Text: AsRef<str>,
        SharedString: From<Text>,
    {
        let _font_generation = self.clear_if_font_generation_changed();
        let key = &CacheKeyRef {
            text: text.as_ref(),
            font_size,
            runs,
            wrap_width,
            force_width: None,
        } as &dyn AsCacheKeyRef;

        let state = self.state.borrow();
        if let Some(layout) = state.current_frame.wrapped_lines.get(key) {
            return layout.clone();
        }
        drop(state);

        let mut state = self.state.borrow_mut();
        let previous_frame_entry = state.previous_frame.wrapped_lines.remove_entry(key);
        if let Some((key, layout)) = previous_frame_entry {
            state
                .current_frame
                .wrapped_lines
                .insert(key.clone(), layout.clone());
            state.current_frame.used_wrapped_lines.push(key);
            layout
        } else {
            drop(state);
            let text = SharedString::from(text);
            let unwrapped_layout = self.layout_line::<&SharedString>(&text, font_size, runs, None);
            let (unwrapped_layout, wrap_boundaries) = match wrap_width {
                Some(wrap_width) if has_right_to_left(&text) => self.wrap_right_to_left_line(
                    &text,
                    font_size,
                    runs,
                    unwrapped_layout,
                    wrap_width,
                    max_lines,
                ),
                Some(wrap_width) => {
                    let wrap_boundaries = unwrapped_layout.compute_wrap_boundaries(
                        text.as_ref(),
                        wrap_width,
                        max_lines,
                    );
                    (unwrapped_layout, wrap_boundaries)
                }
                None => (unwrapped_layout, SmallVec::new()),
            };
            let layout = Arc::new(WrappedLineLayout {
                unwrapped_layout,
                wrap_boundaries,
                wrap_width,
            });
            let key = Arc::new(CacheKey {
                text,
                font_size,
                runs: SmallVec::from(runs),
                wrap_width,
                force_width: None,
            });

            let mut state = self.state.borrow_mut();
            state
                .current_frame
                .wrapped_lines
                .insert(key.clone(), layout.clone());
            state.current_frame.used_wrapped_lines.push(key);

            layout
        }
    }

    /// Wraps a line holding right-to-left text the way a browser does: rows are
    /// broken in reading order and each row reads from its own right edge. The
    /// returned layout holds the rows one after the other, each row's glyphs in
    /// visual order from its left edge, which is the shape painting and hit testing
    /// expect of a wrapped line.
    fn wrap_right_to_left_line(
        &self,
        text: &SharedString,
        font_size: Pixels,
        runs: &[FontRun],
        unwrapped_layout: Arc<LineLayout>,
        wrap_width: Pixels,
        max_lines: Option<usize>,
    ) -> (Arc<LineLayout>, SmallVec<[WrapBoundary; 1]>) {
        let clusters = unwrapped_layout.clusters();
        let rows = unwrapped_layout.logical_wrap_ranges(text, &clusters, wrap_width, max_lines);
        if rows.len() < 2 {
            return (unwrapped_layout, SmallVec::new());
        }

        let mut layout = LineLayout {
            font_size,
            width: px(0.),
            ascent: unwrapped_layout.ascent,
            descent: unwrapped_layout.descent,
            runs: Vec::new(),
            len: text.len(),
        };
        let mut boundaries = SmallVec::new();
        for range in rows {
            let mut row_clusters: Vec<(Pixels, Pixels)> = clusters
                .iter()
                .filter(|(index, _, _)| range.contains(index))
                .map(|(_, left, width)| (*left, *width))
                .collect();
            row_clusters.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            // A row whose glyphs sit side by side in the whole line looks the same
            // laid out alone, so it reuses them. Only a row cut through a run of the
            // other direction comes apart there and is shaped again.
            let contiguous = row_clusters
                .windows(2)
                .all(|pair| (pair[0].0 + pair[0].1 - pair[1].0).abs() < px(0.5));

            let (row_glyphs, row_left, row_width) = if contiguous {
                let row_left = row_clusters.first().map_or(px(0.), |(left, _)| *left);
                let row_width = row_clusters.iter().map(|(_, width)| *width).sum::<Pixels>();
                let glyphs = unwrapped_layout
                    .runs
                    .iter()
                    .flat_map(|run| {
                        run.glyphs
                            .iter()
                            .filter(|glyph| range.contains(&glyph.index))
                            .map(|glyph| (run.font_id, glyph.clone()))
                    })
                    .collect::<Vec<_>>();
                (glyphs, row_left, row_width)
            } else {
                let row_runs = font_runs_in_range(runs, range.clone());
                let row = self.layout_line(
                    SharedString::from(text[range.clone()].to_string()),
                    font_size,
                    &row_runs,
                    None,
                );
                layout.ascent = layout.ascent.max(row.ascent);
                layout.descent = layout.descent.max(row.descent);
                let glyphs = row
                    .runs
                    .iter()
                    .flat_map(|run| {
                        run.glyphs.iter().map(|glyph| {
                            let mut glyph = glyph.clone();
                            glyph.index += range.start;
                            (run.font_id, glyph)
                        })
                    })
                    .collect::<Vec<_>>();
                (glyphs, px(0.), row.width)
            };

            let mut row_glyphs = row_glyphs;
            row_glyphs.sort_by(|a, b| {
                a.1.position
                    .x
                    .partial_cmp(&b.1.position.x)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let row_start_run = layout.runs.len();
            for (font_id, mut glyph) in row_glyphs {
                glyph.position.x = glyph.position.x - row_left + layout.width;
                let same_run = layout.runs.len() > row_start_run
                    && layout.runs.last().is_some_and(|run| run.font_id == font_id);
                match layout.runs.last_mut() {
                    Some(run) if same_run => run.glyphs.push(glyph),
                    _ => layout.runs.push(ShapedRun {
                        font_id,
                        glyphs: vec![glyph],
                    }),
                }
            }
            if row_start_run > 0 && layout.runs.len() > row_start_run {
                boundaries.push(WrapBoundary {
                    run_ix: row_start_run,
                    glyph_ix: 0,
                });
            }
            layout.width += row_width;
        }
        (Arc::new(layout), boundaries)
    }

    pub fn layout_line<Text>(
        &self,
        text: Text,
        font_size: Pixels,
        runs: &[FontRun],
        force_width: Option<Pixels>,
    ) -> Arc<LineLayout>
    where
        Text: AsRef<str>,
        SharedString: From<Text>,
    {
        let _font_generation = self.clear_if_font_generation_changed();
        let key = &CacheKeyRef {
            text: text.as_ref(),
            font_size,
            runs,
            wrap_width: None,
            force_width,
        } as &dyn AsCacheKeyRef;

        let state = self.state.borrow();
        if let Some(layout) = state.current_frame.lines.get(key) {
            return layout.clone();
        }
        drop(state);

        let mut state = self.state.borrow_mut();
        let LineLayoutCacheState {
            previous_frame,
            current_frame,
        } = &mut *state;
        if let Some((key, layout)) = previous_frame.lines.remove_entry(key) {
            current_frame.lines.insert(key.clone(), layout.clone());
            current_frame.used_lines.push(key);
            layout
        } else {
            let text = SharedString::from(text);
            let mut layout = crate::fast::text::shape_line(self, &text, font_size, runs);

            if let Some(force_width) = force_width {
                apply_force_width_to_layout(&mut layout, force_width);
            }

            let key = Arc::new(CacheKey {
                text,
                font_size,
                runs: SmallVec::from(runs),
                wrap_width: None,
                force_width,
            });
            let layout = Arc::new(layout);
            current_frame.lines.insert(key.clone(), layout.clone());
            current_frame.used_lines.push(key);
            layout
        }
    }

    fn clear_if_font_generation_changed(&self) -> usize {
        let font_generation = self.font_generation.load(Ordering::Acquire);
        if self.cached_font_generation.load(Ordering::Acquire) == font_generation {
            return font_generation;
        }

        let mut state = self.state.borrow_mut();
        if self.cached_font_generation.load(Ordering::Acquire) == font_generation {
            return font_generation;
        }

        *state = LineLayoutCacheState::default();
        self.cached_font_generation
            .store(font_generation, Ordering::Release);
        font_generation
    }
}

// Combining marks (e.g. Thai vowel signs, Arabic diacritics) are shaped by
// HarfBuzz at the same x position as their base character. The force-width
// loop must not advance the cell counter for these zero-advance glyphs,
// otherwise they get displaced into the next cell. We detect them by checking
// whether shaped x has advanced by at least half a cell beyond the last base.
fn apply_force_width_to_layout(layout: &mut LineLayout, force_width: Pixels) {
    let mut glyph_pos: usize = 0;
    // NEG_INFINITY ensures the first glyph is always classified as a base.
    let mut last_base_shaped_x = px(f32::NEG_INFINITY);
    let mut last_base_actual_x = px(0.);

    for run in layout.runs.iter_mut() {
        for glyph in run.glyphs.iter_mut() {
            let shaped_x = glyph.position.x;

            if shaped_x > last_base_shaped_x + force_width * 0.5 {
                let forced_x = glyph_pos * force_width;
                if (shaped_x - forced_x).abs() > px(1.) {
                    glyph.position.x = forced_x;
                }
                last_base_shaped_x = shaped_x;
                last_base_actual_x = glyph.position.x;
                glyph_pos += 1;
            } else {
                glyph.position.x = last_base_actual_x + (shaped_x - last_base_shaped_x);
            }
        }
    }
}

/// A run of text with a single font.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[expect(missing_docs)]
pub struct FontRun {
    pub len: usize,
    pub font_id: FontId,
}

trait AsCacheKeyRef {
    fn as_cache_key_ref(&self) -> CacheKeyRef<'_>;
}

#[derive(Clone, Debug, Eq)]
pub(crate) struct CacheKey {
    text: SharedString,
    font_size: Pixels,
    runs: SmallVec<[FontRun; 1]>,
    wrap_width: Option<Pixels>,
    force_width: Option<Pixels>,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
struct CacheKeyRef<'a> {
    text: &'a str,
    font_size: Pixels,
    runs: &'a [FontRun],
    wrap_width: Option<Pixels>,
    force_width: Option<Pixels>,
}

impl PartialEq for dyn AsCacheKeyRef + '_ {
    fn eq(&self, other: &dyn AsCacheKeyRef) -> bool {
        self.as_cache_key_ref() == other.as_cache_key_ref()
    }
}

impl Eq for dyn AsCacheKeyRef + '_ {}

impl Hash for dyn AsCacheKeyRef + '_ {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_cache_key_ref().hash(state)
    }
}

impl AsCacheKeyRef for CacheKey {
    fn as_cache_key_ref(&self) -> CacheKeyRef<'_> {
        CacheKeyRef {
            text: &self.text,
            font_size: self.font_size,
            runs: self.runs.as_slice(),
            wrap_width: self.wrap_width,
            force_width: self.force_width,
        }
    }
}

impl PartialEq for CacheKey {
    fn eq(&self, other: &Self) -> bool {
        self.as_cache_key_ref().eq(&other.as_cache_key_ref())
    }
}

impl Hash for CacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_cache_key_ref().hash(state);
    }
}

impl<'a> Borrow<dyn AsCacheKeyRef + 'a> for Arc<CacheKey> {
    fn borrow(&self) -> &(dyn AsCacheKeyRef + 'a) {
        self.as_ref() as &dyn AsCacheKeyRef
    }
}

impl AsCacheKeyRef for CacheKeyRef<'_> {
    fn as_cache_key_ref(&self) -> CacheKeyRef<'_> {
        *self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GlyphId;

    fn glyph_at(x: f32, index: usize) -> ShapedGlyph {
        ShapedGlyph {
            id: GlyphId(0),
            position: point(px(x), px(0.)),
            index,
            is_emoji: false,
        }
    }

    fn make_layout(glyphs: Vec<ShapedGlyph>) -> LineLayout {
        LineLayout {
            font_size: px(16.),
            width: px(100.),
            ascent: px(12.),
            descent: px(4.),
            runs: vec![ShapedRun {
                font_id: FontId(0),
                glyphs,
            }],
            len: 0,
        }
    }

    fn glyph_x_positions(layout: &LineLayout) -> Vec<f32> {
        layout.runs[0]
            .glyphs
            .iter()
            .map(|g| f32::from(g.position.x))
            .collect()
    }

    #[test]
    fn test_force_width_latin_unchanged() {
        let cell_width = px(8.);
        let mut layout = make_layout(vec![glyph_at(0., 0), glyph_at(8., 1), glyph_at(16., 2)]);

        apply_force_width_to_layout(&mut layout, cell_width);

        let positions = glyph_x_positions(&layout);
        assert_eq!(positions, vec![0., 8., 16.]);
    }

    #[test]
    fn test_force_width_combining_marks_not_advanced() {
        let cell_width = px(8.);
        // Simulates Thai "กี" — base consonant at x=0, combining vowel also at x=0
        let mut layout = make_layout(vec![
            glyph_at(0., 0), // ก (base)
            glyph_at(0., 3), // ี (combining mark, same x)
        ]);

        apply_force_width_to_layout(&mut layout, cell_width);

        let positions = glyph_x_positions(&layout);
        assert_eq!(positions, vec![0., 0.]);
    }

    #[test]
    fn test_force_width_base_after_combining_mark() {
        let cell_width = px(8.);
        let mut layout = make_layout(vec![glyph_at(0., 0), glyph_at(0., 3), glyph_at(8., 6)]);

        apply_force_width_to_layout(&mut layout, cell_width);

        let positions = glyph_x_positions(&layout);
        assert_eq!(positions, vec![0., 0., 8.]);
    }

    #[test]
    fn test_force_width_multiple_combining_marks() {
        let cell_width = px(8.);
        // Simulates "ก้" — base + vowel + tone mark (two combining marks stacked)
        let mut layout = make_layout(vec![
            glyph_at(0., 0), // ก (base)
            glyph_at(0., 3), // vowel (combining)
            glyph_at(0., 6), // tone mark (combining)
            glyph_at(8., 9), // next base
        ]);

        apply_force_width_to_layout(&mut layout, cell_width);

        let positions = glyph_x_positions(&layout);
        assert_eq!(positions, vec![0., 0., 0., 8.]);
    }

    #[test]
    fn test_force_width_corrects_drifted_base_positions() {
        let cell_width = px(8.);
        // Font metrics don't perfectly match cell grid — glyphs drift >1px from cell boundary
        let mut layout = make_layout(vec![
            glyph_at(0.5, 0),  // within 1px tolerance, kept as-is
            glyph_at(10.2, 1), // >1px off from 8.0, corrected
            glyph_at(19.8, 2), // >1px off from 16.0, corrected
        ]);

        apply_force_width_to_layout(&mut layout, cell_width);

        let positions = glyph_x_positions(&layout);
        assert_eq!(positions, vec![0.5, 8., 16.]);
    }

    #[test]
    fn test_force_width_combining_mark_after_within_tolerance_base() {
        let cell_width = px(8.);
        // Base glyph is within 1px of grid so it keeps its shaped position.
        // The combining mark must align to the base's actual position, not the grid slot.
        let mut layout = make_layout(vec![glyph_at(0.5, 0), glyph_at(0.5, 3)]);

        apply_force_width_to_layout(&mut layout, cell_width);

        let positions = glyph_x_positions(&layout);
        assert_eq!(positions, vec![0.5, 0.5]);
    }
}
