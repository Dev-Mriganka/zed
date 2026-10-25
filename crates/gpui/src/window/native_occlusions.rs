//! Regions of the window a native child view draws over, so nothing GPUI paints there can be
//! seen: an embedded browser, for one. The element that positions such a view records its frame
//! during prepaint, and overlays that must stay readable, tooltips first of all, lay themselves
//! out around the recorded regions instead of only around the window's edges.

use std::ops::Range;

use crate::{Bounds, Pixels, Point, Size, point, px, size};

use super::Window;

/// How far above and below an anchor a region still counts as sitting on the anchor's row. A
/// tooltip opens next to its anchor, so a region this close is one it could run into.
const ROW_REACH: Pixels = px(48.);

impl Window {
    /// Records `bounds` as a region a native child view covers in the frame being drawn. The
    /// element that positions the view calls this from its prepaint on every frame the view is
    /// visible; the record lasts one frame, like a hitbox.
    pub fn occlude_native_region(&mut self, bounds: Bounds<Pixels>) {
        self.invalidator.debug_assert_prepaint();
        self.next_frame.native_occlusions.push(bounds);
    }

    /// The regions native child views cover: the ones the frame being drawn has recorded so far,
    /// or the last frame's until it records one, so an overlay laid out early in a frame sees
    /// them too. That keeps the regions one frame longer than the views themselves; the next
    /// frame corrects it.
    pub fn native_occlusions(&self) -> &[Bounds<Pixels>] {
        if self.next_frame.native_occlusions.is_empty() {
            &self.rendered_frame.native_occlusions
        } else {
            &self.next_frame.native_occlusions
        }
    }
}

/// The horizontal room an overlay anchored at `anchor` has on its row: from the right edge of the
/// nearest covered region left of the anchor to the left edge of the nearest one right of it,
/// or the window's edges where there is none. Regions on other rows do not narrow it.
pub fn native_occlusion_row_gap(
    anchor: Bounds<Pixels>,
    occlusions: &[Bounds<Pixels>],
    viewport_size: Size<Pixels>,
) -> Range<Pixels> {
    let row = Bounds::new(
        point(px(0.), anchor.top() - ROW_REACH),
        size(viewport_size.width, anchor.size.height + ROW_REACH * 2.),
    );
    let center_x = anchor.center().x;
    let mut left = px(0.);
    let mut right = viewport_size.width;
    for region in occlusions {
        if !region.intersects(&row) {
            continue;
        }
        if region.right() <= center_x {
            left = left.max(region.right());
        } else if region.left() >= center_x {
            right = right.min(region.left());
        }
    }
    left..right
}

/// Moves `bounds` out of every covered region it overlaps, one region at a time, by the shortest
/// shift that keeps it inside `viewport`. Returns it unchanged when no shift can clear a region.
pub fn nudge_out_of_native_occlusions(
    mut bounds: Bounds<Pixels>,
    occlusions: &[Bounds<Pixels>],
    viewport: Bounds<Pixels>,
) -> Bounds<Pixels> {
    // Each pass clears one region; a corner between two regions settles within a few.
    for _ in 0..4 {
        let Some(region) = occlusions.iter().find(|region| region.intersects(&bounds)) else {
            break;
        };
        let candidates = [
            point(region.left() - bounds.size.width, bounds.origin.y),
            point(region.right(), bounds.origin.y),
            point(bounds.origin.x, region.top() - bounds.size.height),
            point(bounds.origin.x, region.bottom()),
        ];
        let fits = |origin: &Point<Pixels>| {
            origin.x >= viewport.left()
                && origin.y >= viewport.top()
                && origin.x + bounds.size.width <= viewport.right()
                && origin.y + bounds.size.height <= viewport.bottom()
        };
        let shift = |origin: &Point<Pixels>| {
            let dx = (origin.x - bounds.origin.x).as_f32();
            let dy = (origin.y - bounds.origin.y).as_f32();
            dx * dx + dy * dy
        };
        let Some(origin) = candidates
            .iter()
            .filter(|origin| fits(origin))
            .min_by(|a, b| shift(a).total_cmp(&shift(b)))
        else {
            break;
        };
        bounds.origin = *origin;
    }
    bounds
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds(x: f32, y: f32, w: f32, h: f32) -> Bounds<Pixels> {
        Bounds::new(point(px(x), px(y)), size(px(w), px(h)))
    }

    #[test]
    fn row_gap_stops_at_regions_on_the_row_only() {
        let viewport = size(px(1000.), px(600.));
        let browser = bounds(700., 0., 300., 600.);
        let sidebar = bounds(0., 0., 200., 600.);
        let far_below = bounds(400., 500., 300., 100.);
        let anchor = bounds(650., 300., 20., 20.);
        let gap = native_occlusion_row_gap(anchor, &[browser, sidebar, far_below], viewport);
        assert_eq!(gap, px(200.)..px(700.));
        let gap = native_occlusion_row_gap(anchor, &[], viewport);
        assert_eq!(gap, px(0.)..px(1000.));
    }

    #[test]
    fn nudge_takes_the_shortest_shift_that_stays_in_the_window() {
        let viewport = bounds(0., 0., 1000., 600.);
        let browser = bounds(700., 0., 300., 600.);
        let tooltip = bounds(650., 300., 100., 30.);
        let moved = nudge_out_of_native_occlusions(tooltip, &[browser], viewport);
        assert_eq!(moved, bounds(600., 300., 100., 30.));

        let unmovable = bounds(0., 0., 1000., 600.);
        assert_eq!(
            nudge_out_of_native_occlusions(tooltip, &[unmovable], viewport),
            tooltip
        );
        assert_eq!(
            nudge_out_of_native_occlusions(tooltip, &[], viewport),
            tooltip
        );
    }
}
