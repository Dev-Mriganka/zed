use crate::{
    AnyElement, AnyEntity, App, AppContext, Asset, AssetLogger, Bounds, Context, Element,
    ElementId, Entity, Global, GlobalElementId, ImageAssetLoader, ImageCacheError,
    InspectorElementId, IntoElement, LayoutId, ParentElement, Pixels, RenderImage, Resource, Style,
    StyleRefinement, Styled, Window, hash,
};

use crate::asset_cache::CachedLoad;
use refineable::Refineable;
use scheduler::Instant;
use smallvec::SmallVec;
use std::{collections::HashMap, fmt, sync::Arc, time::Duration};

/// An image cache element, all its child img elements will use the cache specified by this element.
/// Note that this could as simple as passing an `Entity<T: ImageCache>`
pub fn image_cache(image_cache_provider: impl ImageCacheProvider) -> ImageCacheElement {
    ImageCacheElement {
        image_cache_provider: Box::new(image_cache_provider),
        style: StyleRefinement::default(),
        children: SmallVec::default(),
    }
}

/// A dynamically typed image cache, which can be used to store any image cache
#[derive(Clone)]
pub struct AnyImageCache {
    image_cache: AnyEntity,
    load_fn: fn(
        image_cache: &AnyEntity,
        resource: &Resource,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<Result<Arc<RenderImage>, ImageCacheError>>,
}

impl<I: ImageCache> From<Entity<I>> for AnyImageCache {
    fn from(image_cache: Entity<I>) -> Self {
        Self {
            image_cache: image_cache.into_any(),
            load_fn: any_image_cache::load::<I>,
        }
    }
}

impl AnyImageCache {
    /// Load an image given a resource
    /// returns the result of loading the image if it has finished loading, or None if it is still loading
    pub fn load(
        &self,
        resource: &Resource,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<Result<Arc<RenderImage>, ImageCacheError>> {
        (self.load_fn)(&self.image_cache, resource, window, cx)
    }
}

mod any_image_cache {
    use super::*;

    pub(crate) fn load<I: 'static + ImageCache>(
        image_cache: &AnyEntity,
        resource: &Resource,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<Result<Arc<RenderImage>, ImageCacheError>> {
        let image_cache = image_cache.clone().downcast::<I>().unwrap();
        image_cache.update(cx, |image_cache, cx| image_cache.load(resource, window, cx))
    }
}

/// An image cache element.
pub struct ImageCacheElement {
    image_cache_provider: Box<dyn ImageCacheProvider>,
    style: StyleRefinement,
    children: SmallVec<[AnyElement; 2]>,
}

impl ParentElement for ImageCacheElement {
    fn extend(&mut self, elements: impl IntoIterator<Item = AnyElement>) {
        self.children.extend(elements)
    }
}

impl Styled for ImageCacheElement {
    fn style(&mut self) -> &mut StyleRefinement {
        &mut self.style
    }
}

impl IntoElement for ImageCacheElement {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Element for ImageCacheElement {
    type RequestLayoutState = SmallVec<[LayoutId; 4]>;
    type PrepaintState = ();

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let image_cache = self.image_cache_provider.provide(window, cx);
        window.with_image_cache(Some(image_cache), |window| {
            let child_layout_ids = self
                .children
                .iter_mut()
                .map(|child| child.request_layout(window, cx))
                .collect::<SmallVec<_>>();
            let mut style = Style::default();
            style.refine(&self.style);
            let layout_id = window.request_layout(style, child_layout_ids.iter().copied(), cx);
            (layout_id, child_layout_ids)
        })
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) -> Self::PrepaintState {
        for child in &mut self.children {
            child.prepaint(window, cx);
        }
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        _prepaint: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        let image_cache = self.image_cache_provider.provide(window, cx);
        window.with_image_cache(Some(image_cache), |window| {
            for child in &mut self.children {
                child.paint(window, cx);
            }
        })
    }
}

/// Owns an image load and its cached result.
///
/// Dropping the entry cancels a pending load. Image caches must remove loaded
/// images from windows before discarding the entry.
pub struct ImageCacheItem(CachedLoad<Result<Arc<RenderImage>, ImageCacheError>>);

impl std::fmt::Debug for ImageCacheItem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImageCacheItem")
            .field("result", &self.get())
            .finish()
    }
}

impl ImageCacheItem {
    /// Starts loading an image into a new cache entry.
    pub fn new(source: &Resource, cx: &mut App) -> Self {
        let future = AssetLogger::<ImageAssetLoader>::load(source.clone(), cx);
        Self(CachedLoad::new(future, cx))
    }

    /// Returns the cached result without subscribing to completion notifications.
    pub fn get(&self) -> Option<Result<Arc<RenderImage>, ImageCacheError>> {
        self.0.get()
    }

    /// Returns the cached result or subscribes the current view to completion.
    pub fn use_image(&self, window: &Window) -> Option<Result<Arc<RenderImage>, ImageCacheError>> {
        self.0.use_by(window.current_view())
    }
}

/// An object that can handle the caching and unloading of images.
/// Implementations of this trait should ensure that images are removed from all windows when they are no longer needed.
pub trait ImageCache: 'static {
    /// Load an image given a resource
    /// returns the result of loading the image if it has finished loading, or None if it is still loading
    fn load(
        &mut self,
        resource: &Resource,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<Result<Arc<RenderImage>, ImageCacheError>>;
}

/// An object that can create an ImageCache during the render phase.
/// See the ImageCache trait for more information.
pub trait ImageCacheProvider: 'static {
    /// Called during the request_layout phase to create an ImageCache.
    fn provide(&mut self, _window: &mut Window, _cx: &mut App) -> AnyImageCache;
}

impl<T: ImageCache> ImageCacheProvider for Entity<T> {
    fn provide(&mut self, _window: &mut Window, _cx: &mut App) -> AnyImageCache {
        self.clone().into()
    }
}

/// An implementation of ImageCache, that uses an LRU caching strategy to unload images when the cache is full
pub struct RetainAllImageCache(HashMap<u64, ImageCacheItem>);

impl fmt::Debug for RetainAllImageCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HashMapImageCache")
            .field("num_images", &self.0.len())
            .finish()
    }
}

impl RetainAllImageCache {
    /// Create a new image cache.
    #[inline]
    pub fn new(cx: &mut App) -> Entity<Self> {
        let e = cx.new(|_cx| RetainAllImageCache(HashMap::new()));
        cx.observe_release(&e, |image_cache, cx| {
            for (_, item) in std::mem::replace(&mut image_cache.0, HashMap::new()) {
                if let Some(Ok(image)) = item.get() {
                    cx.drop_image(image, None);
                }
            }
        })
        .detach();
        e
    }

    /// Load an image from the given source.
    ///
    /// Returns `None` if the image is loading.
    pub fn load(
        &mut self,
        source: &Resource,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<Result<Arc<RenderImage>, ImageCacheError>> {
        let hash = hash(source);

        self.0
            .entry(hash)
            .or_insert_with(|| ImageCacheItem::new(source, cx))
            .use_image(window)
    }

    /// Clear the image cache.
    pub fn clear(&mut self, window: &mut Window, cx: &mut App) {
        for (_, item) in std::mem::replace(&mut self.0, HashMap::new()) {
            if let Some(Ok(image)) = item.get() {
                cx.drop_image(image, Some(window));
            }
        }
    }

    /// Remove the image from the cache by the given source.
    pub fn remove(&mut self, source: &Resource, window: &mut Window, cx: &mut App) {
        let hash = hash(source);
        if let Some(item) = self.0.remove(&hash)
            && let Some(Ok(image)) = item.get()
        {
            cx.drop_image(image, Some(window));
        }
    }

    /// Returns the number of images in the cache.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns true if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl ImageCache for RetainAllImageCache {
    fn load(
        &mut self,
        resource: &Resource,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<Result<Arc<RenderImage>, ImageCacheError>> {
        RetainAllImageCache::load(self, resource, window, cx)
    }
}

/// Constructs a retain-all image cache that uses the element state associated with the given ID.
pub fn retain_all(id: impl Into<ElementId>) -> RetainAllImageCacheProvider {
    RetainAllImageCacheProvider { id: id.into() }
}

/// A provider struct for creating a retain-all image cache inline
pub struct RetainAllImageCacheProvider {
    id: ElementId,
}

impl ImageCacheProvider for RetainAllImageCacheProvider {
    fn provide(&mut self, window: &mut Window, cx: &mut App) -> AnyImageCache {
        window
            .with_global_id(self.id.clone(), |global_id, window| {
                window.with_element_state::<Entity<RetainAllImageCache>, _>(
                    global_id,
                    |cache, _window| {
                        let mut cache = cache.unwrap_or_else(|| RetainAllImageCache::new(cx));
                        (cache.clone(), cache)
                    },
                )
            })
            .into()
    }
}

/// Ghostex: how many bytes of decoded pixels the images fetched from a URI keep, app-wide.
const URI_IMAGE_BUDGET_BYTES: usize = 64 * 1024 * 1024;
/// An image drawn this recently is never let go.
const URI_IMAGE_RECENTLY_DRAWN: Duration = Duration::from_secs(30);
/// How often the cache checks whether it is over its budget, while it holds anything.
const URI_IMAGE_SWEEP_INTERVAL: Duration = Duration::from_secs(30);
/// How long the sweep gives every visible window to draw the frame it asked for.
const URI_IMAGE_REDRAW_SETTLE: Duration = Duration::from_secs(1);

/// Ghostex: the cache an `img` element loads a [`Resource::Uri`] through when no image cache is
/// given. These images used to go into the asset cache, which keeps everything it loads for the
/// life of the app; this one keeps at most `URI_IMAGE_BUDGET_BYTES` of decoded pixels and lets
/// the images drawn longest ago go first, so one drawn again is fetched and decoded again.
///
/// An image is never let go while it is on screen. A view drawn from its cache paints its images
/// without asking for them, so "not asked for lately" does not mean "not on screen": before
/// letting anything go, the sweep redraws every window from scratch, and only the images that
/// frame did not draw are let go, from every window's atlas too.
pub(crate) struct UriImageCache {
    entries: HashMap<u64, UriImage>,
    sweeping: bool,
    /// The last redraw found every image it could let go on screen, so no window is redrawn for
    /// the cache again until another image is loaded.
    settled: bool,
}

struct UriImage {
    item: ImageCacheItem,
    drawn_at: Instant,
}

impl UriImage {
    fn loaded_bytes(&self) -> usize {
        match self.item.get() {
            Some(Ok(image)) => (0..image.frame_count())
                .map(|frame| image.as_bytes(frame).map_or(0, <[u8]>::len))
                .sum(),
            _ => 0,
        }
    }
}

struct UriImageCacheGlobal(Entity<UriImageCache>);

impl Global for UriImageCacheGlobal {}

/// What the sweep does after checking the cache.
enum UriImageSweep {
    /// Nothing is cached: the sweep stops until something is.
    Stop,
    /// Within budget, or everything was drawn lately.
    Wait,
    /// Every window was asked to draw from scratch at this time.
    Redrawn(Instant),
}

impl UriImageCache {
    fn entity(cx: &mut App) -> Entity<Self> {
        if let Some(global) = cx.try_global::<UriImageCacheGlobal>() {
            return global.0.clone();
        }
        let cache = cx.new(|_| Self {
            entries: HashMap::new(),
            sweeping: false,
            settled: false,
        });
        cx.set_global(UriImageCacheGlobal(cache.clone()));
        cache
    }

    /// The image at `resource`, loading it the first time. `window`'s current view is redrawn
    /// when it has loaded if `notify` is set.
    pub(crate) fn load(
        resource: &Resource,
        notify: bool,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<Result<Arc<RenderImage>, ImageCacheError>> {
        Self::entity(cx).update(cx, |cache, cx| {
            let image = match cache.entries.entry(hash(resource)) {
                std::collections::hash_map::Entry::Occupied(image) => image.into_mut(),
                std::collections::hash_map::Entry::Vacant(slot) => {
                    cache.settled = false;
                    slot.insert(UriImage {
                        item: ImageCacheItem::new(resource, cx),
                        drawn_at: Instant::now(),
                    })
                }
            };
            image.drawn_at = Instant::now();
            let loaded = if notify {
                image.item.use_image(window)
            } else {
                image.item.get()
            };
            cache.start_sweeping(cx);
            loaded
        })
    }

    /// Lets the image at `resource` go now, from every window's atlas too.
    pub(crate) fn remove(resource: &Resource, cx: &mut App) {
        let Some(cache) = cx
            .try_global::<UriImageCacheGlobal>()
            .map(|global| global.0.clone())
        else {
            return;
        };
        let removed = cache.update(cx, |cache, _| cache.entries.remove(&hash(resource)));
        if let Some(Ok(image)) = removed.and_then(|removed| removed.item.get()) {
            cx.drop_image(image, None);
        }
    }

    /// Whether `resource` is cached (loading or loaded).
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn contains(resource: &Resource, cx: &App) -> bool {
        cx.try_global::<UriImageCacheGlobal>()
            .is_some_and(|global| global.0.read(cx).entries.contains_key(&hash(resource)))
    }

    fn start_sweeping(&mut self, cx: &mut Context<Self>) {
        if self.sweeping {
            return;
        }
        self.sweeping = true;
        cx.spawn(async move |this, cx| {
            loop {
                cx.background_executor()
                    .timer(URI_IMAGE_SWEEP_INTERVAL)
                    .await;
                let Ok(step) = this.update(cx, |cache, cx| cache.begin_sweep(cx)) else {
                    return;
                };
                match step {
                    UriImageSweep::Stop => return,
                    UriImageSweep::Wait => {}
                    UriImageSweep::Redrawn(at) => {
                        cx.background_executor()
                            .timer(URI_IMAGE_REDRAW_SETTLE)
                            .await;
                        if this
                            .update(cx, |cache, cx| cache.finish_sweep(at, cx))
                            .is_err()
                        {
                            return;
                        }
                    }
                }
            }
        })
        .detach();
    }

    fn begin_sweep(&mut self, cx: &mut Context<Self>) -> UriImageSweep {
        if self.entries.is_empty() {
            self.sweeping = false;
            return UriImageSweep::Stop;
        }
        let total: usize = self.entries.values().map(UriImage::loaded_bytes).sum();
        let stale = self.entries.values().any(|image| {
            image.drawn_at.elapsed() >= URI_IMAGE_RECENTLY_DRAWN && image.loaded_bytes() > 0
        });
        if total <= URI_IMAGE_BUDGET_BYTES || !stale || self.settled {
            return UriImageSweep::Wait;
        }
        let at = Instant::now();
        cx.refresh_windows();
        UriImageSweep::Redrawn(at)
    }

    /// Lets go, oldest first, the images the redraw asked for at `redrawn_at` did not draw, until
    /// the cache is within its budget.
    fn finish_sweep(&mut self, redrawn_at: Instant, cx: &mut Context<Self>) {
        let mut total: usize = self.entries.values().map(UriImage::loaded_bytes).sum();
        let mut stale: Vec<(Instant, u64, usize)> = self
            .entries
            .iter()
            .filter(|(_, image)| {
                image.drawn_at < redrawn_at && image.drawn_at.elapsed() >= URI_IMAGE_RECENTLY_DRAWN
            })
            .map(|(key, image)| (image.drawn_at, *key, image.loaded_bytes()))
            .filter(|(_, _, bytes)| *bytes > 0)
            .collect();
        stale.sort_by_key(|(drawn_at, ..)| *drawn_at);
        for (_, key, bytes) in stale {
            if total <= URI_IMAGE_BUDGET_BYTES {
                break;
            }
            total -= bytes;
            if let Some(Ok(image)) = self.entries.remove(&key).and_then(|image| image.item.get()) {
                cx.drop_image(image, None);
            }
        }
        self.settled = total > URI_IMAGE_BUDGET_BYTES;
    }
}
