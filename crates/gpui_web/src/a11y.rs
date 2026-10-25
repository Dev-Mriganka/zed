//! Accessibility on the web: the AccessKit tree gpui builds for every frame, mirrored into the
//! page's DOM behind the canvas.
//!
//! AccessKit has no browser adapter, and a canvas is one opaque node to the browser. So every
//! node of the tree becomes an element with the matching ARIA role, label and state, placed at
//! the node's bounds and letting pointer events through, so that a screen reader, DevTools,
//! Playwright or `getByRole` sees the same tree the desktop platforms expose, and a click on a
//! mirrored element lands on the canvas at that spot. `window.gpuiA11y` also dispatches AccessKit
//! actions straight to gpui for tests that want to press a button without a pointer.
//!
//! The mirror is on unless the page is loaded with `?a11y=off`, because the tree's main use in a
//! browser is automated testing. It is diffed per frame: gpui sends the whole tree, and the DOM
//! only changes where the tree did.
use std::cell::RefCell;
use std::rc::Rc;

use accesskit::{ActionRequest, NodeId, TreeUpdate};
use gpui::A11yCallbacks;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(inline_js = r#"
const ROLES = {
  Window: 'application', Button: 'button', CheckBox: 'checkbox', Switch: 'switch', Tab: 'tab',
  TabList: 'tablist', TabPanel: 'tabpanel', TreeItem: 'treeitem', Tree: 'tree', Group: 'group',
  Navigation: 'navigation', Document: 'document', Article: 'article', Label: 'text',
  StaticText: 'text', TextRun: 'text', TextInput: 'textbox', MultilineTextInput: 'textbox',
  SearchInput: 'searchbox', Menu: 'menu', MenuItem: 'menuitem', MenuItemCheckBox: 'menuitemcheckbox',
  MenuItemRadio: 'menuitemradio', MenuBar: 'menubar', ListBox: 'listbox', ListBoxOption: 'option',
  List: 'list', ListItem: 'listitem', Heading: 'heading', Link: 'link', Status: 'status',
  Alert: 'alert', Dialog: 'dialog', Tooltip: 'tooltip', Terminal: 'log', Log: 'log',
  ScrollBar: 'scrollbar', Slider: 'slider', SpinButton: 'spinbutton', ProgressIndicator: 'progressbar',
  RadioButton: 'radio', RadioGroup: 'radiogroup', Image: 'img', Toolbar: 'toolbar', Region: 'region',
  Paragraph: 'paragraph', Row: 'row', Cell: 'cell', Table: 'table', Grid: 'grid', Form: 'form',
};

const disabled = new URLSearchParams(location.search).get('a11y') === 'off';
// One mirror per gpui window: the first is the page's main window, every later one an overlay (a menu, a picker, a dialog) drawn on its own canvas, with node ids of its own.
const windows = new Map();
let nextWindow = 1;
function nodeWindow(el) {
  return windows.get(Number(el.dataset.gpuiWindow));
}
function act(el, action) {
  nodeWindow(el)?.dispatch(el.dataset.gpuiNode, action);
}
function findElement(id) {
  for (const mirror of windows.values()) {
    for (const el of mirror.elements.values()) if (el.dataset.gpuiId === id) return el;
  }
  return null;
}

export function gpui_a11y_start(dispatch, canvas, input) {
  if (disabled) return 0;
  const key = nextWindow++;
  const root = document.createElement('div');
  root.id = key === 1 ? 'gpui-a11y' : `gpui-a11y-${key}`;
  root.dataset.gpuiA11yRoot = String(key);
  root.setAttribute('role', 'application');
  root.setAttribute('aria-label', 'GPUI');
  Object.assign(root.style, { position: 'fixed', inset: '0', pointerEvents: 'none', zIndex: String(999 + key), overflow: 'hidden' });
  document.body.appendChild(root);
  windows.set(key, { key, root, canvas, input, dispatch, elements: new Map(), latest: null, timer: null });
  window.gpuiA11y ??= {
    // Dispatch an AccessKit action ("Click", "Focus", "Blur", ...) to the node with this id: the main window's ids are plain, an overlay's are `<window>:<id>`.
    act(id, action = 'Click') { const el = findElement(String(id)); if (el) act(el, action); },
    // The element mirroring a node, for tests that already hold an id.
    element(id) { return findElement(String(id)); },
    // Every node of every window as {id, role, label, description, expanded, selected, toggled, bounds}: a cheap snapshot for a test to assert on.
    snapshot() {
      return [...windows.values()].flatMap((mirror) => [...mirror.elements.values()]).map((el) => ({
        id: el.dataset.gpuiId, role: el.getAttribute('role'), label: el.getAttribute('aria-label') ?? '',
        description: el.getAttribute('aria-description') ?? undefined,
        expanded: el.getAttribute('aria-expanded') ?? undefined, selected: el.getAttribute('aria-selected') ?? undefined,
        checked: el.getAttribute('aria-checked') ?? undefined, focused: el.dataset.gpuiFocused === '1' || undefined,
        bounds: (({ left, top, width, height }) => [left, top, width, height].map(Math.round))(el.getBoundingClientRect()),
      }));
    },
  };
  return key;
}

// A closed window takes its mirror with it.
export function gpui_a11y_stop(key) {
  const mirror = windows.get(key);
  if (!mirror) return;
  clearTimeout(mirror.timer);
  mirror.root.remove();
  windows.delete(key);
}

// Real pointers pass through the mirror to the canvas, so a click that reaches a mirrored element was dispatched by a script (a test driver's synthetic DOM click). It becomes the node's Click, which gpui answers with a press at the node's centre when the element has no click handler of its own (a menu row that acts on mouse-down).
function onMirrorClick(event) {
  event.stopPropagation();
  act(event.currentTarget, 'Click');
}

// A wheel event that reaches a mirrored element was dispatched by a script too (a driver scrolling a ref); it is replayed on the element's own canvas, at the event's point or else the element's centre.
function onMirrorWheel(event) {
  const el = event.currentTarget;
  event.preventDefault();
  event.stopPropagation();
  const box = el.getBoundingClientRect();
  const clientX = event.clientX || box.left + box.width / 2;
  const clientY = event.clientY || box.top + box.height / 2;
  nodeWindow(el)?.canvas.dispatchEvent(new WheelEvent('wheel', {
    bubbles: true, cancelable: true, clientX, clientY, deltaX: event.deltaX, deltaY: event.deltaY, deltaMode: event.deltaMode,
    shiftKey: event.shiftKey, ctrlKey: event.ctrlKey, altKey: event.altKey, metaKey: event.metaKey,
  }));
}

// A mirrored text field is focusable and editable so a driver can type into it the way it types into a page: its keys and inserted text are replayed on its window's own input element, which is what gpui reads, after the gpui field has been clicked into focus. The window stays active meanwhile (see gpui_web's blur handler).
const TEXT_ROLES = new Set(['TextInput', 'MultilineTextInput', 'SearchInput']);
// Resolves once gpui reports the field focused. Click, not Focus: gpui's text inputs advertise Focus but take it, and their caret, from a click, and it can ignore clicks for a few hundred milliseconds after DOM focus moves onto the mirror, so the click repeats until the mirror reports it. Keyed on the first key rather than on `focus`, which a page in an unfocused window never receives. Timers, not animation frames, which an occluded window never runs.
const fieldReady = new WeakMap();
function focusField(el) {
  if (el.dataset.gpuiFocused === '1') return Promise.resolve();
  let pending = fieldReady.get(el);
  if (!pending) {
    pending = new Promise((resolve) => {
      let attempts = 0;
      const attempt = () => {
        if (el.dataset.gpuiFocused === '1' || attempts >= 8) { fieldReady.delete(el); resolve(); return; }
        attempts += 1;
        act(el, 'Click');
        setTimeout(attempt, 200);
      };
      attempt();
    });
    fieldReady.set(el, pending);
  }
  return pending;
}
// Replays run in arrival order, each after the field is focused.
const replayQueue = new WeakMap();
function whenFocused(el, replay) {
  const next = (replayQueue.get(el) ?? Promise.resolve()).then(() => focusField(el)).then(() => replay(nodeWindow(el)?.input));
  replayQueue.set(el, next);
}
function replayKey(input, type, init) {
  input?.dispatchEvent(new KeyboardEvent(type, { bubbles: true, cancelable: true, ...init }));
}
// gpui imports typed text from the input element's `input` event (the IME path) and leaves `insertText` to the browser's default action, which a script-dispatched `beforeinput` never runs, so the edit is applied to the element here before `input` is raised.
function replayText(input, text) {
  if (!input || text.length === 0) return;
  const start = input.selectionStart ?? input.value.length;
  input.setRangeText(text, start, input.selectionEnd ?? start, 'end');
  input.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertText', data: text }));
}
function forwardKey(event) {
  event.preventDefault();
  event.stopPropagation();
  const { type, key, code, shiftKey, ctrlKey, altKey, metaKey, repeat } = event;
  whenFocused(event.currentTarget, (input) => replayKey(input, type, { key, code, shiftKey, ctrlKey, altKey, metaKey, repeat }));
}
function forwardInsertedText(event) {
  event.preventDefault();
  const { inputType } = event;
  const text = event.data ?? '';
  whenFocused(event.currentTarget, (input) => {
    const press = (key, shiftKey = false) => {
      replayKey(input, 'keydown', { key, code: key, shiftKey });
      replayKey(input, 'keyup', { key, code: key, shiftKey });
    };
    if (inputType === 'insertLineBreak' || inputType === 'insertParagraph') press('Enter', true);
    else if (inputType === 'deleteContentBackward') press('Backspace');
    else replayText(input, text);
  });
}
// An insertion that raised no cancellable beforeinput (execCommand) lands in the mirror element; it is replayed and cleared.
function forwardLeftoverText(event) {
  const el = event.currentTarget;
  const text = el.textContent ?? '';
  el.textContent = '';
  if (text.length > 0) whenFocused(el, (input) => replayText(input, text));
}

function setAttr(el, name, value) {
  if (value === undefined || value === null) {
    if (el.hasAttribute(name)) el.removeAttribute(name);
  } else if (el.getAttribute(name) !== String(value)) {
    el.setAttribute(name, String(value));
  }
}

// `nodes` is a JSON array of [id, {role, label, description, bounds, expanded, selected, toggled, actions, children}], and `focus` the focused id.
// gpui sends a tree per drawn frame; the DOM is brought up to date with the latest one at most ten times a second, so a streaming chat does not spend its frames on ARIA attributes.
export function gpui_a11y_update(key, nodesJson, focus, dpr) {
  const mirror = windows.get(key);
  if (!mirror) return;
  mirror.latest = [nodesJson, dpr];
  if (mirror.timer === null) {
    mirror.timer = setTimeout(() => {
      mirror.timer = null;
      const [json, ratio] = mirror.latest;
      mirror.latest = null;
      applyTree(mirror, json, ratio);
    }, 100);
  }
}

function applyTree(mirror, nodesJson, dpr) {
  const { elements, root } = mirror;
  const nodes = JSON.parse(nodesJson);
  const seen = new Set();
  // Node bounds are the window's own; an overlay window's canvas sits somewhere on the page.
  const origin = mirror.canvas.getBoundingClientRect();
  for (const [id, node] of nodes) {
    seen.add(id);
    let el = elements.get(id);
    if (!el) {
      el = document.createElement('div');
      el.dataset.gpuiId = mirror.key === 1 ? id : `${mirror.key}:${id}`;
      el.dataset.gpuiNode = id;
      el.dataset.gpuiWindow = String(mirror.key);
      el.style.position = 'fixed';
      el.style.pointerEvents = 'none';
      el.style.outline = 'none';
      el.style.color = 'transparent';
      el.style.caretColor = 'transparent';
      el.addEventListener('click', onMirrorClick);
      el.addEventListener('wheel', onMirrorWheel, { passive: false });
      elements.set(id, el);
    }
    if (TEXT_ROLES.has(node.role) && !el.dataset.gpuiText) {
      el.dataset.gpuiText = '1';
      el.tabIndex = -1;
      el.contentEditable = 'plaintext-only';
      el.addEventListener('keydown', forwardKey);
      el.addEventListener('keyup', forwardKey);
      el.addEventListener('beforeinput', forwardInsertedText);
      el.addEventListener('input', forwardLeftoverText);
    }
    setAttr(el, 'role', ROLES[node.role] ?? 'group');
    setAttr(el, 'aria-label', node.label);
    setAttr(el, 'aria-description', node.description);
    setAttr(el, 'aria-expanded', node.expanded);
    setAttr(el, 'aria-selected', node.selected);
    setAttr(el, 'aria-checked', node.toggled);
    setAttr(el, 'aria-keyshortcuts', node.shortcut);
    setAttr(el, 'data-gpui-actions', node.actions && node.actions.length ? node.actions.join(' ') : undefined);
    if (node.focused) el.dataset.gpuiFocused = '1'; else delete el.dataset.gpuiFocused;
    if (node.bounds) {
      const [x0, y0, x1, y1] = node.bounds;
      const style = `left:${origin.left + x0 / dpr}px;top:${origin.top + y0 / dpr}px;width:${(x1 - x0) / dpr}px;height:${(y1 - y0) / dpr}px;`;
      // Fixed, not absolute: the elements nest like the tree, and an absolute child would be placed from its parent's corner instead of the page's.
      if (el.dataset.gpuiBox !== style) { el.dataset.gpuiBox = style; el.style.cssText = `position:fixed;pointer-events:none;outline:none;color:transparent;caret-color:transparent;${style}`; }
    }
  }
  for (const [id, el] of elements) {
    if (!seen.has(id)) { el.remove(); elements.delete(id); }
  }
  // Parent each element under its parent's element, in the tree's child order; nodes are fixed-positioned, so nesting only carries structure.
  for (const [id, node] of nodes) {
    const el = elements.get(id);
    const children = node.children ?? [];
    let previous = null;
    for (const childId of children) {
      const child = elements.get(childId);
      if (!child) continue;
      const expectedPrevious = previous;
      if (child.parentElement !== el || child.previousElementSibling !== expectedPrevious) {
        if (expectedPrevious) expectedPrevious.after(child); else el.prepend(child);
      }
      previous = child;
    }
  }
  // gpui lists the root last (it pops it off its stack after every child), so the root is the one node nobody lists as a child.
  const childIds = new Set();
  for (const [, node] of nodes) for (const childId of node.children ?? []) childIds.add(childId);
  for (const [id] of nodes) {
    if (childIds.has(id)) continue;
    const top = elements.get(id);
    if (top && top.parentElement !== root) root.appendChild(top);
  }
}
"#)]
extern "C" {
    fn gpui_a11y_start(
        dispatch: &Closure<dyn FnMut(String, String)>,
        canvas: &web_sys::HtmlCanvasElement,
        input: &web_sys::HtmlTextAreaElement,
    ) -> u32;
    fn gpui_a11y_update(window: u32, nodes_json: &str, focus: &str, dpr: f64);
    fn gpui_a11y_stop(window: u32);
}

/// One window's mirror; `window` is its key in the page's mirror table, 0 when the mirror is off.
pub(crate) struct WebA11y {
    callbacks: A11yCallbacks,
    _dispatch: Closure<dyn FnMut(String, String)>,
    window: u32,
}

impl Drop for WebA11y {
    fn drop(&mut self) {
        if self.window != 0 {
            gpui_a11y_stop(self.window);
        }
    }
}

impl WebA11y {
    pub(crate) fn start(
        callbacks: A11yCallbacks,
        canvas: &web_sys::HtmlCanvasElement,
        input: &web_sys::HtmlTextAreaElement,
    ) -> Rc<RefCell<Self>> {
        let this = Rc::new(RefCell::new(Self {
            callbacks,
            _dispatch: Closure::<dyn FnMut(String, String)>::new(|_, _| {}),
            window: 0,
        }));
        // Weak: the closure lives in the value it points at, and a strong handle would keep a closed window's mirror alive.
        let for_dispatch = Rc::downgrade(&this);
        let dispatch = Closure::<dyn FnMut(String, String)>::new(move |id: String, action: String| {
            let (Ok(id), Some(action)) = (id.parse::<u64>(), action_from_name(&action)) else {
                return;
            };
            let request = ActionRequest {
                action,
                target_tree: accesskit::TreeId::ROOT,
                target_node: NodeId(id),
                data: None,
            };
            // The callback may re-enter gpui, which may be mid-frame when a test calls in; deliver from the closure's own turn.
            if let Some(a11y) = for_dispatch.upgrade()
                && let Ok(a11y) = a11y.try_borrow()
            {
                (a11y.callbacks.action)(request);
            }
        });
        let window = gpui_a11y_start(&dispatch, canvas, input);
        {
            let mut a11y = this.borrow_mut();
            a11y._dispatch = dispatch;
            a11y.window = window;
            if window != 0 {
                // The page is the assistive client: the tree is wanted from the first frame.
                (a11y.callbacks.activation)();
            }
        }
        this
    }

    pub(crate) fn tree_update(&self, update: TreeUpdate, dpr: f64) {
        if self.window == 0 {
            return;
        }
        let mut json = String::from("[");
        for (index, (id, node)) in update.nodes.iter().enumerate() {
            if index > 0 {
                json.push(',');
            }
            json.push_str(&format!("[\"{}\",{{", id.0));
            json.push_str(&format!("\"role\":\"{:?}\"", node.role()));
            if let Some(label) = node.label() {
                json.push_str(&format!(",\"label\":{}", json_string(label)));
            }
            if let Some(description) = node.description() {
                json.push_str(&format!(",\"description\":{}", json_string(description)));
            }
            if let Some(shortcut) = node.keyboard_shortcut() {
                json.push_str(&format!(",\"shortcut\":{}", json_string(shortcut)));
            }
            if let Some(expanded) = node.is_expanded() {
                json.push_str(&format!(",\"expanded\":{expanded}"));
            }
            if node.is_selected().is_some() {
                json.push_str(&format!(",\"selected\":{}", node.is_selected().unwrap()));
            }
            if let Some(toggled) = node.toggled() {
                json.push_str(&format!(
                    ",\"toggled\":\"{}\"",
                    match toggled {
                        accesskit::Toggled::True => "true",
                        accesskit::Toggled::False => "false",
                        accesskit::Toggled::Mixed => "mixed",
                    }
                ));
            }
            if let Some(bounds) = node.bounds() {
                json.push_str(&format!(
                    ",\"bounds\":[{},{},{},{}]",
                    bounds.x0, bounds.y0, bounds.x1, bounds.y1
                ));
            }
            if *id == update.focus {
                json.push_str(",\"focused\":true");
            }
            let actions: Vec<String> = all_actions()
                .filter(|action| node.supports_action(*action))
                .map(|action| format!("{action:?}"))
                .collect();
            if !actions.is_empty() {
                json.push_str(&format!(",\"actions\":{}", json_string_array(&actions)));
            }
            let children: Vec<String> = node.children().iter().map(|child| child.0.to_string()).collect();
            if !children.is_empty() {
                json.push_str(&format!(",\"children\":{}", json_string_array(&children)));
            }
            json.push_str("}]");
        }
        json.push(']');
        gpui_a11y_update(self.window, &json, &update.focus.0.to_string(), dpr);
    }
}

/// Every AccessKit action, through the `enumn` constructor the crate derives for it.
fn all_actions() -> impl Iterator<Item = accesskit::Action> {
    (0..64).filter_map(accesskit::Action::n)
}

fn action_from_name(name: &str) -> Option<accesskit::Action> {
    all_actions().find(|action| format!("{action:?}") == name)
}

fn json_string(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn json_string_array(items: &[String]) -> String {
    let mut out = String::from("[");
    for (index, item) in items.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&json_string(item));
    }
    out.push(']');
    out
}
