import { app } from "/scripts/app.js";

// UI-only search: the ComfyUI STRING widget remains the sole owner of the text.
export function attachTextSearch(node) {
  const textWidget = node.widgets?.find((widget) => widget.name === "text");
  const element = textWidget?.element ?? textWidget?.inputEl;
  const textarea = element?.tagName === "TEXTAREA" ? element : element?.querySelector?.("textarea");
  if (!textarea || !node.addDOMWidget || node.widgets.some((widget) => widget.name === "search_bar")) return;

  // LiteGraph divides surplus height among growable DOM widgets. Only the text
  // editor should grow; the search row must have both a fixed min and max.
  textWidget.options ??= {};
  textWidget.options.minNodeSize = [440, 260];
  textWidget.options.getMinHeight = () => 170;
  textWidget.options.getMaxHeight = () => Infinity;
  const ensureMinimumSize = () => {
    const width = Math.max(node.size?.[0] ?? 0, 460);
    const height = Math.max(node.size?.[1] ?? 0, 340);
    if (node.size?.[0] !== width || node.size?.[1] !== height) node.setSize?.([width, height]);
  };
  ensureMinimumSize();

  const row = document.createElement("div");
  row.style.cssText = "display:flex;align-items:center;gap:4px;padding:3px 4px;box-sizing:border-box;min-width:0;height:38px;overflow:hidden;color:var(--input-text,#ddd);";
  // ComfyUI can shrink a DOM widget's overlay to its intrinsic width after
  // focus/selection. Bind the search row to node width, not overlay width.
  const updateToolbarWidth = () => { row.style.width = `${Math.max(0, node.size[0] - 20)}px`; };
  const originalResize = node.onResize;
  node.onResize = function (...args) {
    originalResize?.apply(this, args);
    updateToolbarWidth();
  };
  const originalConfigure = node.onConfigure;
  node.onConfigure = function (...args) {
    originalConfigure?.apply(this, args);
    ensureMinimumSize();
    updateToolbarWidth();
  };
  updateToolbarWidth();
  const query = document.createElement("input");
  query.type = "text";
  query.placeholder = "Im Text suchen";
  query.setAttribute("aria-label", "Im Text suchen");
  query.style.cssText = "min-width:0;flex:1;background:var(--comfy-input-bg,#222);color:inherit;border:1px solid #666;border-radius:4px;padding:4px;";
  const button = (label, title, action) => {
    const el = document.createElement("button");
    el.type = "button";
    el.textContent = label;
    el.title = title;
    el.setAttribute("aria-label", title);
    el.style.cssText = "cursor:pointer;padding:4px 6px;flex-shrink:0;";
    el.addEventListener("click", action);
    return el;
  };
  const status = document.createElement("span");
  status.setAttribute("aria-live", "polite");
  status.style.cssText = "font-size:11px;white-space:nowrap;";
  let matches = [];
  let current = -1;
  let previousQuery = null;
  let previousText = null;

  function refresh() {
    const term = query.value;
    const text = textarea.value;
    if (term === previousQuery && text === previousText) return;
    previousQuery = term;
    previousText = text;
    current = -1;
    matches = [];
    if (term) {
      const escaped = term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      const regex = new RegExp(escaped, "gi");
      for (const match of text.matchAll(regex)) matches.push([match.index, match.index + match[0].length]);
    }
    status.textContent = term ? `0/${matches.length}` : "";
  }

  function navigate(direction) {
    refresh();
    if (!matches.length) return;
    if (current < 0) {
      current = direction > 0 ? 0 : matches.length - 1;
    } else {
      current = (current + direction + matches.length) % matches.length;
    }
    const [start, end] = matches[current];
    textarea.focus();
    textarea.setSelectionRange(start, end);
    scrollMatchIntoView(start);
    status.textContent = `${current + 1}/${matches.length}`;
  }

  function scrollMatchIntoView(start) {
    if (textarea.scrollHeight <= textarea.clientHeight) return;
    // A textarea has no DOM child for its caret. Mirror the text up to the
    // match with identical wrapping to obtain its real vertical position.
    const style = getComputedStyle(textarea);
    const mirror = document.createElement("div");
    mirror.style.cssText = "position:absolute;left:-10000px;top:0;visibility:hidden;white-space:pre-wrap;overflow-wrap:break-word;box-sizing:border-box;border:0;";
    mirror.style.width = `${textarea.clientWidth}px`;
    mirror.style.font = style.font;
    mirror.style.lineHeight = style.lineHeight;
    mirror.style.letterSpacing = style.letterSpacing;
    mirror.style.padding = style.padding;
    mirror.style.tabSize = style.tabSize;
    mirror.style.wordBreak = style.wordBreak;
    const marker = document.createElement("span");
    marker.textContent = "\u200b";
    mirror.append(document.createTextNode(textarea.value.slice(0, start)), marker);
    document.body.append(mirror);
    textarea.scrollTop = Math.max(0, marker.offsetTop - textarea.clientHeight / 2);
    mirror.remove();
  }

  query.addEventListener("input", refresh);
  textarea.addEventListener("input", refresh);
  query.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      event.stopPropagation();
      navigate(event.shiftKey ? -1 : 1);
    }
  });
  row.append(
    query,
    button("Suchen", "Suchen", () => navigate(1)),
    button("↑", "Vorheriger Treffer", () => navigate(-1)),
    button("↓", "Nächster Treffer", () => navigate(1)),
    status,
  );
  const toolbar = node.addDOMWidget("search_bar", "search_bar", row, {
    serialize: false,
    hideOnZoom: false,
    getMinHeight: () => 38,
    getMaxHeight: () => 38,
  });
  // Display the toolbar before the native textarea, without replacing or serializing it.
  const from = node.widgets.indexOf(toolbar);
  const to = node.widgets.indexOf(textWidget);
  if (from >= 0 && to >= 0) {
    node.widgets.splice(from, 1);
    node.widgets.splice(to, 0, toolbar);
  }
  ensureMinimumSize();
}

app.registerExtension({
  name: "amin.searchableMultilineText",
  nodeCreated(node) {
    if (node.comfyClass === "SearchableMultilineText" || node.type === "SearchableMultilineText") {
      attachTextSearch(node);
    }
  },
});
