import { app } from "/scripts/app.js";

// UI-only search: the ComfyUI STRING widget remains the sole owner of the text.
export function attachTextSearch(node) {
  const textWidget = node.widgets?.find((widget) => widget.name === "text");
  const element = textWidget?.element ?? textWidget?.inputEl;
  const textarea = element?.tagName === "TEXTAREA" ? element : element?.querySelector?.("textarea");
  if (!textarea || !node.addDOMWidget || node.widgets.some((widget) => widget.name === "search_bar")) return;

  const row = document.createElement("div");
  row.style.cssText = "display:flex;align-items:center;gap:4px;padding:3px 4px;box-sizing:border-box;width:100%;height:38px;color:var(--input-text,#ddd);";
  const query = document.createElement("input");
  query.type = "search";
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
      const caret = direction > 0 ? textarea.selectionEnd : textarea.selectionStart;
      current = direction > 0
        ? matches.findIndex(([start]) => start >= caret)
        : matches.findLastIndex(([, end]) => end <= caret);
      if (current < 0) current = direction > 0 ? 0 : matches.length - 1;
    } else {
      current = (current + direction + matches.length) % matches.length;
    }
    const [start, end] = matches[current];
    textarea.focus();
    textarea.setSelectionRange(start, end);
    status.textContent = `${current + 1}/${matches.length}`;
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
  });
  // Display the toolbar before the native textarea, without replacing or serializing it.
  const from = node.widgets.indexOf(toolbar);
  const to = node.widgets.indexOf(textWidget);
  if (from >= 0 && to >= 0) {
    node.widgets.splice(from, 1);
    node.widgets.splice(to, 0, toolbar);
  }
  node.setSize?.(node.computeSize());
}

app.registerExtension({
  name: "amin.searchableMultilineText",
  nodeCreated(node) {
    if (node.comfyClass === "SearchableMultilineText" || node.type === "SearchableMultilineText") {
      attachTextSearch(node);
    }
  },
});
