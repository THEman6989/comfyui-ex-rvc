import { test, expect } from "bun:test";
import { readFileSync } from "node:fs";

class Element {
  constructor(tagName) {
    this.tagName = tagName.toUpperCase();
    this.value = "";
    this.selectionStart = 0;
    this.selectionEnd = 0;
    this.children = [];
    this.handlers = {};
    this.style = {};
    this.clientWidth = 420;
    this.clientHeight = 180;
    this.scrollHeight = 900;
    this.scrollTop = 0;
  }
  setAttribute() {}
  addEventListener(name, callback) { this.handlers[name] = callback; }
  append(...children) { this.children.push(...children); }
  remove() {}
  focus() {}
  setSelectionRange(start, end) { this.selectionStart = start; this.selectionEnd = end; }
  trigger(name, extra = {}) {
    this.handlers[name]?.({ key: "", shiftKey: false, preventDefault() {}, stopPropagation() {}, ...extra });
  }
}

const source = readFileSync(new URL("../web/searchable_text.js", import.meta.url), "utf8")
  .replace('import { app } from "/scripts/app.js";', "")
  .replace("export function attachTextSearch", "function attachTextSearch");
const app = { registerExtension(extension) { this.extension = extension; } };
const document = {
  createElement(tag) { return new Element(tag); },
  createTextNode(text) { return { textContent: text }; },
  body: { append(element) { element.children.forEach(child => child.parent = element); } },
};
Object.defineProperty(Element.prototype, "offsetTop", {
  get() { return (this.parent?.children[0]?.textContent?.split("\n").length ?? 1) * 24; },
});
const getComputedStyle = () => ({
  font: "12px monospace", lineHeight: "24px", letterSpacing: "normal",
  padding: "2px", tabSize: "8", wordBreak: "normal",
});
new Function("app", "document", "getComputedStyle", source)(app, document, getComputedStyle);

function setup(text) {
  const textarea = new Element("textarea");
  textarea.value = text;
  const textWidget = { name: "text", element: textarea, value: text, options: { minNodeSize: [400, 200] } };
  const node = {
    type: "SearchableMultilineText",
    widgets: [textWidget],
    size: [195, 145],
    addDOMWidget(name, type, element, options) {
      const widget = { name, element, options };
      this.widgets.push(widget);
      return widget;
    },
    computeSize() { return [400, 200]; },
    setSize(size) { this.size = size; this.onResize?.(size); },
  };
  app.extension.nodeCreated(node);
  const [query, search, previous, next, status] = node.widgets[0].element.children;
  return { textarea, textWidget, node, query, search, previous, next, status };
}

test("finds literal text by Enter, wraps with arrows, never changes output widget", () => {
  const { textarea, textWidget, node, query, previous, next, status } = setup("a.c\nA.C\nother");
  expect(node.widgets.map(w => w.name)).toEqual(["search_bar", "text"]);
  expect(node.widgets[0].options.serialize).toBe(false);
  query.value = "a.c";
  query.trigger("input");
  expect(status.textContent).toBe("0/2");
  query.trigger("keydown", { key: "Enter" });
  expect([textarea.selectionStart, textarea.selectionEnd, status.textContent]).toEqual([0, 3, "1/2"]);
  next.trigger("click");
  expect([textarea.selectionStart, textarea.selectionEnd, status.textContent]).toEqual([4, 7, "2/2"]);
  next.trigger("click");
  expect(textarea.selectionStart).toBe(0);
  previous.trigger("click");
  expect(textarea.selectionStart).toBe(4);
  expect(textWidget.value).toBe("a.c\nA.C\nother");
});

test("search button, no results and query changes", () => {
  const { textarea, query, search, status } = setup("one\nTwo one");
  query.value = "missing";
  query.trigger("input");
  search.trigger("click");
  expect(status.textContent).toBe("0/0");
  query.value = "one";
  query.trigger("input");
  search.trigger("click");
  expect(textarea.selectionStart).toBe(0);
  expect(status.textContent).toBe("1/2");
  query.trigger("keydown", { key: "Enter", shiftKey: true });
  expect(textarea.selectionStart).toBe(8);
});

test("toolbar stays fixed and the native textarea gets the remaining resized height", () => {
  const { node, textWidget } = setup("text");
  expect(node.size[0]).toBeGreaterThanOrEqual(440);
  expect(node.size[1]).toBeGreaterThanOrEqual(320);
  expect(node.widgets[0].options.getMinHeight()).toBe(38);
  expect(node.widgets[0].options.getMaxHeight()).toBe(38);
  expect(textWidget.options.getMinHeight()).toBeGreaterThanOrEqual(150);
  expect(textWidget.options.minNodeSize[0]).toBeGreaterThanOrEqual(430);
  expect(node.widgets[0].element.style.width).toBe("440px");
  node.setSize([630, 650]);
  expect(node.widgets[0].element.style.width).toBe("610px");
  expect(node.widgets[0].options.getMaxHeight()).toBe(38);
  node.onConfigure({});
  expect(node.size).toEqual([630, 650]);
  node.setSize([195, 145]);
  node.onConfigure({});
  expect(node.size).toEqual([460, 340]);
  expect(node.widgets[0].element.style.width).toBe("440px");
});

test("Enter and arrows scroll to distant matches instead of only selecting text", () => {
  const { textarea, query, previous } = setup("first\n".repeat(60) + "TARGET\nlast");
  query.value = "TARGET";
  query.trigger("keydown", { key: "Enter" });
  expect(textarea.selectionStart).toBeGreaterThan(100);
  expect(textarea.scrollTop).toBeGreaterThan(0);
  previous.trigger("click");
  expect(textarea.scrollTop).toBeGreaterThan(0);
});

test("first Enter starts at the first match even when the caret was near the end", () => {
  const { textarea, query, next } = setup("FOUND\nintermediate\nFOUND\nmore\nFOUND");
  const nearEnd = textarea.value.lastIndexOf("FOUND");
  textarea.setSelectionRange(nearEnd, nearEnd);
  query.value = "FOUND";
  query.trigger("keydown", { key: "Enter" });
  expect(textarea.selectionStart).toBe(0);
  next.trigger("click");
  expect(textarea.selectionStart).toBe(19);
});
