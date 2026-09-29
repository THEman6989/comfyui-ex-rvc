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
  }
  setAttribute() {}
  addEventListener(name, callback) { this.handlers[name] = callback; }
  append(...children) { this.children.push(...children); }
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
const document = { createElement(tag) { return new Element(tag); } };
new Function("app", "document", source)(app, document);

function setup(text) {
  const textarea = new Element("textarea");
  textarea.value = text;
  const textWidget = { name: "text", element: textarea, value: text };
  const node = {
    type: "SearchableMultilineText",
    widgets: [textWidget],
    addDOMWidget(name, type, element, options) {
      const widget = { name, element, options };
      this.widgets.push(widget);
      return widget;
    },
    computeSize() { return [400, 200]; },
    setSize() {},
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
