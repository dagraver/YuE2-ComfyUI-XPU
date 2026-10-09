import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

export function installStyle(id, css) {
    if (document.getElementById(id)) return;
    const style = document.createElement("style");
    style.id = id;
    style.textContent = css;
    document.head.appendChild(style);
}

export function element(tag, className, text) {
    const made = document.createElement(tag);
    if (className) made.className = className;
    if (text !== undefined) made.textContent = text;
    return made;
}

export async function ask(url, body) {
    const response = await api.fetchApi(url, {
        method: body ? "POST" : "GET",
        headers: body ? { "Content-Type": "application/json" } : undefined,
        body: body ? JSON.stringify(body) : undefined,
    });
    let payload = {};
    try {
        payload = await response.json();
    } catch (error) {
        payload = {};
    }
    return { ok: response.ok, status: response.status, payload };
}

const BASE_STYLE_ID = "yue2-controls-style";
const BASE_STYLE = `
.yue2-buttons { display: flex; width: 100%; height: 100%; gap: 6px;
    font-family: system-ui, sans-serif; }
.yue2-buttons button { flex: 1 1 0; min-width: 0; font: inherit; font-size: 12px;
    padding: 0 8px; border-radius: 6px; cursor: pointer;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    color: var(--input-text, #ddd); background: var(--comfy-input-bg, #2b2b2b);
    border: 1px solid var(--border-color, #4e4e4e); }
.yue2-buttons button:hover { background: var(--comfy-menu-bg, #353535); }
.yue2-buttons button:active { background: #3B7DD8; border-color: #3B7DD8; color: #fff; }
.yue2-buttons button[disabled] { opacity: 0.5; cursor: not-allowed; }
`;

export const BUTTON_H = 24;
const BUTTON_MARGIN = 4;

export function buttonRow(node, name, buttons) {
    installStyle(BASE_STYLE_ID, BASE_STYLE);

    const row = document.createElement("div");
    row.className = "yue2-buttons";
    const made = buttons.map((spec) => {
        const button = document.createElement("button");
        button.textContent = spec.label;
        if (spec.tooltip) button.title = spec.tooltip;
        button.addEventListener("click", () => spec.onClick(button));
        row.appendChild(button);
        return button;
    });

    const widget = node.addDOMWidget(name, "yue2_buttons", row, {
        hideOnZoom: false,
        margin: BUTTON_MARGIN,
        hideInPanel: true,
        getValue: () => undefined,
        setValue: () => {},
        getMinHeight: () => BUTTON_H + 2 * BUTTON_MARGIN,
        getMaxHeight: () => BUTTON_H + 2 * BUTTON_MARGIN,
    });
    widget.serialize = false;
    widget.serializeValue = () => undefined;
    return { widget, buttons: made };
}

export function panelWidget(node, name, element, height, grow = false) {
    const widget = node.addDOMWidget(name, "yue2_panel", element, {
        hideOnZoom: false,
        margin: BUTTON_MARGIN,
        hideInPanel: true,
        getValue: () => undefined,
        setValue: () => {},
        getMinHeight: () => height() + 2 * BUTTON_MARGIN,
        getMaxHeight: () => (grow ? undefined : height() + 2 * BUTTON_MARGIN),
    });
    widget.serialize = false;
    widget.serializeValue = () => undefined;
    return widget;
}

export function widgetNamed(node, name) {
    return node?.widgets?.find((w) => w.name === name);
}

export function graphChanged() {
    try {
        const tracker = app.extensionManager?.workflow?.activeWorkflow?.changeTracker;
        if (tracker?.captureCanvasState) tracker.captureCanvasState();
        else tracker?.checkState?.();
    } catch (error) {
        console.error("[YuE2] the workflow was not told that its graph changed:", error);
    }
}

export function setWidgetValue(node, name, value) {
    const widget = widgetNamed(node, name);
    if (!widget || widget.value === value) return;
    widget.value = value;
    widget.callback?.(value, app.canvas, node);
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
    graphChanged();
}

export function showWidget(node, name, shown) {
    const widget = widgetNamed(node, name);
    if (!widget) return;
    widget.hidden = !shown;
    widget.options = widget.options || {};
    widget.options.hidden = !shown;
    if (shown) delete widget.computeSize;
    else widget.computeSize = () => [0, -4];
}

export function sourceOf(node, name) {
    const link = node.inputs?.find((input) => input.name === name)?.link;
    if (link === null || link === undefined) return null;
    const record = app.graph?.links?.get?.(link) ?? app.graph?.links?.[link];
    return record ? app.graph.getNodeById(record.origin_id) : null;
}

const WINDOW_STYLE_ID = "yue2-window-style";
const WINDOW_STYLE = `
.yue2-back { position: fixed; inset: 0; z-index: 1300; display: flex;
    align-items: center; justify-content: center; background: rgba(0, 0, 0, 0.55);
    font-family: system-ui, sans-serif; }
.yue2-panel { width: min(920px, 95vw); height: calc(100vh - 32px); display: flex;
    flex-direction: column; overflow: hidden;
    background: var(--comfy-menu-bg, #353535); color: var(--input-text, #ddd);
    border: 1px solid var(--border-color, #4e4e4e); border-radius: 10px;
    padding: 16px 20px; box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5);
    box-sizing: border-box; }
.yue2-panel:focus { outline: none; }
`;

const ASKED_STYLE_ID = "yue2-asked-style";
const ASKED_STYLE = `
.yue2-panel.yue2-asked { width: min(460px, 92vw); height: auto; max-height: calc(100vh - 64px);
    gap: 10px; padding: 18px 20px; }
.yue2-asked-title { font-size: 14px; font-weight: 600; }
.yue2-asked-text { font-size: 13px; line-height: 1.5; white-space: pre-line; overflow: auto;
    color: var(--descrip-text, #bbb); }
.yue2-asked-row { display: flex; gap: 8px; justify-content: flex-end; margin-top: 4px; }
.yue2-asked button { font: inherit; font-size: 12px; padding: 5px 14px; border-radius: 6px;
    cursor: pointer; color: var(--input-text, #ddd); background: var(--comfy-input-bg, #2b2b2b);
    border: 1px solid var(--border-color, #4e4e4e); }
.yue2-asked button:hover { background: var(--comfy-menu-bg, #353535); }
.yue2-asked button.yue2-asked-go { background: #3B7DD8; border-color: #3B7DD8; color: #fff; }
.yue2-asked button.yue2-asked-danger { background: #B4433A; border-color: #B4433A; color: #fff; }
`;

export function confirmed(text, { title = "", ok = "OK", cancel = "Cancel",
                                  danger = false } = {}) {
    installStyle(ASKED_STYLE_ID, ASKED_STYLE);
    return new Promise((resolve) => {
        let answer = false;
        const made = frame({ onClose: () => resolve(answer) });
        made.panel.classList.add("yue2-asked");
        if (title) made.panel.appendChild(element("div", "yue2-asked-title", title));
        made.panel.appendChild(element("div", "yue2-asked-text", String(text)));
        const row = element("div", "yue2-asked-row");
        const no = element("button", "", cancel);
        const yes = element("button", "yue2-asked-go" + (danger ? " yue2-asked-danger" : ""), ok);
        no.addEventListener("click", () => made.close());
        yes.addEventListener("click", () => {
            answer = true;
            made.close();
        });
        row.appendChild(no);
        row.appendChild(yes);
        made.panel.appendChild(row);
        yes.focus();
    });
}

export function warned(text, { title = "", ok = "OK" } = {}) {
    installStyle(ASKED_STYLE_ID, ASKED_STYLE);
    return new Promise((resolve) => {
        const made = frame({ onClose: () => resolve() });
        made.panel.classList.add("yue2-asked");
        made.panel.appendChild(element("div", "yue2-asked-title", title || "That did not work"));
        made.panel.appendChild(element("div", "yue2-asked-text", String(text)));
        const row = element("div", "yue2-asked-row");
        const yes = element("button", "yue2-asked-go", ok);
        yes.addEventListener("click", () => made.close());
        row.appendChild(yes);
        made.panel.appendChild(row);
        yes.focus();
    });
}

const TEXT_TYPES = new Set(["text", "search", "number", "url", "email", "tel"]);

export function textMenu(event) {
    const target = event.target;
    if (target instanceof HTMLTextAreaElement) return true;
    if (target instanceof HTMLInputElement) return TEXT_TYPES.has(target.type);
    if (target?.isContentEditable) return true;
    if (target?.closest?.("button")) return false;
    const picked = window.getSelection();
    if (!picked || picked.isCollapsed) return false;
    for (let at = 0; at < picked.rangeCount; at += 1) {
        for (const box of picked.getRangeAt(at).getClientRects()) {
            if (event.clientX >= box.left && event.clientX <= box.right
                && event.clientY >= box.top && event.clientY <= box.bottom) return true;
        }
    }
    return false;
}

const OPEN = [];

export function onTop(handle) {
    return OPEN[OPEN.length - 1] === handle;
}

const SIZED_STYLE_ID = "yue2-sized-style";
const SIZED_STYLE = `
.yue2-panel.yue2-sized { position: relative; max-width: calc(100vw - 16px);
    max-height: calc(100vh - 16px); overflow: auto; }
.yue2-panel.yue2-sized.yue2-max { width: calc(100vw - 16px) !important;
    height: calc(100vh - 16px) !important; }
.yue2-panel.yue2-sized > h3:first-of-type { padding-right: 34px; }
.yue2-grip { position: absolute; z-index: 5; }
.yue2-grip-n, .yue2-grip-s { left: 14px; right: 14px; height: 7px; cursor: ns-resize; }
.yue2-grip-e, .yue2-grip-w { top: 14px; bottom: 14px; width: 7px; cursor: ew-resize; }
.yue2-grip-n { top: 0; }
.yue2-grip-s { bottom: 0; }
.yue2-grip-e { right: 0; }
.yue2-grip-w { left: 0; }
.yue2-grip-ne, .yue2-grip-nw, .yue2-grip-se, .yue2-grip-sw { width: 14px; height: 14px; }
.yue2-grip-ne { top: 0; right: 0; cursor: nesw-resize; }
.yue2-grip-sw { bottom: 0; left: 0; cursor: nesw-resize; }
.yue2-grip-nw { top: 0; left: 0; cursor: nwse-resize; }
.yue2-grip-se { bottom: 0; right: 0; cursor: nwse-resize; }
.yue2-panel.yue2-max > .yue2-grip { display: none; }
.yue2-panel.yue2-sized > button.yue2-win-max { position: absolute; top: 9px; right: 10px; z-index: 6;
    width: 28px; height: 24px; padding: 0; display: flex; align-items: center;
    justify-content: center; border-radius: 6px; cursor: pointer;
    color: var(--descrip-text, #aaa); background: transparent; border: 1px solid transparent; }
.yue2-panel.yue2-sized > button.yue2-win-max:hover { color: var(--input-text, #ddd);
    background: var(--comfy-input-bg, #2b2b2b); border-color: var(--border-color, #4e4e4e); }
.yue2-win-max svg { width: 13px; height: 13px; display: block; }
`;

const GRIPS = ["n", "s", "e", "w", "ne", "nw", "se", "sw"];
const LEAST_W = 420;
const LEAST_H = 260;
const EDGE = 16;
const FILL_ICON = '<svg viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.6">'
    + '<rect x="1.5" y="1.5" width="11" height="11" rx="1.5"/></svg>';
const BACK_ICON = '<svg viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.5">'
    + '<rect x="1.5" y="4.5" width="8" height="8" rx="1.2"/>'
    + '<path d="M4.5 4.5V2.7c0-.7.5-1.2 1.2-1.2h5.6c.7 0 1.2.5 1.2 1.2v5.6c0 .7-.5 1.2-1.2 1.2H9.5"/></svg>';

function sizeKey(kind) {
    return "yue2.window." + kind;
}

function rememberedSize(kind) {
    try {
        const kept = JSON.parse(window.localStorage.getItem(sizeKey(kind)) || "null");
        if (kept && typeof kept === "object") return kept;
    } catch (error) {
        void error;
    }
    return {};
}

function rememberSize(kind, size) {
    try {
        window.localStorage.setItem(sizeKey(kind), JSON.stringify(size));
    } catch (error) {
        void error;
    }
}

function within(value, least, most) {
    return Math.max(least, Math.min(most, value));
}

function sizable(panel, kind) {
    installStyle(SIZED_STYLE_ID, SIZED_STYLE);
    panel.classList.add("yue2-sized");
    const kept = rememberedSize(kind);
    if (kept.w > 0) panel.style.width = Math.round(kept.w) + "px";
    if (kept.h > 0) panel.style.height = Math.round(kept.h) + "px";

    const fill = element("button", "yue2-win-max");
    fill.type = "button";
    const hold = () => panel.classList.toggle("yue2-held",
        Boolean(panel.style.height) || panel.classList.contains("yue2-max"));
    const show = (filled) => {
        panel.classList.toggle("yue2-max", filled);
        hold();
        fill.innerHTML = filled ? BACK_ICON : FILL_ICON;
        fill.title = filled ? "Back to the window's own size" : "Fill the browser tab";
        fill.setAttribute("aria-label", fill.title);
    };
    const save = () => rememberSize(kind, {
        w: parseFloat(panel.style.width) || 0,
        h: parseFloat(panel.style.height) || 0,
        max: panel.classList.contains("yue2-max"),
    });
    show(Boolean(kept.max));
    fill.addEventListener("click", (event) => {
        event.stopPropagation();
        show(!panel.classList.contains("yue2-max"));
        save();
    });
    fill.addEventListener("mousedown", (event) => event.preventDefault());
    fill.addEventListener("dblclick", (event) => event.stopPropagation());

    for (const side of GRIPS) {
        const grip = element("div", "yue2-grip yue2-grip-" + side);
        grip.addEventListener("pointerdown", (event) => drag(event, grip, side));
        panel.appendChild(grip);
    }
    panel.appendChild(fill);

    function overflowing() {
        return panel.scrollHeight - panel.clientHeight > 0;
    }

    function fit() {
        const over = panel.scrollHeight - panel.clientHeight;
        if (over <= 0) return;
        panel.style.height = Math.round(Math.min(panel.getBoundingClientRect().height + over,
            window.innerHeight - EDGE)) + "px";
    }

    function drag(event, grip, side) {
        if (event.button !== 0) return;
        event.preventDefault();
        event.stopPropagation();
        const box = panel.getBoundingClientRect();
        const x0 = event.clientX;
        const y0 = event.clientY;
        const across = side.includes("e") ? 2 : side.includes("w") ? -2 : 0;
        const down = side.includes("s") ? 2 : side.includes("n") ? -2 : 0;
        try {
            grip.setPointerCapture(event.pointerId);
        } catch (error) {
            void error;
        }
        const move = (moved) => {
            const was = { width: panel.style.width, height: panel.style.height, over: overflowing() };
            if (across) {
                panel.style.width = Math.round(within(box.width + across * (moved.clientX - x0),
                    LEAST_W, window.innerWidth - EDGE)) + "px";
            }
            if (down) {
                panel.style.height = Math.round(within(box.height + down * (moved.clientY - y0),
                    LEAST_H, window.innerHeight - EDGE)) + "px";
            }
            hold();
            fit();
            if (!was.over && overflowing()) {
                panel.style.width = was.width;
                panel.style.height = was.height;
            }
            hold();
        };
        const end = () => {
            grip.removeEventListener("pointermove", move);
            grip.removeEventListener("pointerup", end);
            grip.removeEventListener("pointercancel", end);
            grip.removeEventListener("lostpointercapture", end);
            save();
        };
        grip.addEventListener("pointermove", move);
        grip.addEventListener("pointerup", end);
        grip.addEventListener("pointercancel", end);
        grip.addEventListener("lostpointercapture", end);
    }
}

export function frame({ onClose, sticky, size } = {}) {
    installStyle(WINDOW_STYLE_ID, WINDOW_STYLE);
    const back = element("div", "yue2-back");
    const panel = element("div", "yue2-panel");
    panel.tabIndex = -1;
    panel.setAttribute("role", "dialog");
    panel.setAttribute("aria-modal", "true");
    if (size) sizable(panel, size);
    back.appendChild(panel);
    const me = { close };
    OPEN.push(me);

    function close() {
        const at = OPEN.indexOf(me);
        if (at >= 0) OPEN.splice(at, 1);
        document.removeEventListener("keydown", onKey, true);
        back.remove();
        onClose?.();
    }

    function onKey(event) {
        if (event.key !== "Escape" || OPEN[OPEN.length - 1] !== me) return;
        event.stopPropagation();
        me.onEscape ? me.onEscape() : close();
    }

    panel.addEventListener("keydown", (event) => event.stopPropagation());
    document.addEventListener("keydown", onKey, true);
    back.addEventListener("pointerdown", (event) => {
        if (event.target === back && !sticky) close();
    });
    document.body.appendChild(back);
    return { back, panel, close, handle: me };
}
