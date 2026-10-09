"""Visual design system for Qwen3-TTS Studio.

One place for everything that controls how the app looks:

* ``LIGHT`` / ``DARK``  - colour tokens (single source of truth)
* ``build_theme()``     - the Gradio theme, fed from those tokens
* ``APP_CSS``           - layout + component styling, written against the tokens
* ``APP_JS``            - tiny page-load script (settings panel toggle)
* ``HEADER_HTML``       - the app bar

Dark mode is automatic: Gradio adds a ``dark`` class to <body> (from the OS
preference or ``?__theme=dark``) and the tokens below are redefined under it.
"""

import inspect

import gradio as gr

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

LIGHT = {
    "bg": "#f4f5fa",
    "surface": "#ffffff",
    "surface-2": "#f8f9fc",
    "surface-3": "#eef0f6",
    "line": "#e3e6ee",
    "line-strong": "#cdd2e0",
    "text": "#161a2c",
    "text-2": "#454c63",
    "text-3": "#79809a",
    "accent": "#5b5ff6",
    "accent-hover": "#4a4ee6",
    "accent-soft": "rgba(91, 95, 246, 0.10)",
    "accent-text": "#4347d6",
    "success": "#15803d",
    "success-soft": "rgba(22, 163, 74, 0.10)",
    "danger": "#c81e1e",
    "danger-soft": "rgba(220, 38, 38, 0.08)",
    "warn": "#a16207",
    "warn-soft": "rgba(202, 138, 4, 0.12)",
    "shadow-sm": "0 1px 2px rgba(22, 26, 44, 0.05), 0 1px 3px rgba(22, 26, 44, 0.04)",
    "shadow-md": "0 4px 14px -4px rgba(22, 26, 44, 0.12), 0 2px 6px rgba(22, 26, 44, 0.05)",
    "glow": "rgba(91, 95, 246, 0.45)",
}

DARK = {
    "bg": "#0e1016",
    "surface": "#161922",
    "surface-2": "#1b1f2b",
    "surface-3": "#232837",
    "line": "#272c3b",
    "line-strong": "#394058",
    "text": "#e8eaf2",
    "text-2": "#b2b8cc",
    "text-3": "#818aa3",
    "accent": "#7479f8",
    "accent-hover": "#8b8ffa",
    "accent-soft": "rgba(116, 121, 248, 0.16)",
    "accent-text": "#aeb1fc",
    "success": "#4ade80",
    "success-soft": "rgba(74, 222, 128, 0.12)",
    "danger": "#f87171",
    "danger-soft": "rgba(248, 113, 113, 0.12)",
    "warn": "#fbbf24",
    "warn-soft": "rgba(251, 191, 36, 0.12)",
    "shadow-sm": "0 1px 2px rgba(0, 0, 0, 0.35)",
    "shadow-md": "0 6px 18px -6px rgba(0, 0, 0, 0.55)",
    "glow": "rgba(116, 121, 248, 0.40)",
}


def _token_css() -> str:
    def block(tokens: dict[str, str]) -> str:
        return "\n".join(f"    --ui-{k}: {v};" for k, v in tokens.items())

    return f":root {{\n{block(LIGHT)}\n}}\nbody.dark, .dark {{\n{block(DARK)}\n}}\n"


# ---------------------------------------------------------------------------
# Gradio theme
# ---------------------------------------------------------------------------


def build_theme() -> gr.themes.Base:
    """Gradio theme wired to the colour tokens above (light + dark)."""
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.violet,
        neutral_hue=gr.themes.colors.slate,
        text_size=gr.themes.Size(
            xxs="10.5px", xs="11.5px", sm="13px", md="14.5px", lg="16px",
            xl="19px", xxl="24px",
        ),
        radius_size=gr.themes.Size(
            xxs="3px", xs="5px", sm="8px", md="10px", lg="12px", xl="16px",
            xxl="22px",
        ),
        font=[
            gr.themes.GoogleFont("Inter"),
            "ui-sans-serif",
            "system-ui",
            "Segoe UI",
            "sans-serif",
        ],
        font_mono=[
            gr.themes.GoogleFont("JetBrains Mono"),
            "ui-monospace",
            "Consolas",
            "monospace",
        ],
    )

    settable = set(inspect.signature(gr.themes.Base.set).parameters)
    values: dict[str, str] = {}

    def put(name: str, light: str, dark: str | None = None) -> None:
        values[name] = light
        if f"{name}_dark" in settable:
            values[f"{name}_dark"] = light if dark is None else dark

    def both(name: str, token: str) -> None:
        put(name, LIGHT[token], DARK[token])

    # Page + text
    both("body_background_fill", "bg")
    both("body_text_color", "text")
    both("body_text_color_subdued", "text-3")
    both("background_fill_primary", "surface")
    both("background_fill_secondary", "surface-2")
    both("border_color_primary", "line")
    both("border_color_accent", "accent")
    both("color_accent", "accent")
    both("color_accent_soft", "accent-soft")
    both("link_text_color", "accent-text")
    both("link_text_color_hover", "accent-hover")
    both("link_text_color_active", "accent-hover")
    both("link_text_color_visited", "accent-text")
    put("body_text_weight", "400")

    # Blocks are flat: the surrounding card provides the container.
    put("block_background_fill", "transparent")
    put("block_border_width", "0px")
    put("block_shadow", "none")
    put("block_padding", "0px")
    put("block_radius", "12px")
    put("form_gap_width", "0px")
    put("layout_gap", "16px")
    both("panel_background_fill", "surface-2")

    # Labels and hints
    both("block_title_text_color", "text-2")
    put("block_title_text_weight", "600")
    put("block_title_text_size", "13px")
    both("block_info_text_color", "text-3")
    put("block_info_text_size", "12px")
    both("block_label_text_color", "text-3")
    both("block_label_background_fill", "surface-2")
    put("block_label_border_width", "0px")
    put("block_label_text_size", "12px")
    put("block_label_text_weight", "500")
    put("block_label_radius", "8px")

    # Inputs
    both("input_background_fill", "surface")
    both("input_background_fill_focus", "surface")
    both("input_background_fill_hover", "surface")
    both("input_border_color", "line-strong")
    both("input_border_color_hover", "line-strong")
    both("input_border_color_focus", "accent")
    put("input_border_width", "1px")
    put("input_shadow", "none")
    put(
        "input_shadow_focus",
        f"0 0 0 3px {LIGHT['accent-soft']}",
        f"0 0 0 3px {DARK['accent-soft']}",
    )
    put("input_padding", "10px 12px")
    put("input_radius", "10px")
    put("input_text_size", "14.5px")
    both("input_placeholder_color", "text-3")

    # Buttons
    both("button_primary_background_fill", "accent")
    both("button_primary_background_fill_hover", "accent-hover")
    put("button_primary_border_color", "transparent")
    put("button_primary_border_color_hover", "transparent")
    put("button_primary_text_color", "#ffffff")
    put("button_primary_text_color_hover", "#ffffff")
    both("button_secondary_background_fill", "surface")
    both("button_secondary_background_fill_hover", "surface-3")
    both("button_secondary_border_color", "line-strong")
    both("button_secondary_border_color_hover", "line-strong")
    both("button_secondary_text_color", "text-2")
    both("button_secondary_text_color_hover", "text")
    put("button_border_width", "1px")
    both("button_shadow", "shadow-sm")
    put("button_large_padding", "12px 22px")
    put("button_large_radius", "12px")
    put("button_large_text_weight", "600")
    put("button_large_text_size", "15px")
    put("button_small_padding", "7px 14px")
    put("button_small_radius", "9px")
    put("button_small_text_weight", "550")
    put("button_small_text_size", "13px")

    # Radios / checkboxes
    both("checkbox_background_color", "surface")
    both("checkbox_background_color_selected", "accent")
    both("checkbox_border_color", "line-strong")
    both("checkbox_border_color_selected", "accent")
    both("checkbox_border_color_focus", "accent")
    both("checkbox_label_background_fill", "surface")
    both("checkbox_label_background_fill_hover", "surface-2")
    both("checkbox_label_background_fill_selected", "accent-soft")
    both("checkbox_label_border_color", "line-strong")
    both("checkbox_label_border_color_hover", "line-strong")
    put("checkbox_label_border_width", "1px")
    put("checkbox_label_padding", "8px 14px")
    put("checkbox_label_shadow", "none")
    both("checkbox_label_text_color", "text-2")
    both("checkbox_label_text_color_selected", "accent-text")
    put("checkbox_label_text_size", "13.5px")

    # Misc
    both("slider_color", "accent")
    both("loader_color", "accent")
    both("table_even_background_fill", "surface")
    both("table_odd_background_fill", "surface-2")
    both("table_border_color", "line")
    both("error_background_fill", "danger-soft")
    both("error_text_color", "danger")
    both("shadow_drop", "shadow-sm")
    both("shadow_drop_lg", "shadow-md")

    return theme.set(**values)


# ---------------------------------------------------------------------------
# App bar + page-load script
# ---------------------------------------------------------------------------

HEADER_HTML = """
<header class="app-header">
  <div class="brand">
    <div class="brand-mark" aria-hidden="true">
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor"
           stroke-width="2.4" stroke-linecap="round">
        <path d="M4 10v4M8 6v12M12 3v18M16 8v8M20 11v2"/>
      </svg>
    </div>
    <div class="brand-text">
      <h1 class="main-title">Qwen3-TTS Studio</h1>
      <p class="sub-title">Voice cloning, text-to-speech &amp; podcast generation</p>
    </div>
  </div>
  <button type="button" class="params-toggle" aria-expanded="true"
          title="Show or hide the generation settings panel">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         stroke-width="2" stroke-linecap="round" aria-hidden="true">
      <line x1="4" y1="6" x2="20" y2="6"/><line x1="4" y1="12" x2="20" y2="12"/>
      <line x1="4" y1="18" x2="20" y2="18"/>
      <circle cx="9" cy="6" r="2.2" fill="currentColor"/>
      <circle cx="15" cy="12" r="2.2" fill="currentColor"/>
      <circle cx="8" cy="18" r="2.2" fill="currentColor"/>
    </svg>
    <span>Generation settings</span>
  </button>
</header>
"""

# Runs once on page load (gr.Blocks(js=...)). Uses event delegation so it keeps
# working when Gradio re-renders the header.
APP_JS = """
() => {
  const KEY = 'qts-params-collapsed';
  const apply = (collapsed) => {
    document.querySelectorAll('.params-col').forEach(
      (c) => c.classList.toggle('is-collapsed', collapsed));
    document.querySelectorAll('.params-toggle').forEach((b) => {
      b.setAttribute('aria-expanded', String(!collapsed));
      b.classList.toggle('is-off', collapsed);
    });
  };
  let saved = false;
  try { saved = localStorage.getItem(KEY) === '1'; } catch (e) {}
  apply(saved);
  const obs = new MutationObserver(() => {
    if (document.querySelector('.params-toggle') && document.querySelector('.params-col')) {
      apply(saved);
      obs.disconnect();
    }
  });
  obs.observe(document.body, { childList: true, subtree: true });
  setTimeout(() => obs.disconnect(), 15000);
  document.addEventListener('click', (e) => {
    const btn = e.target.closest && e.target.closest('.params-toggle');
    if (!btn) return;
    const col = document.querySelector('.params-col');
    if (!col) return;
    const collapsed = !col.classList.contains('is-collapsed');
    apply(collapsed);
    try { localStorage.setItem(KEY, collapsed ? '1' : '0'); } catch (e) {}
  });
}
"""

# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------

_COMPONENT_CSS = """
/* ===================================================================
   BASE
   =================================================================== */
html, body { background: var(--ui-bg) !important; }
.gradio-container {
    max-width: 1680px !important;
    margin: 0 auto !important;
    padding: 20px 28px 56px !important;
    background: transparent !important;
    -webkit-font-smoothing: antialiased;
    color: var(--ui-text);
    /* Gradio sets overflow:hidden here, which turns this element into the scroll
       container and breaks position:sticky on the settings panel. clip keeps the
       clipping without creating a scroll container. */
    overflow: visible !important;
    overflow-x: clip !important;
}
footer, .built-with, .gradio-container > footer { display: none !important; }
.gradio-container * { scrollbar-width: thin; scrollbar-color: var(--ui-line-strong) transparent; }
.gradio-container :focus-visible { outline: 2px solid var(--ui-accent); outline-offset: 2px; }

/* ===================================================================
   APP BAR
   =================================================================== */
.gradio-container .app-header {
    display: flex; align-items: center; justify-content: space-between;
    gap: 16px; padding: 2px 2px 14px;
}
.brand { display: flex; align-items: center; gap: 14px; min-width: 0; }
.brand-mark {
    flex: none; width: 44px; height: 44px; border-radius: 14px;
    display: grid; place-items: center; color: #fff;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 58%, #a855f7 100%);
    box-shadow: 0 10px 22px -10px var(--ui-glow), inset 0 1px 0 rgba(255,255,255,.28);
}
.gradio-container .app-header h1.main-title {
    font-size: 1.4rem; font-weight: 700; letter-spacing: -0.022em;
    margin: 0; padding: 0; line-height: 1.2; color: var(--ui-text); border: 0;
}
.gradio-container .app-header p.sub-title {
    margin: 2px 0 0; font-size: .86rem; color: var(--ui-text-3);
}
.params-toggle {
    display: inline-flex; align-items: center; gap: 8px; cursor: pointer;
    padding: 8px 14px; border-radius: 11px; font: inherit; font-size: .85rem; font-weight: 550;
    color: var(--ui-accent-text); background: var(--ui-accent-soft);
    border: 1px solid transparent; transition: background .15s, color .15s;
}
.params-toggle:hover { background: var(--ui-surface-3); color: var(--ui-text); }
.params-toggle.is-off { color: var(--ui-text-2); background: var(--ui-surface); border-color: var(--ui-line-strong); }

/* ===================================================================
   PAGE LAYOUT: main column + sticky settings column
   =================================================================== */
.app-body { flex-wrap: nowrap !important; align-items: flex-start !important; gap: 20px !important; }
.main-col { flex: 1 1 0 !important; min-width: 0 !important; }
.params-col {
    flex: 0 0 320px !important; width: 320px; min-width: 320px !important;
    position: sticky; top: 16px; align-self: flex-start;
    max-height: calc(100vh - 32px); overflow-y: auto;
    background: var(--ui-surface); border: 1px solid var(--ui-line);
    border-radius: 20px; padding: 18px 18px 20px !important;
    box-shadow: var(--ui-shadow-sm); gap: 14px !important;
}
.params-col.is-collapsed { display: none !important; }
@media (max-width: 1100px) {
    .app-body { flex-wrap: wrap !important; }
    .params-col { flex: 1 1 100% !important; width: 100%; min-width: 0 !important; position: static; max-height: none; }
}

/* ===================================================================
   TABS
   =================================================================== */
/* main navigation: pills */
.main-tabs > .tab-nav {
    border: 0 !important; gap: 4px; padding: 0 0 14px !important; margin: 0 !important;
}
.main-tabs > .tab-nav button {
    border: 0 !important; border-radius: 11px !important; margin: 0 !important;
    padding: 9px 16px !important; font-size: .92rem !important; font-weight: 550 !important;
    color: var(--ui-text-2) !important; background: transparent !important;
    transition: background .15s, color .15s;
}
.main-tabs > .tab-nav button:hover { background: var(--ui-surface-3) !important; color: var(--ui-text) !important; }
.main-tabs > .tab-nav button.selected {
    background: var(--ui-accent-soft) !important; color: var(--ui-accent-text) !important; font-weight: 650 !important;
}
.main-tabs > .tab-nav button.selected::after, .main-tabs > .tab-nav button::after { display: none !important; }
.main-tabs > .tabitem {
    background: var(--ui-surface) !important; border: 1px solid var(--ui-line) !important;
    border-radius: 20px !important; padding: 26px 28px 30px !important; box-shadow: var(--ui-shadow-sm);
}

/* nested navigation: segmented control, no card */
.sub-tabs > .tab-nav {
    display: inline-flex !important; width: fit-content; gap: 2px; padding: 4px !important;
    background: var(--ui-surface-3); border: 0 !important; border-radius: 12px; margin: 0 0 4px !important;
}
.sub-tabs > .tab-nav button {
    border: 0 !important; border-radius: 9px !important; margin: 0 !important;
    padding: 6px 16px !important; font-size: .86rem !important; font-weight: 550 !important;
    color: var(--ui-text-2) !important; background: transparent !important;
}
.sub-tabs > .tab-nav button.selected {
    background: var(--ui-surface) !important; color: var(--ui-text) !important;
    box-shadow: var(--ui-shadow-sm); font-weight: 650 !important;
}
.sub-tabs > .tab-nav button::after { display: none !important; }
.sub-tabs > .tabitem { background: transparent !important; border: 0 !important; padding: 14px 0 0 !important; box-shadow: none !important; }

/* ===================================================================
   TYPOGRAPHY + SECTION HEADERS
   =================================================================== */
.gradio-container .section-header {
    font-size: .72rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase;
    color: var(--ui-text-3); margin: 0 0 2px; padding: 0; border: 0;
}
.gradio-container .prose h2 { font-size: 1.2rem; font-weight: 700; letter-spacing: -0.015em; margin: 0 0 4px; }
.gradio-container .prose em { font-style: normal; color: var(--ui-text-3); }
.gradio-container .prose p { margin: 0 0 .4rem; }
.gradio-container .info-text, .gradio-container .info-text * { color: var(--ui-text-3); font-size: .84rem; }
.gradio-container .info-text ul { padding-left: 1.1rem; margin: 0; }

/* ===================================================================
   FORM CONTROLS
   =================================================================== */
.gradio-container .form {
    background: transparent !important; border: 0 !important; box-shadow: none !important;
    gap: 16px !important; overflow: visible !important;
}
.gradio-container .form > .block, .gradio-container .form > fieldset.block { border: 0 !important; }
.gradio-container .block.hide-container { margin: 0; }
.gradio-container label > span[data-testid="block-info"],
.gradio-container span.has-info { font-weight: 600; color: var(--ui-text-2); }
.gradio-container textarea, .gradio-container input[type="text"],
.gradio-container input[type="number"], .gradio-container input[type="password"] {
    transition: border-color .15s, box-shadow .15s;
}
.gradio-container textarea { line-height: 1.55; }
.gradio-container textarea:disabled, .gradio-container input:disabled {
    background: var(--ui-surface-2) !important; color: var(--ui-text-2) !important;
    -webkit-text-fill-color: var(--ui-text-2); opacity: 1;
}
.gradio-container input[type="range"] { accent-color: var(--ui-accent); }
.gradio-container .wrap-inner, .gradio-container .secondary-wrap { border-radius: 10px; }
.gradio-container ul.options { border-radius: 12px; border: 1px solid var(--ui-line); box-shadow: var(--ui-shadow-md); }

/* radio chips + checkboxes */
.gradio-container fieldset .wrap { gap: 8px; flex-wrap: wrap; }
.gradio-container fieldset label { border-radius: 10px; transition: background .15s, border-color .15s; }
.gradio-container fieldset label.selected { border-color: var(--ui-accent) !important; }

/* ===================================================================
   BUTTONS
   =================================================================== */
.gradio-container button.lg, .gradio-container button.sm { transition: background .15s, transform .12s, box-shadow .15s, border-color .15s; }
.gradio-container button.secondary:hover { border-color: var(--ui-line-strong); }
.gradio-container button.generate-btn {
    min-height: 48px !important; border-radius: 13px !important; font-weight: 650 !important;
    letter-spacing: .005em; background: var(--ui-accent) !important; color: #fff !important; border: 0 !important;
    box-shadow: 0 10px 22px -12px var(--ui-glow), inset 0 1px 0 rgba(255,255,255,.18) !important;
}
.gradio-container button.generate-btn:hover:not(:disabled) {
    background: var(--ui-accent-hover) !important; transform: translateY(-1px);
    box-shadow: 0 14px 26px -12px var(--ui-glow), inset 0 1px 0 rgba(255,255,255,.18) !important;
}
.gradio-container button.generate-btn:active:not(:disabled) { transform: translateY(0); }
.gradio-container button.generate-btn:disabled { opacity: .65; cursor: progress; }
.gradio-container button.stop {
    background: var(--ui-danger-soft) !important; color: var(--ui-danger) !important;
    border-color: transparent !important; box-shadow: none !important;
}
.gradio-container button.stop:hover { filter: brightness(.96); }
.mini-btn-row { gap: 8px !important; }
.mini-btn-row button { min-width: 0 !important; }

/* ===================================================================
   ACCORDIONS
   =================================================================== */
.gradio-container .block:has(> button.label-wrap) {
    border: 1px solid var(--ui-line) !important; border-radius: 14px !important;
    background: var(--ui-surface-2) !important; overflow: hidden;
}
.gradio-container button.label-wrap {
    padding: 11px 14px !important; font-size: .88rem !important; font-weight: 600 !important;
    color: var(--ui-text) !important; transition: background .15s;
}
.gradio-container button.label-wrap:hover { background: var(--ui-surface-3) !important; }
.gradio-container button.label-wrap .icon { font-size: .6rem; color: var(--ui-text-3); }
.gradio-container .block:has(> button.label-wrap) > div:not(.wrap) { padding: 4px 14px 16px; }

/* ===================================================================
   MEDIA: audio players, uploads, tables
   =================================================================== */
.gradio-container .block:has(audio),
.gradio-container .block:has(> .empty),
.gradio-container .block:has(.file-preview-holder),
.gradio-container .block:has(.table-wrap) {
    border: 1px solid var(--ui-line) !important; border-radius: 14px !important;
    background: var(--ui-surface-2) !important; overflow: hidden;
}
.gradio-container .block:has(> .empty) { min-height: 112px; }
.gradio-container .block:has(> .empty) .empty { min-height: 112px; display: grid; place-items: center; color: var(--ui-text-3); }
.gradio-container .block:has(input[type="file"]) {
    border: 1.5px dashed var(--ui-line-strong) !important; border-radius: 14px !important;
    background: var(--ui-surface-2) !important; overflow: hidden; transition: border-color .15s, background .15s;
}
.gradio-container .block:has(input[type="file"]):hover { border-color: var(--ui-accent) !important; background: var(--ui-accent-soft) !important; }
.gradio-container .block:has(audio) { padding: 0 !important; }
.align-end { align-items: flex-end !important; }
.align-end > button { min-height: 42px; height: 42px; }

/* ===================================================================
   GENERATION SETTINGS PANEL
   =================================================================== */
.save-indicator-wrap { position: absolute !important; top: 16px; right: 16px; width: auto !important; min-width: 0 !important; z-index: 2; pointer-events: none; }
.params-title { font-size: 1rem; font-weight: 700; letter-spacing: -0.01em; color: var(--ui-text); }
.params-sub { font-size: .8rem; color: var(--ui-text-3); margin-top: 2px; }
.hint { font-size: .84rem; line-height: 1.5; color: var(--ui-text-3); }
.speakers-head { flex-wrap: nowrap !important; align-items: center !important; margin-top: 6px; }
.speakers-head > * { min-width: 0 !important; }
.speakers-head > :first-child { flex: 1 1 auto !important; }
.speakers-head > button { flex: 0 0 auto !important; }
.save-indicator {
    display: inline-flex; align-items: center; gap: 5px; font-size: .74rem; font-weight: 600;
    color: var(--ui-success); background: var(--ui-success-soft); padding: 4px 10px; border-radius: 99px;
    opacity: 0; transition: opacity .3s ease; white-space: nowrap;
}
.save-indicator.show { opacity: 1; animation: ui-fade-out 3s ease forwards; }
.save-indicator.show::before { content: "\\2713"; font-weight: 800; }
@keyframes ui-fade-out { 0%,80% { opacity: 1; } 100% { opacity: 0; } }
.params-label { font-size: .72rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; color: var(--ui-text-3); }
.preset-btn-group { display: grid !important; grid-template-columns: repeat(3, 1fr); gap: 8px !important; }
.preset-btn-group > * { min-width: 0 !important; width: 100%; }
.preset-btn-lg { padding: 8px 6px !important; font-size: .84rem !important; font-weight: 600 !important; border-radius: 10px !important; }
.reset-btn {
    background: transparent !important; border: 0 !important; box-shadow: none !important;
    color: var(--ui-text-3) !important; font-size: .8rem !important; font-weight: 550 !important;
    padding: 4px 0 !important; text-decoration: underline; text-underline-offset: 3px;
}
.reset-btn:hover { color: var(--ui-accent-text) !important; background: transparent !important; }
.compact-slider-row { gap: 16px !important; flex-wrap: wrap; }
.params-col .info, .params-col span[data-testid="block-info"] + div { font-size: .76rem !important; line-height: 1.4 !important; }
.params-col input[type="number"] { width: 64px; padding: 5px 8px !important; font-size: .85rem; text-align: right; border-radius: 8px; }
.params-note { font-size: .74rem; font-weight: 600; color: var(--ui-text-3); margin: 6px 0 0; }

/* ===================================================================
   CHARACTER COUNT / STATUS CALLOUTS
   =================================================================== */
.char-count {
    display: inline-block; font-size: .76rem; font-weight: 550; color: var(--ui-text-3);
    background: var(--ui-surface-3); padding: 3px 10px; border-radius: 99px; font-variant-numeric: tabular-nums;
}
.char-count.char-warning { color: var(--ui-warn); background: var(--ui-warn-soft); }
.char-count.char-error { color: var(--ui-danger); background: var(--ui-danger-soft); }

/* Inline-coloured status messages emitted by the generation handlers */
.gradio-container div[style*="#dc3545"], .gradio-container div[style*="#28a745"], .gradio-container div[style*="#b8860b"] {
    padding: 9px 13px; border-radius: 11px; font-size: .88rem; font-weight: 550; line-height: 1.45;
}
.gradio-container [style*="#dc3545"] { color: var(--ui-danger) !important; }
.gradio-container div[style*="#dc3545"] { background: var(--ui-danger-soft); }
.gradio-container [style*="#28a745"] { color: var(--ui-success) !important; }
.gradio-container div[style*="#28a745"] { background: var(--ui-success-soft); }
.gradio-container [style*="#b8860b"] { color: var(--ui-warn) !important; }
.gradio-container div[style*="#b8860b"] { background: var(--ui-warn-soft); }
.gradio-container [style*="#888"], .gradio-container [style*="#666"] { color: var(--ui-text-3) !important; }

/* ===================================================================
   EMPTY STATES
   =================================================================== */
.empty-state, .persona-gallery-empty {
    text-align: center; padding: 28px 20px; color: var(--ui-text-3); font-size: .9rem;
    border: 1.5px dashed var(--ui-line-strong); border-radius: 14px; background: var(--ui-surface-2);
}

/* ===================================================================
   PODCAST: speaker slots
   =================================================================== */
.speaker-row {
    display: flex !important; flex-wrap: nowrap !important; align-items: flex-end !important;
    gap: 10px !important; padding: 12px; border: 1px solid var(--ui-line); border-radius: 14px; background: var(--ui-surface-2);
}
.speaker-row > .form { flex: 1 1 auto !important; min-width: 0 !important; display: grid !important; gap: 10px !important; }
.speaker-row > .form > * { min-width: 0 !important; width: auto !important; }
.speaker-row.ai-slot > .form { grid-template-columns: minmax(96px, 124px) minmax(0, 1fr); }
.speaker-row.custom-slot > .form { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }
.speaker-row.custom-slot > .form > :nth-child(3) { grid-column: 1 / -1; }
.speaker-row > button { flex: 0 0 42px !important; min-width: 42px !important; width: 42px !important; height: 42px; padding: 0 !important; border-radius: 10px !important; }
.refresh-btn { min-width: 0 !important; }

/* ===================================================================
   PODCAST: progress
   =================================================================== */
.step-indicator { display: flex; align-items: flex-start; padding: 4px 4px 0; }
.step-item { display: flex; flex-direction: column; align-items: center; gap: 8px; min-width: 76px; }
.step-icon {
    width: 38px; height: 38px; border-radius: 50%; display: grid; place-items: center;
    font-size: .8rem; font-weight: 700; border: 2px solid transparent; transition: all .3s ease;
    background: var(--ui-surface-3); color: var(--ui-text-3);
}
.step-icon.completed { background: var(--ui-accent); color: #fff; font-size: 1rem; box-shadow: 0 6px 14px -6px var(--ui-glow); }
.step-icon.current {
    background: var(--ui-surface); color: var(--ui-accent-text); border-color: var(--ui-accent);
    font-size: .68rem; animation: ui-pulse 2s ease-in-out infinite;
}
@keyframes ui-pulse { 0%,100% { box-shadow: 0 0 0 0 var(--ui-glow); } 50% { box-shadow: 0 0 0 8px transparent; } }
.step-label { font-size: .68rem; font-weight: 650; letter-spacing: .07em; text-transform: uppercase; color: var(--ui-text-3); }
.step-item.completed .step-label, .step-item.current .step-label { color: var(--ui-text); }
.step-connector { flex: 1; height: 2px; margin: 18px 6px 0; border-radius: 2px; background: var(--ui-line-strong); transition: background .3s; }
.step-connector.completed { background: var(--ui-accent); }

.progress-bar-slider { padding-top: 2px; }
.progress-bar-slider input[type="range"] {
    -webkit-appearance: none; appearance: none; height: 8px !important; border-radius: 99px;
    pointer-events: none; background-color: var(--ui-surface-3); box-shadow: inset 0 0 0 1px var(--ui-line);
    /* Gradio drives the fill through the inline background-size; supply the fill colour itself. */
    background-image: linear-gradient(var(--ui-accent), var(--ui-accent)) !important;
    background-repeat: no-repeat !important; background-position: left center !important;
    transition: background-size .5s cubic-bezier(.4, 0, .2, 1);
}
.progress-bar-slider input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; width: 0; height: 0; opacity: 0; }
.progress-bar-slider input[type="range"]::-moz-range-thumb { width: 0; height: 0; opacity: 0; border: 0; }
.progress-bar-slider input[type="number"] {
    border: 0 !important; background: transparent !important; box-shadow: none !important;
    font-weight: 700; font-variant-numeric: tabular-nums; pointer-events: none; -moz-appearance: textfield;
}
.progress-bar-slider input[type="number"]::-webkit-inner-spin-button { display: none; }
.progress-bar-slider .tab-like-container, .progress-bar-slider button.reset-button { display: none !important; }
.status-row { gap: 12px !important; flex-wrap: wrap; }
.progress-anchor { scroll-margin-top: 16px; }

/* ===================================================================
   PODCAST: transcript + outline preview
   =================================================================== */
.dlg-list > *, .outline-list > *, .history-display.prose > *, .history-display .prose > * { flex-shrink: 0; }
.dlg-list, .outline-list { display: flex; flex-direction: column; gap: 8px; max-height: 440px; overflow-y: auto; padding: 2px 4px 2px 0; }
.dlg {
    display: grid; grid-template-columns: 34px 1fr; gap: 12px; padding: 11px 14px;
    border: 1px solid var(--ui-line); border-radius: 14px; background: var(--ui-surface-2);
}
.dlg-avatar {
    width: 34px; height: 34px; border-radius: 50%; display: grid; place-items: center;
    font-size: .72rem; letter-spacing: .02em; font-weight: 700; background: var(--ui-accent-soft); color: var(--ui-accent-text);
}
.dlg.s1 .dlg-avatar { background: rgba(16,185,129,.14); color: #0f9f6e; }
.dlg.s2 .dlg-avatar { background: rgba(245,158,11,.16); color: #b7791f; }
.dlg.s3 .dlg-avatar { background: rgba(244,63,94,.12); color: #e11d48; }
.dark .dlg.s1 .dlg-avatar { color: #34d399; } .dark .dlg.s2 .dlg-avatar { color: #fbbf24; } .dark .dlg.s3 .dlg-avatar { color: #fb7185; }
.dlg-speaker { font-size: .78rem; font-weight: 700; letter-spacing: .02em; color: var(--ui-text); margin-bottom: 2px; }
.dlg-text { font-size: .92rem; line-height: 1.6; color: var(--ui-text-2); }
.dlg-tag {
    display: inline-block; font-size: .72rem; font-weight: 600; line-height: 1; padding: 3px 8px; margin: 0 2px;
    border-radius: 99px; color: var(--ui-text-3); background: var(--ui-surface-3); vertical-align: 1px;
}
.dlg-more { text-align: center; font-size: .82rem; color: var(--ui-text-3); padding: 4px 0 2px; }
.outline-item { display: grid; grid-template-columns: 28px 1fr; gap: 10px; padding: 10px 12px; border: 1px solid var(--ui-line); border-radius: 12px; background: var(--ui-surface-2); }
.outline-num { width: 28px; height: 28px; border-radius: 9px; display: grid; place-items: center; font-size: .78rem; font-weight: 700; background: var(--ui-accent-soft); color: var(--ui-accent-text); }
.outline-title { font-weight: 650; font-size: .9rem; color: var(--ui-text); }
.outline-desc { font-size: .84rem; line-height: 1.5; color: var(--ui-text-3); margin-top: 2px; }

/* ===================================================================
   HISTORY
   =================================================================== */
.history-display.prose, .history-display .prose { display: flex; flex-direction: column; gap: 10px; max-height: 420px; overflow-y: auto; padding: 2px 4px 2px 0; }
.history-card {
    display: grid; gap: 6px; padding: 12px 14px; border: 1px solid var(--ui-line); border-radius: 14px;
    background: var(--ui-surface); transition: border-color .15s, box-shadow .15s;
}
.history-card:hover { border-color: var(--ui-line-strong); box-shadow: var(--ui-shadow-sm); }
.history-card-header { display: flex; align-items: center; gap: 10px; }
.history-icon { width: 30px; height: 30px; border-radius: 9px; display: grid; place-items: center; background: var(--ui-accent-soft); font-size: .95rem; }
.history-time { font-size: .78rem; font-weight: 550; color: var(--ui-text-3); font-variant-numeric: tabular-nums; }
.history-star { margin-left: auto; font-size: .95rem; color: var(--ui-warn); }
.history-text { font-size: .9rem; line-height: 1.5; color: var(--ui-text); display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.history-meta { display: flex; flex-wrap: wrap; gap: 6px; font-size: .74rem; color: var(--ui-text-2); }
.history-meta > span { background: var(--ui-surface-3); padding: 2px 9px; border-radius: 99px; }
.history-id { font-family: var(--font-mono, ui-monospace, Consolas, monospace); font-size: .68rem; color: var(--ui-text-3); opacity: .85; }

/* ===================================================================
   PERSONAS
   =================================================================== */
.persona-cards-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 14px; padding: 4px 0; }
.persona-card {
    background: var(--ui-surface); border: 1px solid var(--ui-line); border-radius: 16px; padding: 16px 18px;
    transition: border-color .15s, box-shadow .15s, transform .15s;
}
.persona-card:hover { border-color: var(--ui-line-strong); box-shadow: var(--ui-shadow-md); transform: translateY(-2px); }
.persona-card-header { display: flex; justify-content: space-between; align-items: flex-start; gap: 10px; margin-bottom: 10px; }
.persona-name { font-size: 1.02rem; font-weight: 700; letter-spacing: -0.01em; color: var(--ui-text); }
.persona-voice-badge { font-size: .66rem; font-weight: 700; letter-spacing: .06em; padding: 3px 9px; border-radius: 99px; background: var(--ui-accent-soft); color: var(--ui-accent-text); }
.persona-traits { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
.persona-trait { font-size: .76rem; padding: 3px 10px; border-radius: 99px; background: var(--ui-surface-3); color: var(--ui-text-2); }
.persona-bio { font-size: .85rem; line-height: 1.5; color: var(--ui-text-3); display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }

/* ===================================================================
   OPENAI API
   =================================================================== */
.speaker-row.api-slot > .form { grid-template-columns: minmax(110px, 150px) minmax(0, 1fr) minmax(110px, 150px); }
.api-status {
    display: inline-flex; align-items: center; gap: 8px; padding: 6px 12px; border-radius: 99px;
    font-size: .84rem; font-weight: 600; background: var(--ui-surface-3); color: var(--ui-text-2);
}
.api-status code { font-size: .8rem; background: transparent; color: inherit; }
.api-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--ui-text-3); flex: none; }
.api-status-on { background: var(--ui-success-soft); color: var(--ui-success); }
.api-status-on .api-dot { background: var(--ui-success); box-shadow: 0 0 0 3px var(--ui-success-soft); }
.api-msg { font-size: .86rem; color: var(--ui-text-2); }
.api-msg-error { color: var(--ui-danger); }

/* ===================================================================
   MOTION + SMALL SCREENS
   =================================================================== */
@media (prefers-reduced-motion: reduce) {
    .gradio-container * { animation: none !important; transition: none !important; }
}
@media (max-width: 760px) {
    .gradio-container { padding: 14px 12px 40px !important; }
    .main-tabs > .tabitem { padding: 18px 16px 22px !important; border-radius: 16px !important; }
    .params-toggle span { display: none; }
    .step-item { min-width: 56px; }
    .step-connector { margin-left: 2px; margin-right: 2px; }
    .speaker-row.ai-slot > .form { grid-template-columns: minmax(0, 1fr); }
    .speaker-row.api-slot > .form { grid-template-columns: minmax(0, 1fr); }
}
"""

APP_CSS = _token_css() + _COMPONENT_CSS
