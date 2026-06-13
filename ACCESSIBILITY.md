# Accessibility — Legge Stanca (L. 4/2004) · WCAG 2.1 AA · EN 301 549

**Scope:** the AUT Video Pipeline desktop control panel (Electron renderer, `electron/renderer/`).
The Python engine (`avp`) is a CLI/library, out of scope for UI accessibility.

## Conformance status: **conforme** to WCAG 2.1 level AA (verified)

Verified **2026-06-13**. Every view and dynamic state passes both automated and manual checks
(see methodology). The remaining items are administrative/UX, listed under *Residual*.

## Methodology

1. **Automated — axe-core 4.10.2**, rulesets `wcag2a, wcag2aa, wcag21a, wcag21aa` (plus
   `best-practice`), run on **every view/state**: Projects, New, Settings, and the project
   workspace tabs (Review / Build / Preview / Publish), including the not-ready "guard" state.
   **Result: 0 violations** in all states.
2. **Manual / runtime** — asserted programmatically on the live UI:
   - `lang="it"`; skip-link → valid target; landmarks (`banner` / `nav` / `main`)
   - ARIA tab pattern (tablist/tab/tabpanel, exactly one `aria-selected`, roving `tabindex`),
     keyboard-operable (←/→/Home/End move selection **and** focus)
   - 18/18 controls have an accessible name; 10 live regions (`role=status` / `aria-live`)
   - focus moves to the view heading on navigation; visible `:focus-visible`
   - status is conveyed by **shape + text, not colour alone**; `prefers-reduced-motion` honoured
   - contrast measured (secondary text 8.28:1 — well above the 4.5:1 AA threshold)

## WCAG 2.1 AA — criteria summary

| Criterion | Status |
|---|---|
| 1.3.1 Info & Relationships | ✅ landmarks, lists, ARIA tabs |
| 1.4.1 Use of Colour | ✅ status = glyph + screen-reader text |
| 1.4.3 Contrast (Minimum) | ✅ ≥ 4.5:1 (measured 8.28:1) |
| 2.1.1 Keyboard | ✅ tabs via ←/→/Home/End |
| 2.4.1 Bypass Blocks | ✅ skip-link |
| 2.4.3 Focus Order | ✅ focus → heading on view change |
| 2.4.7 Focus Visible | ✅ `:focus-visible` |
| 2.3.3 Animation from Interactions | ✅ `prefers-reduced-motion` |
| 3.1.1 Language of Page | ✅ `lang="it"` |
| 4.1.2 Name, Role, Value | ✅ named controls + ARIA roles |
| 4.1.3 Status Messages | ✅ `aria-live` regions |

## Residual / honest limitations

- Automated (axe-core) + manual review cover the automatically- and programmatically-testable
  criteria. A **usability test with a real screen reader** (VoiceOver / NVDA) is the recommended
  final validation of the lived experience; **not yet performed**.
- A formal **AgID "Dichiarazione di accessibilità"** is mandatory only for public-sector bodies.
  If ever required, file it on the AgID portal using this document as the technical basis.
- Verified on the browser-preview build of the renderer; the packaged Electron app loads the
  identical renderer code, so the results carry over.

## Reporting an accessibility problem

Open an issue on the repository stating the view, the assistive technology used, and the problem.
