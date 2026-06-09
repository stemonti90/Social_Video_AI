# Fonts — commercial-safe, bundled

The pipeline **burns** caption / endcard / credit text into the distributed video, so the font
must be licensed for **commercial use *and* embedding**. The macOS system fonts (Arial,
Helvetica) are proprietary (Monotype/Linotype) and are **not** licensed for embedding in a
distributed commercial product — they were replaced on 2026-06 during the license audit.

| File | Font | License | Commercial + embed |
|---|---|---|---|
| `Montserrat-Bold.ttf`, `Montserrat-Regular.ttf` | Montserrat | SIL OFL 1.1 — see `Montserrat-OFL.txt` | ✅ |
| `DejaVuSans-Bold.ttf` | DejaVu Sans (Bitstream Vera + Arev) | see `DejaVu-LICENSE.txt` | ✅ |

- **Default** = Montserrat (`config.captions.font`). DejaVu Sans is the always-present fallback.
- `captions._font_path()` resolves fonts **only** from this folder — never `/System/Library/Fonts`.
- To add a font: drop `<Name>-Bold.ttf` here **with its license file**, then set `captions.font: <Name>`.
