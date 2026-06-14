"""Generate an ASS subtitle file with karaoke-style word highlighting.

We emit one Dialogue event per word: the short phrase it belongs to is shown with the
current word highlighted, which gives the familiar word-by-word "pop" used on shorts.
ASS is burned in later by ffmpeg's subtitles filter."""
from __future__ import annotations

from pathlib import Path

from .config import CaptionStyle, FunnelConfig, VideoConfig
from .stt import Word


def _ass_time(t: float) -> str:
    cs = max(0, int(round(t * 100)))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _sanitize(text: str) -> str:
    return text.replace("{", "(").replace("}", ")")


def write_ass(words: list[Word], path: Path, style: CaptionStyle, video: VideoConfig) -> None:
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {video.width}
PlayResY: {video.height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Def,{style.font},{style.fontsize},{style.primary_color},&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,{style.outline},1,2,80,80,{style.margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    group = max(1, style.group)
    lines: list[str] = []
    for i in range(0, len(words), group):
        phrase = words[i:i + group]
        for j, current in enumerate(phrase):
            rendered = []
            for k, w in enumerate(phrase):
                token = _sanitize(w.text)
                if k == j:
                    rendered.append(
                        f"{{\\c{style.highlight_color}}}{token}{{\\c{style.primary_color}}}"
                    )
                else:
                    rendered.append(token)
            text = " ".join(rendered)
            lines.append(
                f"Dialogue: 0,{_ass_time(current.start)},{_ass_time(current.end)},Def,,0,0,0,,{text}"
            )
    path.write_text(header + "\n".join(lines) + "\n")


# --- Pillow renderer: karaoke captions as transparent PNGs (no libass needed) ---

def _caption_events(words: list[Word], group: int):
    """One event per word: (phrase_words, active_index, start, end) for karaoke highlight."""
    events = []
    for i in range(0, len(words), group):
        phrase = words[i:i + group]
        for j, w in enumerate(phrase):
            events.append((phrase, j, w.start, w.end))
    return events


def _ass_to_rgb(ass_color: str, default: tuple[int, int, int]) -> tuple[int, int, int]:
    """Convert an ASS &HAABBGGRR color to an (R, G, B) tuple."""
    try:
        h = ass_color.replace("&H", "").replace("&", "").zfill(8)
        return (int(h[-2:], 16), int(h[-4:-2], 16), int(h[-6:-4], 16))
    except Exception:  # noqa: BLE001
        return default


# Bundled, commercially-licensed fonts live here. We NEVER use the macOS system fonts
# (Arial/Helvetica) — those are proprietary and not licensed for embedding in a distributed
# commercial video. Shipped: Montserrat (SIL OFL) + DejaVu Sans (Bitstream Vera/Arev).
_FONT_DIR = Path(__file__).resolve().parents[2] / "assets" / "fonts"


def _font_path(font_name: str = "Montserrat") -> str | None:
    base = (font_name or "").replace(" ", "")
    for p in (
        _FONT_DIR / f"{base}-Bold.ttf",
        _FONT_DIR / f"{base}-Regular.ttf",
        _FONT_DIR / f"{base}.ttf",
        _FONT_DIR / "Montserrat-Bold.ttf",     # clean default
        _FONT_DIR / "DejaVuSans-Bold.ttf",     # always-present fallback
    ):
        if p.exists():
            return str(p)
    if _FONT_DIR.exists():                      # any bundled ttf, last resort
        anyttf = sorted(_FONT_DIR.glob("*.ttf"))
        if anyttf:
            return str(anyttf[0])
    return None


def _truetype(size: int, font_name: str = "Montserrat"):
    from PIL import ImageFont
    path = _font_path(font_name)
    if path:
        try:
            return ImageFont.truetype(path, size)
        except Exception:  # noqa: BLE001
            pass
    return ImageFont.load_default()


def _load_font(style: CaptionStyle):
    return _truetype(style.fontsize, style.font)


def render_caption_pngs(words: list[Word], out_dir: Path, style: CaptionStyle, video: VideoConfig,
                        total_dur: float | None = None):
    """Render one transparent PNG per word-event. Returns [(png_path, start, end)] with a
    GAP-FREE timeline (each caption stays on screen until the next begins; the first starts at
    0.0 and the last runs to total_dur) so text is NEVER absent. Each caption sits on a
    semi-opaque rounded plate for legibility over any footage."""
    from PIL import Image, ImageDraw

    out_dir.mkdir(parents=True, exist_ok=True)
    font = _load_font(style)
    primary = _ass_to_rgb(style.primary_color, (255, 255, 255))
    highlight = _ass_to_rgb(style.highlight_color, (255, 229, 0))
    band_h = int(style.fontsize * 2.8)
    width = video.width

    events = _caption_events(words, max(1, style.group))
    if not events:
        return []
    # Gap-free, MONOTONIC display timeline: text is on screen from frame 0 to total_dur with no
    # gaps and no two captions ever overlapping — even if STT word timings arrive out of order or
    # bunched closer than the per-event minimum. Force starts non-decreasing (and within
    # total_dur), then run each caption exactly until the next one begins; the last → total_dur.
    n = len(events)
    cap = float(total_dur) if total_dur else None
    starts = [max(0.0, e[2]) for e in events]
    for i in range(1, n):
        starts[i] = max(starts[i], starts[i - 1])    # never let a later caption start earlier
    if cap is not None:
        starts = [min(s, cap) for s in starts]       # keep the whole timeline within the video
    starts[0] = 0.0                                  # cover the lead-in: text from the first frame
    ends = []
    for i in range(n):
        nxt = starts[i + 1] if i + 1 < n else (cap if cap is not None else events[i][3] + 0.8)
        ends.append(max(starts[i], nxt))             # contiguous (end == next start), never < start

    items = []
    for idx, (phrase, active, _s, _e) in enumerate(events):
        img = Image.new("RGBA", (width, band_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        tokens = [w.text for w in phrase]
        space_w = draw.textlength(" ", font=font)
        widths = [draw.textlength(t, font=font) for t in tokens]
        total = sum(widths) + space_w * (len(tokens) - 1)
        x = (width - total) / 2.0
        y = band_h / 2.0
        # legibility plate behind the text
        pad_x, pad_y = int(style.fontsize * 0.55), int(style.fontsize * 0.40)
        half = style.fontsize * 0.72
        draw.rounded_rectangle([x - pad_x, y - half - pad_y, x + total + pad_x, y + half + pad_y],
                               radius=int(style.fontsize * 0.32), fill=(0, 0, 0, 140))
        for k, token in enumerate(tokens):
            draw.text((x, y), token, font=font, anchor="lm",
                      fill=(highlight if k == active else primary),
                      stroke_width=style.outline, stroke_fill=(0, 0, 0, 255))
            x += widths[k] + space_w
        png = out_dir / f"cap_{idx:04d}.png"
        img.save(png)
        items.append((png, starts[idx], ends[idx]))
    return items


def render_phrase_pngs(phrases, out_dir: Path, style: CaptionStyle, video: VideoConfig):
    """phrases = [(text, start, end)] → one wrapped subtitle PNG per phrase on a legibility plate.
    Used for translated subtitles (e.g. English audio + Italian subs); timing is the per-segment
    window passed in (already gap-free, so text is always on screen)."""
    from PIL import Image, ImageDraw

    out_dir.mkdir(parents=True, exist_ok=True)
    font = _load_font(style)
    primary = _ass_to_rgb(style.primary_color, (255, 255, 255))
    max_w = int(video.width * 0.86)
    line_h = int(style.fontsize * 1.25)
    measure = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    items = []
    for idx, (text, start, end) in enumerate(phrases):
        lines = _wrap(measure, text, font, max_w)
        pad_x, pad_y = int(style.fontsize * 0.55), int(style.fontsize * 0.45)
        band_h = line_h * len(lines) + pad_y * 2
        img = Image.new("RGBA", (video.width, band_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        tw = max((draw.textlength(ln, font=font) for ln in lines), default=0)
        cx = video.width / 2.0
        draw.rounded_rectangle([cx - tw / 2 - pad_x, 0, cx + tw / 2 + pad_x, band_h],
                               radius=int(style.fontsize * 0.32), fill=(0, 0, 0, 150))
        y = pad_y
        for ln in lines:
            draw.text((cx, y), ln, font=font, anchor="ma", fill=primary,
                      stroke_width=style.outline, stroke_fill=(0, 0, 0, 255))
            y += line_h
        png = out_dir / f"sub_{idx:04d}.png"
        img.save(png)
        items.append((png, start, end))
    return items


def _wrap(draw, text: str, font, max_w: int) -> list[str]:
    lines, cur = [], ""
    for w in text.split():
        trial = f"{cur} {w}".strip()
        if draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [text]


def render_endcard(path: Path, funnel: FunnelConfig, video: VideoConfig) -> None:
    """A deep-space card: app name + tagline + 'link in bio' — backdrop for the spoken CTA."""
    from PIL import Image, ImageDraw

    w, h = video.width, video.height
    img = Image.new("RGB", (w, h), (7, 9, 20))
    draw = ImageDraw.Draw(img)
    cx = w // 2

    name_f = _truetype(int(w * 0.11))
    tag_f = _truetype(int(w * 0.045))
    small_f = _truetype(int(w * 0.040))

    draw.text((cx, int(h * 0.40)), funnel.app_name, font=name_f, fill=(255, 255, 255),
              anchor="mm", stroke_width=4, stroke_fill=(0, 0, 0))
    y = int(h * 0.50)
    for line in _wrap(draw, funnel.tagline, tag_f, int(w * 0.86)):
        draw.text((cx, y), line, font=tag_f, fill=(176, 208, 255), anchor="mm")
        y += int(w * 0.06)
    draw.text((cx, int(h * 0.62)), f"{funnel.handle}   ·   Link in bio", font=small_f,
              fill=(255, 229, 0), anchor="mm", stroke_width=2, stroke_fill=(0, 0, 0))
    img.save(path)


def render_cosmic_backdrop(path: Path, video: VideoConfig, seed: int = 0) -> None:
    """Last-resort generated space backdrop (nebula + starfield) so no segment is ever black."""
    import random
    from PIL import Image, ImageChops, ImageDraw, ImageFilter

    rnd = random.Random(seed * 9173 + 7)
    w, h = video.width, video.height
    base = Image.new("RGB", (w, h), (6, 8, 16))

    neb = Image.new("RGB", (w, h), (0, 0, 0))
    nd = ImageDraw.Draw(neb)
    palette = [(46, 28, 84), (24, 52, 96), (96, 42, 58), (28, 74, 86), (70, 40, 96)]
    for _ in range(6):
        rad = rnd.randint(w // 3, w)
        cx, cy = rnd.randint(0, w), rnd.randint(0, h)
        nd.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=rnd.choice(palette))
    neb = neb.filter(ImageFilter.GaussianBlur(w // 6))
    base = ImageChops.add(base, neb, scale=1.7)

    d = ImageDraw.Draw(base)
    for _ in range(320):
        x, y = rnd.randint(0, w - 2), rnd.randint(0, h - 2)
        r = 1 if rnd.random() < 0.82 else 2
        b = rnd.randint(150, 255)
        d.ellipse([x, y, x + r, y + r], fill=(b, b, min(255, b + 12)))
    base.save(path)


def render_credit(path: Path, text: str, video: VideoConfig) -> None:
    """A small bottom-corner source credit (e.g. 'NASA')."""
    from PIL import Image, ImageDraw

    font = _truetype(int(video.width * 0.026))
    pad = 16
    probe = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
    tw = int(probe.textlength(text, font=font))
    th = int(video.width * 0.045)
    img = Image.new("RGBA", (max(tw + pad * 2, 40), th), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((pad, th // 2), text, font=font, anchor="lm", fill=(235, 235, 235),
              stroke_width=2, stroke_fill=(0, 0, 0))
    img.save(path)
