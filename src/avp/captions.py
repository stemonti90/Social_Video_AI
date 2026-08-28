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
    max_w = width * 0.95                              # the line + plate must stay inside the frame
    for idx, (phrase, active, _s, _e) in enumerate(events):
        img = Image.new("RGBA", (width, band_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        tokens = [w.text for w in phrase]

        # Shrink the font for THIS group until the line (+ plate padding ≈ 1.1·fs) fits inside the
        # frame — otherwise a long word/group ran off both edges (x went negative → clipped/invisible).
        def _line_total(fnt):
            sw = draw.textlength(" ", font=fnt)
            return sum(draw.textlength(t, font=fnt) for t in tokens) + sw * (len(tokens) - 1)

        fs, f = style.fontsize, font
        while fs > 30 and _line_total(f) + int(fs * 1.1) > max_w:
            fs = int(fs * 0.9)
            f = _truetype(fs, style.font)

        space_w = draw.textlength(" ", font=f)
        widths = [draw.textlength(t, font=f) for t in tokens]
        total = sum(widths) + space_w * (len(tokens) - 1)
        x = (width - total) / 2.0
        y = band_h / 2.0
        # legibility plate behind the text (sized to the actual font used)
        pad_x, pad_y = int(fs * 0.55), int(fs * 0.40)
        half = fs * 0.72
        draw.rounded_rectangle([x - pad_x, y - half - pad_y, x + total + pad_x, y + half + pad_y],
                               radius=int(fs * 0.32), fill=(0, 0, 0, 140))
        for k, token in enumerate(tokens):
            draw.text((x, y), token, font=f, anchor="lm",
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
    """An inviting but sober end card over a soft cosmic backdrop: brand chip + app name +
    tagline + a clear amber CTA button + handle — the visual behind the spoken call-to-action."""
    from PIL import Image, ImageDraw

    w, h = video.width, video.height
    cx = w // 2
    amber, ink = (255, 194, 75), (10, 14, 24)

    # soft generated space scene (not a flat panel), then darkened so text/button pop
    render_cosmic_backdrop(path, video, seed=99)
    img = Image.blend(Image.open(path).convert("RGB"), Image.new("RGB", (w, h), (5, 7, 16)), 0.5)
    draw = ImageDraw.Draw(img)

    # brand chip with initials (amber rounded square)
    words = funnel.app_name.split()
    initials = (words[0][:1] + words[1][:1] if len(words) >= 2 else funnel.app_name[:2]).upper()
    chip, chip_y = int(w * 0.17), int(h * 0.30)
    draw.rounded_rectangle([cx - chip // 2, chip_y - chip // 2, cx + chip // 2, chip_y + chip // 2],
                           radius=int(chip * 0.28), fill=amber)
    draw.text((cx, chip_y), initials, font=_truetype(int(chip * 0.46)), fill=ink, anchor="mm")

    # app name
    draw.text((cx, int(h * 0.45)), funnel.app_name, font=_truetype(int(w * 0.092)),
              fill=(255, 255, 255), anchor="mm", stroke_width=4, stroke_fill=(0, 0, 0))

    # tagline
    tag_f = _truetype(int(w * 0.042))
    y = int(h * 0.525)
    for line in _wrap(draw, funnel.tagline, tag_f, int(w * 0.82)):
        draw.text((cx, y), line, font=tag_f, fill=(190, 214, 255), anchor="mm")
        y += int(w * 0.058)

    # amber CTA pill button (label configurable per channel language, not hardcoded Italian)
    cta, cta_f = (getattr(funnel, "cta_button", "") or "Get the app  ·  Link in bio"), _truetype(int(w * 0.05))
    bw, bh, by = int(draw.textlength(cta, font=cta_f) + w * 0.14), int(w * 0.125), int(h * 0.66)
    draw.rounded_rectangle([cx - bw // 2, by - bh // 2, cx + bw // 2, by + bh // 2],
                           radius=bh // 2, fill=amber)
    draw.text((cx, by), cta, font=cta_f, fill=ink, anchor="mm")

    # handle
    draw.text((cx, int(h * 0.735)), funnel.handle, font=_truetype(int(w * 0.036)),
              fill=(220, 228, 245), anchor="mm")
    img.save(path)


def render_cosmic_backdrop(path: Path, video: VideoConfig, seed: int = 0) -> None:
    """Last-resort generated space backdrop. The old version blended big pastel ellipses with
    ImageChops.add(scale=1.7) — which DIVIDES the result, muddying everything into a dark teal
    'old book canvas' that read as a rendering error on screen for whole segments. Now: near-black
    sky, a faint blue/violet nebula veil via screen-blend (never green), and a dense starfield with
    varied brightness plus a few glowing stars — unmistakably space even under Ken Burns zoom."""
    import random
    from PIL import Image, ImageChops, ImageDraw, ImageFilter

    rnd = random.Random(seed * 9173 + 7)
    w, h = video.width, video.height
    base = Image.new("RGB", (w, h), (3, 4, 10))

    # faint nebula veil — blue/violet only, screen-blended so it can only BRIGHTEN, never muddy
    neb = Image.new("RGB", (w, h), (0, 0, 0))
    nd = ImageDraw.Draw(neb)
    palette = [(30, 24, 70), (18, 34, 78), (54, 26, 72), (12, 22, 60)]
    for _ in range(5):
        rad = rnd.randint(w // 3, int(w * 0.9))
        cx, cy = rnd.randint(0, w), rnd.randint(0, h)
        nd.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=rnd.choice(palette))
    neb = neb.filter(ImageFilter.GaussianBlur(w // 5))
    base = ImageChops.screen(base, neb)

    # dense starfield: many faint, some medium, a handful bright with a soft glow
    d = ImageDraw.Draw(base)
    for _ in range(1400):
        x, y = rnd.randint(0, w - 2), rnd.randint(0, h - 2)
        b = rnd.randint(70, 210)
        d.point((x, y), fill=(b, b, min(255, b + 14)))
    for _ in range(140):
        x, y = rnd.randint(0, w - 3), rnd.randint(0, h - 3)
        b = rnd.randint(170, 255)
        d.ellipse([x, y, x + 1, y + 1], fill=(b, b, min(255, b + 10)))
    glow = Image.new("RGB", (w, h), (0, 0, 0))
    gd = ImageDraw.Draw(glow)
    for _ in range(16):
        x, y = rnd.randint(20, w - 20), rnd.randint(20, h - 20)
        r = rnd.randint(2, 4)
        gd.ellipse([x - r * 3, y - r * 3, x + r * 3, y + r * 3], fill=(40, 44, 70))
        gd.ellipse([x - r, y - r, x + r, y + r], fill=(235, 238, 255))
    base = ImageChops.screen(base, glow.filter(ImageFilter.GaussianBlur(2)))
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
