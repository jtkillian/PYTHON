# python c:/Users/jkill/OneDrive/Desktop/JDW-DEV/GitHub/JDwinkle/PYTHON/apps/mu/tools/RickRoll-pdf.py input.pdf output.pdf --mode chunk/full --dpi 400
import argparse
import sys
import fitz  # PyMuPDF


LYRICS = """We're no strangers to love
You know the rules and so do I
A full commitment's what I'm thinkin' of
You wouldn't get this from any other guy
I just wanna tell you how I'm feeling
Gotta make you understand
Never gonna give you up, never gonna let you down
Never gonna run around and desert you
Never gonna make you cry, never gonna say goodbye
Never gonna tell a lie and hurt you
We've known each other for so long
Your heart's been aching, but you're too shy to say it
Inside, we both know what's been going on
We know the game and we're gonna play it
And if you ask me how I'm feeling
Don't tell me you're too blind to see
Never gonna give you up, never gonna let you down
Never gonna run around and desert you
Never gonna make you cry, never gonna say goodbye
Never gonna tell a lie and hurt you
Never gonna give you up, never gonna let you down
Never gonna run around and desert you
Never gonna make you cry, never gonna say goodbye
Never gonna tell a lie and hurt you
We've known each other for so long
Your heart's been aching, but you're too shy to say it
Inside, we both know what's been going on
We know the game and we're gonna play it
I just wanna tell you how I'm feeling
Gotta make you understand
Never gonna give you up, never gonna let you down
Never gonna run around and desert you
Never gonna make you cry, never gonna say goodbye
Never gonna tell a lie and hurt you
Never gonna give you up, never gonna let you down
Never gonna run around and desert you
Never gonna make you cry, never gonna say goodbye
Never gonna tell a lie and hurt you
Never gonna give you up, never gonna let you down
Never gonna run around and desert you
Never gonna make you cry, never gonna say goodbye
Never gonna tell a lie and hurt you
""".strip()


def rasterize_to_doc(
    src_path: str, dpi: int = 300, image_format: str = "png", jpeg_quality: int = 92
) -> fitz.Document:
    """Render each page to an image and return a new image-only PDF doc."""
    with fitz.open(src_path) as src:
        out = fitz.open()
        for page in src:
            pix = page.get_pixmap(dpi=dpi)
            if image_format.lower() in ("jpg", "jpeg"):
                stream = pix.tobytes("jpeg", jpg_quality=jpeg_quality)  # NOTE: 'jpg_quality'
            else:
                stream = pix.tobytes("png")
            new = out.new_page(width=page.rect.width, height=page.rect.height)
            new.insert_image(new.rect, stream=stream)
    return out


def measure_text_width(text: str, fontname: str, fontsize: float) -> float:
    """Accurate width (points) for wrapping."""
    return fitz.get_text_length(text, fontname=fontname, fontsize=fontsize)


def wrap_text_for_rect(text: str, fontname: str, fontsize: float, max_width: float):
    """Greedy wrap using exact font metrics per line."""
    lines_out = []
    for para in text.splitlines():
        words = para.split(" ")
        line = ""
        for w in words:
            trial = (line + " " + w).strip() if line else w
            if measure_text_width(trial, fontname, fontsize) <= max_width:
                line = trial
            else:
                if line:
                    lines_out.append(line)
                if measure_text_width(w, fontname, fontsize) <= max_width:
                    line = w
                else:
                    chunk = ""
                    for ch in w:
                        if measure_text_width(chunk + ch, fontname, fontsize) <= max_width:
                            chunk += ch
                        else:
                            lines_out.append(chunk)
                            chunk = ch
                    line = chunk
        lines_out.append(line)  # keep final/blank line
    return lines_out


def add_hidden_text_layer(
    doc: fitz.Document,
    text_source: list[str],
    mode: str = "full",  # 'full' = whole lyrics each page; 'chunk' = split across pages
    fontname: str = "helv",
    fontsize: float = 12.0,
    margin: float = 36.0,
):
    """
    Write an invisible (render_mode=3) selectable text layer.
    """
    full_text = "\n".join(text_source).strip()
    remaining_lines = full_text.splitlines() if mode == "chunk" else None

    for page in doc:
        rect = page.rect
        left, top = margin, margin
        right, bottom = rect.width - margin, rect.height - margin
        max_width = max(0, right - left)
        max_height = max(0, bottom - top)

        if mode == "full":
            page_text = full_text
        else:
            if not remaining_lines:
                page_text = ""
            else:
                wrapped = []
                for line in remaining_lines:
                    wrapped.extend(wrap_text_for_rect(line, fontname, fontsize, max_width))
                line_height = fontsize * 1.25
                max_lines = int(max_height // line_height) if line_height > 0 else len(wrapped)
                page_lines = wrapped[:max_lines] if max_lines > 0 else []
                page_text = "\n".join(page_lines)
                remaining_lines = wrapped[max_lines:]

        sh = page.new_shape()
        x, y = left, top
        line_height = fontsize * 1.25
        lines = wrap_text_for_rect(page_text, fontname, fontsize, max_width) if page_text else []

        # render_mode=3 = invisible text (still selectable/searchable)
        for ln in lines:
            if y + line_height > bottom + 1:
                break
            sh.insert_text(
                fitz.Point(x, y), ln, fontname=fontname, fontsize=fontsize, render_mode=3
            )
            y += line_height

        sh.commit(overlay=True)


def main():
    ap = argparse.ArgumentParser(
        description="Rickroll a PDF: rasterize (remove original selectable text) "
        "and add an invisible, selectable Rick Astley lyrics layer."
    )
    ap.add_argument("input", help="Input PDF")
    ap.add_argument("output", help="Output PDF")
    ap.add_argument(
        "--dpi", type=int, default=300, help="Rasterization DPI (300 is a good default)"
    )
    ap.add_argument(
        "--image-format", choices=["png", "jpeg"], default="png", help="Image format to embed"
    )
    ap.add_argument(
        "--jpeg-quality", type=int, default=92, help="JPEG quality if using --image-format jpeg"
    )
    ap.add_argument(
        "--layer-fontsize", type=float, default=12.0, help="Hidden layer font size (points)"
    )
    ap.add_argument("--layer-margin", type=float, default=36.0, help="Page margin (points)")
    ap.add_argument(
        "--mode",
        choices=["full", "chunk"],
        default="full",
        help="full = whole song on every page; chunk = split song across pages",
    )
    args = ap.parse_args()

    img_doc = rasterize_to_doc(
        args.input, dpi=args.dpi, image_format=args.image_format, jpeg_quality=args.jpeg_quality
    )

    add_hidden_text_layer(
        img_doc,
        text_source=[LYRICS],
        mode=args.mode,
        fontname="helv",
        fontsize=args.layer_fontsize,
        margin=args.layer_margin,
    )

    img_doc.save(args.output, deflate=True)
    img_doc.close()
    print(f"✅ Wrote {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("❌ Error:", e)
        sys.exit(1)
