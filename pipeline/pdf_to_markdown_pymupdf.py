import io
import shutil
import subprocess
from pathlib import Path

def _extract_text(pdf_path: Path) -> str:
    try:
        from pdfminer.high_level import extract_text
        return extract_text(str(pdf_path)) or ""
    except Exception:
        return ""

def _extract_images_pdfimages(pdf_path: Path, img_dir: Path) -> bool:
    exe = shutil.which("pdfimages")
    if not exe:
        return False
    img_dir.mkdir(parents=True, exist_ok=True)
    # -all to keep original formats; -p to include page numbers; -q quiet
    prefix = img_dir / "image"
    cmd = [exe, "-all", "-p", "-q", str(pdf_path), str(prefix)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

def _extract_images_pdfminer(pdf_path: Path, img_dir: Path) -> None:
    # Best-effort image extraction using pdfminer.six
    from pdfminer.pdfparser import PDFParser
    from pdfminer.pdfdocument import PDFDocument
    from pdfminer.pdfpage import PDFPage
    from pdfminer.pdftypes import resolve1
    from PIL import Image

    img_dir.mkdir(parents=True, exist_ok=True)

    def _as_list(x):
        if x is None:
            return []
        if isinstance(x, (list, tuple)):
            return list(x)
        return [x]

    with open(pdf_path, "rb") as fh:
        parser = PDFParser(fh)
        doc = PDFDocument(parser)
        for page_index, page in enumerate(PDFPage.create_pages(doc)):
            try:
                resources = resolve1(page.resources)
            except Exception:
                resources = None
            if not resources:
                continue
            xobjs = resolve1(resources.get("XObject"))
            if not xobjs:
                continue
            img_idx = 0
            for name, xobj in xobjs.items():
                try:
                    stream = resolve1(xobj)
                    subtype = stream.attrs.get("Subtype")
                    if str(subtype) != "/Image":
                        continue
                    filters = [str(f) for f in _as_list(stream.attrs.get("Filter"))]
                    data = stream.get_data()  # decoded by pdfminer based on filters
                    ext = "bin"
                    # Map common filters to file types; when decoded, we try Pillow
                    if any("DCTDecode" in f for f in filters):
                        ext = "jpg"
                        out = img_dir / f"_page_{page_index}_Picture_{img_idx}.{ext}"
                        out.write_bytes(data)
                    elif any("JPXDecode" in f for f in filters):
                        ext = "jp2"
                        out = img_dir / f"_page_{page_index}_Picture_{img_idx}.{ext}"
                        out.write_bytes(data)
                    else:
                        # Try to materialize via Pillow from decoded bytes
                        try:
                            im = Image.open(io.BytesIO(data))
                            # Force to RGB/LA where applicable for wider compatibility
                            if im.mode not in ("RGB", "RGBA", "L", "LA"):
                                im = im.convert("RGB")
                            ext = "png"
                            out = img_dir / f"_page_{page_index}_Picture_{img_idx}.{ext}"
                            im.save(out)
                        except Exception:
                            # As a last resort, dump raw bytes
                            out = img_dir / f"_page_{page_index}_Picture_{img_idx}.bin"
                            out.write_bytes(data)
                    img_idx += 1
                except Exception:
                    continue

def convert_pdf_to_markdown(pdf_path: Path, out_root: Path) -> Path:
    pdf_path = Path(pdf_path).resolve()
    work_id = pdf_path.stem
    out_dir = Path(out_root) / "Marker_Output" / work_id
    img_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    # Text (plain) → write as .md for downstream compatibility
    text = _extract_text(pdf_path)
    md_path = out_dir / f"{work_id}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")

    # Images: prefer pdfimages if available, else pure-Python fallback
    if not _extract_images_pdfimages(pdf_path, img_dir):
        try:
            _extract_images_pdfminer(pdf_path, img_dir)
        except Exception:
            pass

    return md_path