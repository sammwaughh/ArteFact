import io
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image

def convert_pdf_to_markdown(pdf_path: Path, out_root: Path) -> Path:
    pdf_path = Path(pdf_path).resolve()
    work_id = pdf_path.stem
    out_dir = Path(out_root) / "Marker_Output" / work_id
    img_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    md_lines = []
    with fitz.open(pdf_path) as doc:
        for page_index in range(len(doc)):
            page = doc[page_index]
            # Text as markdown
            md_text = page.get_text("markdown") or ""
            md_lines.append(f"\n\n## Page {page_index}\n")
            if md_text.strip():
                md_lines.append(md_text.strip())

            # Extract images
            images = page.get_images(full=True)
            for img_idx, (xref, *_rest) in enumerate(images):
                img_info = doc.extract_image(xref)
                img_bytes = img_info["image"]
                ext = img_info.get("ext", "png")
                # Normalize/convert to RGB for JPEG if needed
                img_name = f"_page_{page_index}_Picture_{img_idx}.{ext}"
                img_path = img_dir / img_name
                try:
                    with open(img_path, "wb") as f:
                        f.write(img_bytes)
                except Exception:
                    # Fallback via PIL if raw write fails
                    im = Image.open(io.BytesIO(img_bytes))
                    im = im.convert("RGB")
                    img_name = f"_page_{page_index}_Picture_{img_idx}.jpg"
                    img_path = img_dir / img_name
                    im.save(img_path, format="JPEG", quality=90)

                # Reference in markdown
                rel = img_path.relative_to(out_dir)
                md_lines.append(f"\n![{img_name}]({rel.as_posix()})\n")

    md_path = out_dir / f"{work_id}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines).strip() + "\n")
    return md_path