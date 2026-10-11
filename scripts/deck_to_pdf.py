"""Convert docs/presentation.pptx to docs/presentation.pdf with LibreOffice (headless).

    python scripts/deck_to_pdf.py
If LibreOffice is not installed, open the .pptx in PowerPoint and use File > Export > PDF.
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        sys.exit("LibreOffice not found: export docs/presentation.pptx to PDF from PowerPoint instead.")
    subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", str(ROOT / "docs"), str(ROOT / "docs" / "presentation.pptx")], check=True, timeout=180)
    print("wrote docs/presentation.pdf")


if __name__ == "__main__":
    main()
