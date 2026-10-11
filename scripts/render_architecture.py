"""Render docs/architecture.svg to docs/architecture.png (2x). Needs playwright + chromium,
or open the SVG in any browser and export. python scripts/render_architecture.py"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    from playwright.sync_api import sync_playwright

    svg = (ROOT / "docs" / "architecture.svg").read_text(encoding="utf-8")
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1180, "height": 600}, device_scale_factor=2.5)
        pg.set_content(f"<html><body style='margin:0'>{svg}</body></html>")
        pg.locator("svg").screenshot(path=str(ROOT / "docs" / "architecture.png"))
        b.close()
    print("wrote docs/architecture.png")


if __name__ == "__main__":
    main()
