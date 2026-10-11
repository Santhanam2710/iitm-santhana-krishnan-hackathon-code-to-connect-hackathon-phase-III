"""Capture dashboard screenshots for the README and the deck (needs: pip install playwright; playwright install chromium).

    python scripts/take_screenshots.py [--port 8077]
Starts the demo server on the SVB replay at high speed, waits for the replay to finish, then
captures each tab to docs/screenshots/.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"


def status(port: int) -> dict:
    return json.loads(urllib.request.urlopen(f"http://localhost:{port}/api/status", timeout=5).read())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--speed", type=float, default=20000)
    a = ap.parse_args()
    subprocess.run(["bash", str(ROOT / "scripts" / "serve_bg.sh"), str(a.port), str(a.speed), "svb_2023-03"], check=True)
    for _ in range(240):
        st = status(a.port)
        if st["replay"].get("total") and not st["replay"].get("running"):
            break
        time.sleep(1)
    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
        pg.goto(f"http://localhost:{a.port}")
        pg.wait_for_timeout(2500)
        pg.locator(".card").nth(3).click()
        pg.wait_for_timeout(600)
        pg.screenshot(path=str(OUT / "01_live_signals.png"))
        pg.click("text=Module A · Rebalancer")
        pg.wait_for_timeout(1500)
        pg.screenshot(path=str(OUT / "02_module_a.png"))
        pg.click("text=Module B · Stress test")
        pg.wait_for_timeout(1500)
        pg.screenshot(path=str(OUT / "03_module_b.png"))
        pg.screenshot(path=str(OUT / "03b_module_b_full.png"), full_page=True)
        pg.click("text=Engine Lab")
        pg.fill("#labIn", "Microsoft beats Google to acquire gaming studio while Amazon slumps")
        pg.click("#labRun")
        pg.wait_for_timeout(1200)
        pg.screenshot(path=str(OUT / "04_engine_lab.png"))
        pg.click("text=System & results")
        pg.wait_for_timeout(1500)
        pg.screenshot(path=str(OUT / "05_system.png"))
        b.close()
    subprocess.run(["bash", "-c", f"fuser -k {a.port}/tcp >/dev/null 2>&1 || true"])
    print("screenshots ->", OUT)


if __name__ == "__main__":
    main()
