"""Build replay files for the demo and the integration tests.

Two windows are reconstructed from widely reported public events:
  * svb_2023-03   SVB / regional-bank run, 8-19 March 2023
  * covid_2020-03 COVID crash, 17 Feb - 23 Mar 2020

Provenance (important for the jury and for academic honesty):
  * Items marked ``reconstructed`` paraphrase publicly reported events in our own words.
    They are not verbatim headlines from any outlet, timestamps are approximate (UTC),
    and figures are rounded.
  * Items marked ``is_synthetic: true`` are invented social posts or noise added to test
    dedup, the confirmation gate and entity disambiguation. They describe no real event.
  * ``--gdelt`` (run on your own machine) adds REAL headlines for the same window from the
    GDELT DOC 2.0 API, tagged ``source_type: gdelt``. GDELT is blocked in some sandboxes.

Usage:
  python scripts/build_replay.py               # write data/replay/*.jsonl
  python scripts/build_replay.py --gdelt       # also merge real GDELT headlines
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

R = "reconstructed"

# (UTC timestamp, source_type, source_name, text, is_synthetic)
SVB = [
    ("2023-03-08T21:15:00Z", "newswire", "replay:wire-a", "SVB Financial says it sold about $21 billion of securities at a loss of roughly $1.8 billion and plans a $2.25 billion share sale to shore up capital", False),
    ("2023-03-08T21:40:00Z", "newswire", "replay:wire-b", "SVB Financial announces emergency capital raise after $1.8 billion loss on bond portfolio sale", False),
    ("2023-03-08T23:05:00Z", "newswire", "replay:wire-a", "Rating agency downgrades SVB Financial, citing unrealized losses and deposit outflows", False),
    ("2023-03-09T00:30:00Z", "social", "replay:social", "anyone else moving their startup's cash out of SVB tonight? this capital raise looks bad", True),
    ("2023-03-09T13:10:00Z", "newswire", "replay:wire-c", "Venture investors advise portfolio companies to withdraw deposits from Silicon Valley Bank", False),
    ("2023-03-09T14:45:00Z", "newswire", "replay:wire-a", "SVB Financial shares plunge 60% as depositors pull funds and capital raise spooks investors", False),
    ("2023-03-09T15:20:00Z", "social", "replay:social", "$SIVB down 60% lol this is a bank run in real time", True),
    ("2023-03-09T16:05:00Z", "newswire", "replay:wire-b", "Bank stocks tumble as SVB turmoil sparks fears of deposit flight across regional lenders", False),
    ("2023-03-09T17:30:00Z", "newswire", "replay:wire-c", "JPMorgan and Bank of America shares fall as bank index posts worst day in nearly three years", False),
    ("2023-03-09T19:50:00Z", "social", "replay:social", "my company just wired everything out of SVB. 42 billion in withdrawal requests today?? insane", True),
    ("2023-03-09T21:30:00Z", "newswire", "replay:wire-a", "Customers attempted to withdraw about $42 billion from Silicon Valley Bank in a single day, filings later showed", False),
    ("2023-03-10T12:40:00Z", "newswire", "replay:wire-b", "Trading in SVB Financial shares halted premarket as talks to sell the bank falter", False),
    ("2023-03-10T13:15:00Z", "newswire", "replay:wire-c", "SVB Financial abandons share sale and explores emergency sale of the bank", False),
    ("2023-03-10T16:20:00Z", "newswire", "replay:wire-a", "Regulators close Silicon Valley Bank and appoint the FDIC as receiver in largest bank failure since 2008", False),
    ("2023-03-10T16:35:00Z", "newswire", "replay:wire-b", "Silicon Valley Bank seized by California regulators; FDIC takes control of deposits", False),
    ("2023-03-10T16:40:00Z", "newswire", "replay:wire-b", "Silicon Valley Bank seized by California regulators; FDIC takes control of deposits", False),
    ("2023-03-10T17:10:00Z", "social", "replay:social", "uninsured deposits frozen at SVB, payroll on monday is going to be a disaster for a lot of startups", True),
    ("2023-03-10T19:00:00Z", "newswire", "replay:wire-c", "Regional bank stocks slide again as contagion fears spread to First Republic and Western Alliance", False),
    ("2023-03-12T22:20:00Z", "newswire", "replay:wire-a", "New York regulators close Signature Bank as deposit flight spreads", False),
    ("2023-03-12T22:30:00Z", "newswire", "replay:wire-b", "Treasury, Fed and FDIC say all depositors at SVB and Signature Bank will be protected; Fed launches emergency lending facility for banks", False),
    ("2023-03-12T23:00:00Z", "newswire", "replay:wire-c", "Federal Reserve creates Bank Term Funding Program to backstop deposits and stabilize the banking system", False),
    ("2023-03-13T13:45:00Z", "newswire", "replay:wire-a", "First Republic shares plunge more than 60% despite federal backstop; regional bank trading halted repeatedly", False),
    ("2023-03-13T14:30:00Z", "newswire", "replay:wire-b", "Two-year Treasury yields post their sharpest drop in decades as investors bet the Fed will pause rate hikes", False),
    ("2023-03-13T15:10:00Z", "newswire", "replay:wire-c", "Rating agency places six US regional banks on review for downgrade after SVB collapse", False),
    ("2023-03-13T16:00:00Z", "social", "replay:social", "First Republic halted again. if you have more than 250k there, why???", True),
    ("2023-03-13T18:20:00Z", "newswire", "replay:wire-a", "Large banks including JPMorgan and Bank of America see deposit inflows as customers flee smaller lenders", False),
    ("2023-03-14T12:30:00Z", "newswire", "replay:wire-b", "US consumer prices rise 6.0% from a year earlier in February, keeping pressure on the Fed despite bank turmoil", False),
    ("2023-03-14T14:40:00Z", "newswire", "replay:wire-c", "Regional bank shares rebound as fears of contagion ease", False),
    ("2023-03-15T09:30:00Z", "newswire", "replay:wire-a", "Credit Suisse shares plunge 24% after top shareholder rules out more funding", False),
    ("2023-03-15T10:15:00Z", "newswire", "replay:wire-b", "Credit Suisse credit default swaps spike to record as European bank stocks tumble", False),
    ("2023-03-15T11:00:00Z", "social", "replay:social", "credit suisse CDS going vertical. this is not a drill", True),
    ("2023-03-15T21:40:00Z", "newswire", "replay:wire-c", "Swiss National Bank says it will provide liquidity to Credit Suisse if needed", False),
    ("2023-03-16T13:15:00Z", "newswire", "replay:wire-a", "European Central Bank raises interest rates by 50 basis points despite banking turmoil", False),
    ("2023-03-16T20:30:00Z", "newswire", "replay:wire-b", "Eleven of the largest US banks, including JPMorgan Chase and Bank of America, deposit $30 billion in First Republic", False),
    ("2023-03-17T12:30:00Z", "filing", "replay:filing", "SVB Financial Group files for Chapter 11 bankruptcy protection", False),
    ("2023-03-17T15:00:00Z", "newswire", "replay:wire-c", "First Republic shares slump again as investors question rescue deal", False),
    ("2023-03-19T19:30:00Z", "newswire", "replay:wire-a", "UBS agrees to buy Credit Suisse for about $3.2 billion in emergency rescue brokered by Swiss authorities", False),
    # synthetic noise and traps
    ("2023-03-09T18:00:00Z", "social", "replay:social", "made an apple crumble to cope with this week honestly", True),
    ("2023-03-11T15:00:00Z", "newswire", "replay:wire-d", "Fifteen years after Lehman, analysts debate lessons from the 2008 crisis", True),
    ("2023-03-14T17:00:00Z", "social", "replay:social", "rumor: another big bank is about to fail tomorrow, trust me", True),
]

COVID = [
    ("2020-02-21T14:00:00Z", "newswire", "replay:wire-a", "Coronavirus cases surge in Italy and South Korea, raising fears of a global pandemic", False),
    ("2020-02-24T15:00:00Z", "newswire", "replay:wire-b", "Stocks plunge as virus spreads beyond China; Dow falls more than 1,000 points", False),
    ("2020-02-27T20:30:00Z", "newswire", "replay:wire-c", "S&P 500 enters correction just six trading days after a record high", False),
    ("2020-02-17T22:00:00Z", "newswire", "replay:wire-a", "Apple says it will miss quarterly revenue guidance as the virus disrupts iPhone supply in China", False),
    ("2020-03-03T15:00:00Z", "newswire", "replay:wire-b", "Federal Reserve announces emergency half-point rate cut to counter coronavirus threat", False),
    ("2020-03-09T13:30:00Z", "newswire", "replay:wire-c", "Oil prices crash about 25% as Saudi Arabia and Russia start a price war", False),
    ("2020-03-09T14:00:00Z", "newswire", "replay:wire-a", "Trading halted after S&P 500 drops 7% at the open; Exxon Mobil shares tumble with crude", False),
    ("2020-03-11T16:30:00Z", "newswire", "replay:wire-b", "World Health Organization declares the coronavirus outbreak a pandemic", False),
    ("2020-03-11T18:00:00Z", "newswire", "replay:wire-c", "Boeing draws down a $13.8 billion loan as travel collapse hits airlines", False),
    ("2020-03-12T01:00:00Z", "newswire", "replay:wire-a", "US suspends travel from Europe for 30 days in sweeping response to pandemic", False),
    ("2020-03-12T20:15:00Z", "newswire", "replay:wire-b", "Dow plunges 10% in worst day since 1987 crash as pandemic panic deepens", False),
    ("2020-03-12T21:00:00Z", "social", "replay:social", "circuit breakers twice in one week. this is a crash", True),
    ("2020-03-15T21:00:00Z", "newswire", "replay:wire-c", "Federal Reserve cuts rates to near zero and launches $700 billion quantitative easing program", False),
    ("2020-03-16T20:30:00Z", "newswire", "replay:wire-a", "Stocks crash 12% as volatility index hits record despite emergency Fed action", False),
    ("2020-03-17T14:00:00Z", "newswire", "replay:wire-b", "Boeing seeks $60 billion in government aid for aerospace industry as liquidity crisis looms", False),
    ("2020-03-17T16:00:00Z", "newswire", "replay:wire-c", "Federal Reserve revives commercial paper funding facility to keep credit flowing", False),
    ("2020-03-18T15:00:00Z", "newswire", "replay:wire-a", "Automakers halt US production as virus spreads; Tesla faces pressure to close factory", False),
    ("2020-03-18T19:00:00Z", "newswire", "replay:wire-b", "Corporate bond spreads widen to levels last seen in 2009 as investors flee credit", False),
    ("2020-03-19T14:00:00Z", "newswire", "replay:wire-c", "Weekly jobless claims surge as layoffs spread through the economy", False),
    ("2020-03-20T12:00:00Z", "newswire", "replay:wire-a", "Boeing suspends dividend and share buybacks to preserve cash", False),
    ("2020-03-21T15:00:00Z", "social", "replay:social", "every airline and boeing are going to need bailouts. junk bonds getting destroyed", True),
    ("2020-03-23T12:00:00Z", "newswire", "replay:wire-b", "Federal Reserve pledges unlimited asset purchases and launches corporate bond buying facilities", False),
    ("2020-03-23T13:00:00Z", "newswire", "replay:wire-c", "Fed backstop for corporate credit sparks rebound in bond markets", False),
    ("2020-02-25T16:00:00Z", "social", "replay:social", "stocking up on apple juice and pasta lol", True),
]


def rows_for(items: list[tuple], window: str) -> list[dict]:
    out = []
    for ts, st, sn, text, syn in sorted(items, key=lambda x: x[0]):
        out.append({"ts": ts, "source_type": st, "source_name": sn, "text": text, "is_synthetic": syn,
                    "provenance": "synthetic" if syn else R, "window": window})
    return out


async def gdelt_rows(query: str, start: datetime, end: datetime) -> list[dict]:
    from trident.ingest.gdelt import gdelt_doc_search

    rows = []
    t = start
    while t < end:
        t2 = min(t + timedelta(hours=12), end)
        try:
            arts = await gdelt_doc_search(query, t, t2, maxrecords=75)
        except Exception as exc:
            print("GDELT failed:", exc)
            return rows
        for a in arts:
            sd = a.get("seendate", "")
            ts = datetime.strptime(sd, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z") if sd else None
            if ts and a.get("language", "English") == "English":
                rows.append({"ts": ts, "source_type": "gdelt", "source_name": f"gdelt:{a.get('domain', 'web')}", "text": a.get("title", ""),
                             "url": a.get("url"), "is_synthetic": False, "provenance": "gdelt-real"})
        t = t2
        await asyncio.sleep(1.0)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gdelt", action="store_true", help="merge real GDELT headlines (needs internet)")
    a = ap.parse_args()
    out_dir = ROOT / "data" / "replay"
    out_dir.mkdir(parents=True, exist_ok=True)
    windows = {
        "svb_2023-03": (SVB, '("Silicon Valley Bank" OR SVB OR "Signature Bank" OR "First Republic" OR "Credit Suisse")',
                        datetime(2023, 3, 8, 18, tzinfo=timezone.utc), datetime(2023, 3, 20, tzinfo=timezone.utc)),
        "covid_2020-03": (COVID, '(coronavirus AND (stocks OR Fed OR recession OR oil))',
                          datetime(2020, 2, 17, tzinfo=timezone.utc), datetime(2020, 3, 24, tzinfo=timezone.utc)),
    }
    for name, (items, q, start, end) in windows.items():
        rows = rows_for(items, name)
        if a.gdelt:
            extra = asyncio.run(gdelt_rows(q, start, end))
            print(f"{name}: +{len(extra)} GDELT headlines")
            rows = sorted(rows + extra, key=lambda r: r["ts"])
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {out_dir / (name + '.jsonl')} ({len(rows)} items)")


if __name__ == "__main__":
    main()
