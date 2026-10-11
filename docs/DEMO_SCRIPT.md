# Demo script (target 4:45, hard limit 5:00)

Setup before recording: `python -m trident demo --speed 600`, browser at 110% zoom on the Live
signals tab, a terminal visible behind it. Numbers below are from the shipped results; read
the current ones from `docs/results/RESULTS.md` if you regenerated them.

| Time | Screen | Action | Say (paraphrase in your own words) |
|---|---|---|---|
| 0:00-0:25 | Dashboard, replay clock at 8 Mar 2023 | none | "In March 2023 a US bank failed in under 48 hours, and the run spread through news and social media before any risk report caught up. TRIDENT reads that text and turns it into a decision." |
| 0:25-0:45 | Terminal | show `python -m trident demo` already running; flash the architecture image | "One command from the README, no API keys, no network: it replays the SVB week. Four kinds of sources, one engine, three fields per signal, two consumers." |
| 0:45-1:45 | Live signals | click the 21:15 SVB card, then a held decision in the gate bar | "Each signal has an entity, a sentiment from −1 to +1, an event class and an impact from 1 to 10, plus the exact phrases that drove them, the model versions and a data hash. This first headline scored impact 8 but only one source reported it, so the gate held it. 25 minutes later a second wire confirmed it." |
| 1:45-2:00 | Engine Lab | run "Microsoft beats Google to acquire gaming studio while Amazon slumps" | "Sentiment is per company: positive for Microsoft, negative for Google and Amazon in the same sentence. Engine time is about a millisecond." |
| 2:00-2:50 | Module A | point at the JPM and BAC rows of the heatmap, then the suppressed-trades tile | "Module A tilts a 15-stock index toward positive news, scaled down for volatile names, inside caps and a turnover limit. Bank news spills into JPMorgan and Bank of America through sector contagion. The no-trade band blocked hundreds of small trades: it does not chase headlines." |
| 2:50-4:00 | Module B | pause on STRESS TEST TRIGGERED; walk the waterfall; CET1 bar; reverse stress | "Module B reprices a synthetic wholesale book the way a bank would: loans with Vasicek PD migration, bonds with duration and convexity, derivatives with Greeks and Black-Scholes. A credit event at impact 9 costs about 80 million and takes CET1 from 13% to 8.4%. The reverse stress test says the 7% buffer breaks at about 1.1 times the full-severity credit scenario." Run the COVID preset: "the 2020 crash would take us below the 4.5% minimum." |
| 4:00-4:35 | Slide 5 (Key Results) | show the slide | "Measured: 1.4 ms per document, sentiment F1 0.65 against VADER's 0.47 on our authored set, 58 of 58 behavioral tests, impact correlated with realized moves out of sample. One honest limitation: our synthetic training text is easy to detect, so our registry rejected models trained on it." |
| 4:35-4:50 | Slide 6 or dashboard | none | "TRIDENT: one engine, two strategies, three signals. Text in, capital decision out, every step auditable." |

Recovery lines if something goes wrong live:
* Replay already finished: press **Restart replay** (top right); it restarts in seconds.
* Server down: `python -m trident demo --speed 6000` brings the full story back within a minute.
* Network question: "Live mode runs in the other terminal; the demo deliberately never depends on the network."
