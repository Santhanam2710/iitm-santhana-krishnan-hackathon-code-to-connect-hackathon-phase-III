# Wholesale book derivation

Source transactions: `data/portfolio/transactions_sample.csv` (5,000 rows).

Amount distribution: lognormal fit mu=3.61, sigma=1.09; loan tickets rescaled to a $1,800m book.

| Sector | Spend share | Fraud rate | Relative risk | EAD $m | PD | LGD | Beta |
|---|---|---|---|---|---|---|---|
| Consumer | 28.7% | 0.506% | 1.21 | 517.4 | 1.58% | 40% | 1.0 |
| Real Estate | 16.7% | 0.475% | 1.13 | 300.6 | 2.58% | 35% | 1.2 |
| Energy | 15.2% | 0.279% | 0.67 | 272.9 | 1.65% | 40% | 1.0 |
| Technology | 12.6% | 0.325% | 0.77 | 226.9 | 0.85% | 45% | 0.9 |
| Healthcare | 12.0% | 0.312% | 0.74 | 215.7 | 0.75% | 40% | 0.8 |
| Industrials | 10.4% | 0.729% | 1.73 | 186.3 | 1.42% | 30% | 1.1 |
| Banks | 4.5% | 0.000% | 0.00 | 80.2 | 0.45% | 45% | 1.3 |

PD = sector base PD x (0.75 + 0.25 x relative fraud rate), clipped to [0.3%, 6%].
Market book parameters are fixed, typical values; see the table in wholesale_book.csv.

Assumption: consumer spend mix is a proxy for the sector mix of a bank's corporate clients. It is a modelling
convenience for the hackathon, stated openly; a real book would come from the bank's own exposure data.
