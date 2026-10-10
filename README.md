# OOP with Python — Advent 2026, Assignment 2

**Student:** _<your name>_ · **Reg. no.:** _<your registration number>_ · **Programme:** MSCS / MSDS

Two mini-projects: a weather client built directly on the HTTP APIs, and an extensible
Uganda tax engine. Reusable logic lives in `src/`, each mini-project has one notebook, and
the whole test suite runs offline.

| # | Mini-project | Notebook | Package |
|---|---|---|---|
| 1 | Fetching and displaying a weather forecast | `notebooks/project1_weather.ipynb` | `src/weather/` |
| 2 | Uganda tax engine | `notebooks/project2_tax.ipynb` | `src/tax/` |

## Repository layout

```
├── README.md
├── requirements.txt
├── src/
│   ├── weather/
│   │   ├── http.py        # Envelope, RetryPolicy, ApiClient, ReplayClient
│   │   ├── sources.py     # WeatherSource ABC, the two providers, geocode/get_forecast
│   │   ├── wmo.py         # WMO 4677 weather-code decoding
│   │   ├── display.py     # tables, summaries and charts
│   │   └── errors.py      # the weather exception hierarchy
│   └── tax/
│       ├── money.py       # the Money value object (Decimal only)
│       ├── core.py        # abstract Tax, TaxContext, TaxResult
│       ├── paye.py        # bands, effective-dated schedules, both residencies
│       ├── payroll.py     # the deduction pipeline
│       ├── vat.py         # standard / zero-rated / exempt
│       ├── corporate.py   # CIT, plus extension A (withholding + rental)
│       └── errors.py      # TaxError hierarchy
├── notebooks/             # one per mini-project, committed with outputs
├── tests/                 # 129 tests, all offline
├── data/samples/          # 16 recorded API responses, for offline runs
├── figures/               # PNGs written by the notebooks
└── tools/
    ├── build_notebooks.py # percent-format .py -> executed .ipynb
    └── nb_src/            # the notebooks' real source, in plain text
```

## Setup

```bash
git clone <this-repo-url>
cd assignment2-a
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Developed on Python 3.13.7 with requests 2.34.2, pandas 3.0.2, matplotlib 3.11.2,
pytest 9.1.1 and hypothesis 6.168.5.

### Before running mini-project 1

MET Norway requires a genuine contact address in the `User-Agent` and will reject clients
without one. Set yours as an environment variable so it never enters the repository:

```bash
export WEATHER_CONTACT_EMAIL="your.name@students.mak.ac.ug"    # Windows: set ...
```

If it is unset the notebook **still runs**: it detects the placeholder and replays the
recorded responses in `data/samples/` instead of making live requests. That is deliberate.
Inventing a contact address would breach MET Norway's terms, and the brief says the same.

## Running

```bash
pytest -q                                   # 129 tests, no network required

jupyter notebook notebooks/                 # then Kernel ▸ Restart & Run All

python tools/build_notebooks.py             # rebuild and execute both notebooks
python tools/build_notebooks.py project2_tax --no-exec
```

Both notebooks run top to bottom with *Restart & Run All* whether or not the network is
available. `build_notebooks.py` refuses to write a notebook whose cells raised, so a
committed notebook is one that executed cleanly.

**Reproducibility.** Mini-project 1 is reproducible through the recorded responses in
`data/samples/`, each stamped with the status, content type and timing of the original
call. Mini-project 2 is deterministic; where the notebook samples incomes it uses a fixed
`random.seed(2026)`.

## Design decisions worth asking about

**Mini-project 1**

* **One guarded entry point.** Everything that touches the network goes through
  `ApiClient.get_json`, so the timeout, `raise_for_status()`, the backoff and the
  `User-Agent` are configured in exactly one place.
* **Retry policy as an object.** `RetryPolicy.delays()` is a pure function, so the backoff
  schedule can be asserted without sleeping or making a request.
* **Transient and permanent failures are different.** A 503 is retried; a 400 raises
  `InvalidRequestError` immediately and carries the provider's own `reason`. Retrying an
  out-of-range latitude would waste four round trips and delay the error the user needs.
* **Parsing is separated from fetching.** `parse_places` and `parse_open_meteo_daily` are
  pure functions over a dict, which is what lets all 53 weather tests run offline — and is
  where the real bugs are.
* **`ReplayClient` does not subclass `ApiClient`.** They share an interface, not an
  implementation. Subclassing would inherit a session, a retry policy and a timeout the
  replay client has no use for.
* **Two providers behind one ABC.** `WeatherSource.daily()` returns the same table shape,
  so the cross-check is a loop and not two code paths — even though Open-Meteo
  publishes daily aggregates and MET Norway has to be rolled up from hourly data.

**Mini-project 2**

* **`Money` refuses `float` loudly**, at construction and in multiplication, and refuses
  `Money * Money` as meaningless. It holds a `Decimal` and rounds half-up to whole
  shillings in one place.
* **`Money` does not subclass `Decimal`.** Inheriting would make the rounding policy
  bypassable and let a float in through an inherited operator. The whole point of the type
  is its ability to *refuse*.
* **Schedules are effective-dated.** `NoScheduleForDateError` is raised instead of
  applying whichever schedule is nearest; adding a future rate change is a data edit.
* **Rates are cited.** Every `PayeSchedule` carries a `source` string, printed in the
  notebook. The non-resident schedule is not in the brief and is marked as sourced from
  URA's published tables.
* **Results explain themselves.** `TaxResult.explain()` prints the band-by-band workings.
  A tax figure a taxpayer cannot follow is one they cannot check.
* **`RentalIncomeTax` deliberately does not inherit from `PayeTax`**, despite the
  temptation — rental income is a separate chargeable head, and inheriting would couple
  two schedules that must change independently.

## Summary of findings

**Mini-project 1 — Weather.** Three things emerged that the code now handles explicitly.
First, Open-Meteo signals "no matches" by **omitting the `results` key entirely** rather
than returning an empty list, so the natural `payload["results"]` raises `KeyError` on an
ordinary 200 response; `parse_places` uses `.get()` and raises `LocationNotFoundError`.
Second, the MET Norway endpoint failed through `urllib` with
`SSL: CERTIFICATE_VERIFY_FAILED` while succeeding through `requests` on the same machine —
`requests` bundles `certifi`, `urllib` trusts the platform store — which is a difference of
capability, not convenience, and would have been invisible had only Open-Meteo been tried.
Third, the two providers agree closely on temperature (about 1.0 °C mean difference, 3.2 °C
at worst) but diverge on precipitation (3.6 mm mean, 8.2 mm at worst). That is the expected
result, not a defect: temperature is a smooth field both models resolve, while
tropical rainfall is convective and sub-grid, and the two daily totals come from different
aggregations.

**Mini-project 2 — Tax.** PAYE reproduces every band edge in the brief exactly: nil to
335,000, then 15,000 at 410,000 and 33,750 at 485,000, reaching 2,888,250 at 10,000,000.
Two edges look surprising and are correct — 335,001 charges nothing because 20% of one
shilling rounds away, and 10,000,001 matches 10,000,000 for the same reason. The check that
matters is the **marginal rate above 10,000,000, which is 40%**, not 30%: the 30% band
continues and the additional 10% sits on top. Applying Local Service Tax before PAYE rather
than after is worth 30% of the LST at the top marginal rate — getting the order wrong
overstates the tax. On VAT, treating exempt supplies as zero-rated would have reclaimed
UGX 270,000 of blocked input VAT and turned a UGX 540,000 payment into UGX 270,000, an
understatement an audit would find. On `float`: `round(2.675, 2)` returns `2.67` because
the stored double sits below 2.675, so round-half-up rounds half *down* — which is
why `Money` refuses floats at the door.

## Data sources and citations

* **Open-Meteo** geocoding and forecast APIs. Data under CC BY 4.0; free for
  non-commercial use. <https://open-meteo.com/>
* **MET Norway Locationforecast 2.0.** Data under CC BY 4.0 / NLOD; requires an
  identifying `User-Agent` with genuine contact details.
  <https://api.met.no/weatherapi/locationforecast/2.0/documentation>
* **WMO 4677 weather codes**, as tabulated in the Open-Meteo documentation.
* **PAYE, NSSF, VAT and corporate rates** from the brief's baseline table, cross-checked
  against the Income Tax Act Cap 340 (Third Schedule) as published by URA
  <https://ura.go.ug>. The **non-resident PAYE schedule** is not given in the brief and is
  taken from URA's published monthly tables. Local Service Tax is set by each local
  government, so the bands used here are an indicative default and are configurable.
* Recorded API responses in `data/samples/` were captured on 2026-10-10 and contain only
  public forecast values — no personal data.

## AI-use declaration

_Edit this section so that it describes accurately how **you** used AI._

This repository was developed with the help of an AI coding assistant. It was used to
scaffold the repository structure, draft the class designs, implementations, tests and
notebook text, and to check results. I have reviewed, run and tested all of the code:
`pytest` passes and both notebooks run top to bottom with Restart & Run All. I can explain
each method, formula and design decision. Key results are checked against hand
calculations, which appear as `assert` statements in the notebooks.
