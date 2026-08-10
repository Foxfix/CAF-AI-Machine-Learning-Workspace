# Project 1 - EDA: New York City Airbnb Open Data


## Dataset - verified real source

- **Name:** New York City Airbnb Open Data (AB_NYC_2019.csv)
- **Kaggle page:** https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data
- **Uploader:** `dgomonov`
- **Original source:** Inside Airbnb (http://insideairbnb.com/get-the-data) - Kaggle's copy is adapted from this.
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0) - confirmed via the figshare mirror citation of this exact dataset (https://figshare.com/articles/dataset/Airbnb_NYC_2019_data/21120481). **Re-check the license badge on the live Kaggle page before publishing this repo publicly** - Kaggle listings can update their metadata over time; this is a production-time check, not an assumption.
- **Size:** ~48,895 rows, 16 columns, single file `AB_NYC_2019.csv`.
- **Known columns** (confirmed from multiple independent public analyses of this exact file): `id, name, host_id, host_name, neighbourhood_group, neighbourhood, latitude, longitude, room_type, price, minimum_nights, number_of_reviews, last_review, reviews_per_month, calculated_host_listings_count, availability_365`.

## How to download the data (do this yourself - the file is NOT committed to this repo)

1. Create a free Kaggle account if you don't have one: https://www.kaggle.com
2. Go to https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data
3. Click "Download" (or use the Kaggle API - see below) and extract `AB_NYC_2019.csv`.
4. Place the file at `data/raw.csv` in this repo (rename it - the filename in the repo is standardized so `src/data_utils.py` doesn't have to guess).

### Option B - Kaggle API (reproducible, scriptable)
```bash
pip install kaggle
# requires ~/.kaggle/kaggle.json with your API credentials, generated from
# your Kaggle account settings page - never commit this file.
kaggle datasets download -d dgomonov/new-york-city-airbnb-open-data -p data/ --unzip
mv data/AB_NYC_2019.csv data/raw.csv
```

## Why this dataset was chosen for an EDA project
Realistic messiness for a beginner-appropriate EDA exercise: missing `reviews_per_month` (correlates with zero-review listings - must be reasoned about, not blindly dropped), free-text `neighbourhood` values, `price == 0` listings that are very likely data errors, and enough columns (geography, room type, host behavior) to support several genuinely different analytical questions rather than one.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
jupyter lab notebooks/analysis.ipynb
```
Run all cells top to bottom. The notebook imports its cleaning/loading logic from `src/data_utils.py` - it does not duplicate that logic inline.

## Repository structure
```
project1-eda-airbnb/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── raw.csv              # you place this here after downloading - see above
├── notebooks/
│   └── analysis.ipynb
└── src/
    └── data_utils.py
```

## Deliverables checklist
- [ ] `data/raw.csv` downloaded per instructions above (not committed - see `.gitignore`)
- [ ] `notebooks/analysis.ipynb` executed top-to-bottom with no errors
- [ ] Missing-value / outlier handling decisions documented in markdown cells
- [ ] At least 4 EDA questions answered, each with a chart + written insight (see notebook)
- [ ] Closing "Business Insights" section (5–8 bullets)

## License note on this repo's own code
The code in `src/` and `notebooks/` is original and may be licensed however you like for your portfolio (e.g. MIT). The **data itself** remains under Airbnb/Inside Airbnb's CC BY 4.0 terms - attribute the original source in any public write-up, per the license.
