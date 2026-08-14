# Project 5, Option A - Customer Segmentation (Unsupervised Learning)


## Dataset - verified real source

- **Name:** Mall Customer Segmentation Data
- **Kaggle page:** https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
- **Uploader:** `vjchoudhary7`
- **Verified via:** multiple independent public analyses of this exact file (GitHub repos, Medium write-ups) cross-checked against each other for column names - not taken from a single unverifiable source.
- **Size:** 200 rows, 5 columns: `CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)`.
- **License:** check the current badge on the live Kaggle page before publishing this repo - re-verify at production time, per this curriculum's standing rule for all Kaggle-hosted datasets.
- Deliberately small: this project is about clustering **methodology** (choosing k, validating clusters, interpreting them), not big-data engineering.

### Alternative (if you want a larger, less "toy" dataset)
**UCI "Wholesale customers Data Set"** - 440 clients, annual spending across 6 product categories (Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicatessen), hosted permanently at the UCI Machine Learning Repository. If you use this instead, update this README to say so and adjust `src/clustering_utils.py`'s expected columns.


## How to download the data
1. Create a free Kaggle account: https://www.kaggle.com
2. Go to https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
3. Download and place the CSV at `data/raw.csv` (rename from `Mall_Customers.csv`).

### Or via Kaggle API
```bash
pip install kaggle
kaggle datasets download -d vjchoudhary7/customer-segmentation-tutorial-in-python -p data/ --unzip
mv data/Mall_Customers.csv data/raw.csv
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
jupyter lab notebooks/segmentation.ipynb
```

## Repository structure
```
project3a-customer-segmentation/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── raw.csv                 # you place this here - not committed
├── notebooks/
│   └── segmentation.ipynb
├── src/
│   ├── __init__.py
│   └── clustering_utils.py
└── tests/
    └── test_clustering_utils.py   # runs against synthetic data, proves the logic works
```

## Deliverables checklist
- [ ] `data/raw.csv` downloaded, confirmed 200 rows / 5 columns
- [ ] Scaling applied before distance-based clustering (justified in notebook)
- [ ] Elbow method AND silhouette score plotted vs. k; k chosen with written justification
- [ ] KMeans and AgglomerativeClustering (with dendrogram) both fit and compared
- [ ] PCA to 2D for visualization, explained variance ratio reported
- [ ] Cluster interpretation table with persona names
- [ ] Final silhouette score reported honestly (small datasets like this rarely exceed ~0.5–0.6 - don't overclaim)
