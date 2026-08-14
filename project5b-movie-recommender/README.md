# 🎬 Movie Recommender System with MovieLens 100K
**🤖 Classical Machine Learning · 🎯 Recommender Systems · ⭐ 100,000 Ratings**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.2.2-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.26.4-013243?style=flat&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.0-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-1.13.1-8CAAE6?style=flat&logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8.4-11557C?style=flat&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13.2-3776AB?style=flat)
![JupyterLab](https://img.shields.io/badge/JupyterLab-4.2.0-F37626?style=flat&logo=jupyter&logoColor=white)

---

## 📚 Part of a practical **AI & Machine Learning Workspace**

✨ *A detailed, real-world junior-level task for this repository, included in the full guide* 👉 [AI & Machine Learning Workspace](https://codeaiflow.cloud/b/ai-machine-learning-workspace)

---

# Project 5, Option B - Movie Recommender System (MovieLens 100K)

## Dataset - verified real source

- **Name:** MovieLens 100K (`ml-100k`)
- **Publisher:** GroupLens Research, University of Minnesota
- **Official page:** https://grouplens.org/datasets/movielens/100k/
- **Direct download:** https://files.grouplens.org/datasets/movielens/ml-100k.zip
- **Verified facts** (cross-checked across GroupLens' own page, the official README, and multiple independent academic citations of this exact dataset):
  - 100,000 ratings (1–5 stars) from **943 users** on **1,682 movies**.
  - Collected September 19, 1997 – April 22, 1998.
  - Every included user rated at least 20 movies (the dataset was pre-cleaned this way).
  - Key files: `u.data` (tab-separated: `user_id, item_id, rating, timestamp`), `u.item` (pipe-separated movie metadata including a 19-column one-hot genre encoding), `u.user` (demographics: age, gender, occupation, zip).
- **License / citation requirement:** GroupLens requires citation in any published use. Cite:
  > F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages. DOI: http://dx.doi.org/10.1145/2827872
  Read the full `README` bundled in the zip for GroupLens' terms of use before any public redistribution of derived results.


## How to download the data
```bash
mkdir -p data
curl -O https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip -d data/
mv ml-100k.zip data/  # or delete it, your call
```
This produces `data/ml-100k/u.data`, `data/ml-100k/u.item`, etc. - `src/` code expects this exact path (`data/ml-100k/`).

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
jupyter lab notebooks/recommender.ipynb
```

## Repository structure
```
project3b-movie-recommender/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── ml-100k/                # you place this here - not committed, see above
├── notebooks/
│   └── recommender.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── content_based.py
│   ├── collaborative.py
│   └── evaluation.py
└── tests/
    └── test_recommender.py     # runs against a small synthetic ratings set, proves the logic works
```

## Why sparse, not dense
943 × 1682 = ~1.59M cells, only ~100K filled (~6.3% density). This is small enough to densify without crashing on a laptop, but the code is written using `scipy.sparse` deliberately - it's the correct pattern for this problem class, and the habit matters more than whether this specific dataset technically requires it. If you densify instead, say so explicitly in the notebook and note the memory trade-off.

## Deliverables checklist
- [ ] `u.data` and `u.item` loaded, ratings matrix built as a sparse matrix
- [ ] Content-based filtering implemented (genre-vector cosine similarity)
- [ ] Collaborative filtering implemented (item-based CF or SVD-based) and compared to content-based
- [ ] Hybrid blend implemented and shown to beat each method alone on at least one metric
- [ ] Cold-start behavior explicitly demonstrated for a zero-rating user
- [ ] precision@10 / recall@10 reported on a held-out split (not just RMSE on predicted ratings)
- [ ] GroupLens citation included in any write-up derived from this dataset
