# Information Retrieval System - Collaborative Filtering Assignment

## Team Members
- **Noureldeen Maher Mesbah** - 221101140
- **Youssef Zakaria Soubhi Abo Srewa** - 221101030
- **Youssef Mohamed** - 221101573

---

## Dataset

**Dianping Social Recommendation Dataset (2015)**

📥 **Download Link:** [https://lihui.info/file/Dianping_SocialRec_2015.tar.bz2](https://lihui.info/file/Dianping_SocialRec_2015.tar.bz2)

After downloading, extract the dataset to:
```
./dataset/Dianping_SocialRec_2015/rating.txt
```

---

## Installation

1. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Project Structure

```
irs_ass/
├── dataset/
│   └── Dianping_SocialRec_2015/
│       └── rating.txt
├── results/                          # Output files and plots
├── utils/
│   ├── clustering.py                 # K-Means, StandardScaler, metrics
│   ├── data_loader.py                # Data loading utilities
│   ├── similarity.py                 # Similarity functions
│   └── prediction.py                 # Prediction functions
├── section1_statistical_analysis/
│   └── section1_statistical_analysis.py
├── section2_neighborhood_cf/
│   ├── part1_user_based_cf/
│   │   └── part1_user_based_cf.py
│   └── part2_item_based_cf/
│       └── part2_item_based_cf.py
├── section3_clustering_based_cf/
│   ├── part1_user_clustering_avg_ratings/
│   │   └── part1_user_clustering_avg_ratings.py
│   ├── part2_user_clustering_common_ratings/
│   │   └── part2_user_clustering_common_ratings.py
│   ├── part3_item_clustering_avg_raters/
│   │   └── part3_item_clustering.py
│   └── part4_cold_start_clustering/
│       └── part4_cold_start_clustering.py
├── requirements.txt
└── README.md
```

---

## Running Sequence

Execute the files in the following order:

### Step 1: Statistical Analysis
```bash
cd section1_statistical_analysis
python section1_statistical_analysis.py
cd ..
```
This generates the preprocessed dataset and computes statistics.

---

### Step 2: Neighborhood-Based CF

#### Part 1: User-Based CF
```bash
cd section2_neighborhood_cf/part1_user_based_cf
python part1_user_based_cf.py
cd ../..
```

#### Part 2: Item-Based CF
```bash
cd section2_neighborhood_cf/part2_item_based_cf
python part2_item_based_cf.py
cd ../..
```

---

### Step 3: Clustering-Based CF

#### Part 1: User Clustering (Average Ratings)
```bash
cd section3_clustering_based_cf/part1_user_clustering_avg_ratings
python part1_user_clustering_avg_ratings.py
cd ../..
```

#### Part 2: User Clustering (Common Ratings)
```bash
cd section3_clustering_based_cf/part2_user_clustering_common_ratings
python part2_user_clustering_common_ratings.py
cd ../..
```

#### Part 3: Item Clustering
```bash
cd section3_clustering_based_cf/part3_item_clustering_avg_raters
python part3_item_clustering.py
cd ../..
```

#### Part 4: Cold-Start Problem
```bash
cd section3_clustering_based_cf/part4_cold_start_clustering
python part4_cold_start_clustering.py
cd ../..
```

---

## Output

All results are saved in the `results/` directory:
- CSV files with computed statistics
- PNG plots for visualizations
- TXT files with analysis summaries

---

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- tqdm