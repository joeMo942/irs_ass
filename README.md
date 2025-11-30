# Information Retrieval Systems - Assignment

This repository contains the implementation of various recommendation system techniques for the IRS assignment.

## Folder Structure

```
├── dataset/                          # Dataset files
├── section1_statistical_analysis/    # Statistical analysis notebooks
├── section2_neighborhood_cf/         # Neighborhood-based Collaborative Filtering
│   ├── part1_user_based_cf/         # User-based CF implementation
│   └── part2_item_based_cf/         # Item-based CF implementation
├── section3_clustering_based_cf/     # Clustering-based Collaborative Filtering
│   ├── part1_user_clustering_avg_ratings/      # User clustering with average ratings
│   ├── part2_user_clustering_common_ratings/   # User clustering with common ratings
│   ├── part3_item_clustering_avg_raters/       # Item clustering with average raters
│   └── part4_cold_start_clustering/            # Cold start problem solution
├── utils/                            # Helper functions and utilities
└── results/                          # Output files and visualizations
```

## Sections

### Section 1: Statistical Analysis
- Exploratory data analysis
- Dataset statistics and insights
- Data visualization

### Section 2: Neighborhood Collaborative Filtering
- **Part 1**: User-based Collaborative Filtering
- **Part 2**: Item-based Collaborative Filtering

### Section 3: Clustering-based Collaborative Filtering
- **Part 1**: User clustering with average ratings
- **Part 2**: User clustering with common ratings
- **Part 3**: Item clustering with average raters
- **Part 4**: Cold start clustering approach

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place your dataset files in the `dataset/` folder

3. Run notebooks in order, starting from Section 1

## Requirements

See `requirements.txt` for a complete list of dependencies.

## Results

All output files, visualizations, and model results are stored in the `results/` folder.