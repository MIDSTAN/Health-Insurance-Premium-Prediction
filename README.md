Health Insurance Premium Prediction: Scratch ML Models
Overview
This project implements three machine learning regression models from scratch (or hybrid where necessary) to predict health insurance premiums using the Insurance Dataset. The dataset includes features like age, sex, BMI, number of children, smoker status, and region to predict medical expenses.
Key Features

Linear Regression from Scratch: Uses the normal equation for closed-form solution.
Gradient Boosting Regressor: Sequential ensemble of decision trees fitted to residuals (uses scikit-learn trees for simplicity).
Random Forest Regressor from Scratch: Bagging with random feature subsets and custom decision trees using MSE splits.
Interactive Prediction: User-friendly CLI for inputting features and getting predictions from all models.
Evaluation: Manual 80/20 train-test split with MAE and R² metrics.
No External ML Libs for Core Logic: Minimal dependencies; core algorithms coded manually.

The project achieves competitive performance: GB (R² ≈ 0.86), RF (R² ≈ 0.85), LR (R² ≈ 0.75) on test sets.
Dataset

Source: Kaggle Insurance Dataset (converted to numerical CSV).
Features: age (int), sex (0/1), bmi (float), children (int), smoker (0/1), region (1-4).
Target: expenses (float, medical charges).
Download: Place insurance_converted.csv in /Dataset/ folder.

Project Structure
Health Insurance Premium/
├── Model/
│   ├── Converting Dataset for Training/
│   │   └── Dataset/
│   │       └── insurance_converted.csv
│   ├── Training/
│   │   ├── linearRegression_scratch.py
│   │   ├── gradientBoosting.py
│   │   └── randomForest_scratch.py
│   └── Testing/
│       └── test.py  # Unified tester
└── README.md
