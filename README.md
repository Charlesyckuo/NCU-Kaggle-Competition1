# Kaggle Competition 1

This repository contains the code and data used for a Kaggle competition as part of a master's course. The project focuses on multi-target prediction using CatBoost and XGBoost, along with feature analysis using PCA. Below is the detailed project structure, requirements, and instructions for running the code.

---

## Project Structure

```
├── Final_Code_and_data                  # Main folder containing code and data for final submission
│   ├── train_data.csv                   # Training dataset
│   ├── test_data.csv                    # Testing dataset
│   ├── sample_submission.csv            # Sample submission format for Kaggle
│   ├── cat_submission.csv               # CatBoost model predictions
│   ├── final_submission.csv             # Final submission combining CatBoost and XGBoost predictions
│   ├── cat.ipynb                        # Notebook for CatBoost training and predictions
│   ├── xgboost.ipynb                    # Notebook for XGBoost training and predictions
├── Other_Code_logs_png                  # Folder with additional scripts, logs, and visualizations
│   ├── catboost_optuna.py               # Script for optimizing CatBoost parameters using Optuna
│   ├── other.ipynb                      # Miscellaneous analysis notebook
│   ├── pca.ipynb                        # Notebook for PCA-based feature analysis
│   ├── catboost_bayesian_logs           # Folder for CatBoost Bayesian optimization logs
│   │   ├── gender_best_params.csv       # Best parameters for the 'gender' target
│   │   ├── hold_best_params.csv         # Best parameters for the 'hold racket handed' target
│   │   ├── level_best_params.csv        # Best parameters for the 'level' target
│   │   ├── play_years_best_params.csv   # Best parameters for the 'play years' target
│   │   ├── gender_submission.csv        # Submission results for the 'gender' target
│   │   ├── hold_submission.csv          # Submission results for the 'hold racket handed' target
│   │   ├── level_submission.csv         # Submission results for the 'level' target
│   │   ├── play_years_submission.csv    # Submission results for the 'play years' target
│   ├── PCA_feature_find                 # PCA projection images
│   │   ├── gender
│   │   │   └── output.png               # Gender target 2D projection
│   │   ├── hold
│   │   │   └── output.png               # Hold racket handed target 2D projection
│   │   ├── level
│   │   │   └── output.png               # Level target 2D projection
│   │   ├── play_years
│   │       └── output.png               # Play years target 2D projection
│   ├── xgboost_optuna_visualizations    # XGBoost Optuna optimization visualizations
│   │   ├── gender
│   │   │   ├── target_optimization_history.png
│   │   │   ├── target_parallel_coordinate.png
│   │   │   ├── target_param_importances.png
│   │   │   ├── target_slice.png
│   │   │   └── best_params.csv
│   │   ├── ... (Similar structure for hold, level, play_years)
```

---

## Requirements

To set up the environment, run the following command:

```bash
conda create -n kaggle_test python=3.10 scikit-learn==1.5.1 numpy==1.26.4 pandas==2.2.2 xgboost==2.1.2 catboost==1.2.3
```

> **Note**: The provided environment allows only the scripts in `Final_Code_and_data` to execute successfully. For other scripts, additional dependencies may be required.

---

## Instructions for Running the Code

### Submitting Final Results

1. Navigate to the `Final_Code_and_data` folder and ensure the correct kernel (`kaggle_test`) is selected.
2. Run `cat.ipynb` to:
   - Train four CatBoost models for the four targets.
   - Save the predictions in `cat_submission.csv`.
3. Run `xgboost.ipynb` to:
   - Train four XGBoost models for the four targets.
   - Replace the `level` predictions with those from `cat_submission.csv`.
   - Save the final predictions in `final_submission.csv`.
4. Submit `final_submission.csv` to Kaggle.

### Detailed Code Instructions

#### PCA Feature Analysis

- Run `pca.ipynb` to:
  - Add candidate features in `selected_features`.
  - Balance datasets by adjusting the `target_column` in `balance_dataset`.
  - Analyze feature combinations by setting `num_features` in `analyze_random_feature_combination`.

#### CatBoost-Optuna Parameter Optimization

- Run `catboost_optuna.py` to:
  - Use Optuna for automated parameter tuning.
  - Specify the target in `y_train_full`.
  - Configure trials and timeout in `study.optimize`.
  - Output best parameters and predictions in `catboost_bayesian_logs`.

#### CatBoost Training and Prediction

- Run `cat.ipynb` to:
  - Train models for the four targets using CatBoost.
  - Use parameters from `catboost_optuna.py` for optimal performance.
  - Save predictions in `cat_submission.csv`.

#### XGBoost Training and Prediction

- Run `xgboost.ipynb` to:
  - Remove features with importance < 0.01.
  - Train classifiers for the four targets.
  - Combine predictions from CatBoost and XGBoost to create `final_submission.csv`.

#### Other Scripts

- Run `other.ipynb` to:
  - Experiment with SMOTE for handling imbalanced data.
  - Visualize feature importance and perform random search for XGBoost.

---

## Notes

- Most scripts and data are intermediate outputs; only `Final_Code_and_data` is related to the final submission.

