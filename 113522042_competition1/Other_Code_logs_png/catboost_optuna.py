import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
import optuna
import os

# 讀取數據
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 分離特徵與目標
X_train_full = train_data.drop(columns=['data_ID', 'player_ID', 'gender', 'play years', 'hold racket handed', 'level'])
y_train_full = train_data['hold racket handed']
X_test = test_data.drop(columns=['data_ID'])

# 保存結果用的變數
best_params = {}
output_dir = "cat_cat"
os.makedirs(output_dir, exist_ok=True)


unique_classes, counts = np.unique(y_train_full, return_counts=True)
total_samples = len(y_train_full)
num_classes = len(unique_classes)
class_weights = [total_samples / (num_classes * count) for count in counts]
catboost_class_weights = class_weights

print("Class weights:", catboost_class_weights)

# 定義貝葉斯優化的目標函數
def objective(trial):
    # 定義參數空間
    catboost_params = {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "depth": trial.suggest_int("depth", 4, 12),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
        "random_strength": trial.suggest_float("random_strength", 1, 20),
        "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise"]),
        "loss_function": "MultiClass",
        "task_type": "GPU",
        "devices": "0",
        "class_weights": catboost_class_weights,
        "verbose": 0,
    }
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kf.split(X_train_full, y_train_full):
        # 拆分訓練集和驗證集
        X_train, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
        y_train, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

        # 標準化（只基於訓練集）
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # 模型訓練
        model = CatBoostClassifier(**catboost_params)
        model.fit(X_train_scaled, y_train, eval_set=(X_val_scaled, y_val), verbose=0)
        
        # 預測與計算 AUC
        y_pred_proba = model.predict_proba(X_val_scaled)
        score = roc_auc_score(pd.get_dummies(y_val), y_pred_proba, multi_class="ovr", average="micro")
        scores.append(score)
    
    # 返回平均分數
    return np.mean(scores)

# 使用 Optuna 進行貝葉斯優化
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, timeout=3600)

# 儲存最佳參數
best_params = study.best_params
print("Best parameters:", best_params)

# 移除不相關的鍵
#del best_params["best_score"]

# 使用最佳參數進行最終模型訓練
final_model = CatBoostClassifier(**best_params, task_type="GPU", devices="0", verbose=100)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
final_model.fit(X_train_scaled, y_train_full)

# 測試集預測
X_test_scaled = scaler.fit_transform(X_test)
y_test_pred_proba = final_model.predict_proba(X_test_scaled)

# 保存預測結果
submission = pd.DataFrame(y_test_pred_proba, columns=['hold racket handed'])
submission.to_csv(f"{output_dir}/hold_racket_handed.csv", index=False)
