import pandas as pd
from sklearn.model_selection import KFold, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import numpy as np

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

X_train= train_data.drop(columns=['data_ID', 'player_ID', 'gender', 'play years', 'hold racket handed', 'level'])
X_test = test_data.drop(columns=['data_ID'])

y_play_years = train_data['play years']
y_level = train_data['level']
y_gender = train_data['gender']
y_hold= train_data['hold racket handed']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

param = {
    'n_estimators': 150,
    'max_depth': 5,
    'learning_rate': 0.06,
    'eval_metric': 'logloss',
    'subsample': 0.75,
    'colsample_bytree': 0.8,
    'min_child_weight': 2,
    'gamma': 0.2,
    'random_state': 42
}

play_years_model = XGBClassifier(**param, n_jobs = -1)
play_years_model.fit(X_scaled, y_play_years, verbose=100)

y_pred_years = play_years_model.predict_proba(X_test_scaled)

level_model = XGBClassifier(**param,n_jobs=-1) 
level_model.fit(X_scaled, y_level, verbose=100)

y_pred_level = level_model.predict_proba(X_test_scaled)

gender_model = XGBClassifier(**param, n_jobs = -1) 
gender_model.fit(X_scaled, y_gender, verbose=True)

y_pred_gender = gender_model.predict_proba(X_test_scaled)

hold_model = XGBClassifier(**param, n_jobs=-1)
hold_model.fit(X_scaled, y_hold, verbose=True)

y_pred_hold = hold_model.predict_proba(X_test_scaled)

submission = pd.DataFrame({
    'data_ID': test_data['data_ID'],
    'gender': y_pred_gender[:, 1],
    'hold racket handed': y_pred_hold[:, 1],
    'play years_0': y_pred_years[:, 0],
    'play years_1': y_pred_years[:, 1],
    'play years_2': y_pred_years[:, 2],
    'level_0': y_pred_level[:, 0],
    'level_1': y_pred_level[:, 1],
    'level_2': y_pred_level[:, 2]
})

# 儲存提交檔案
submission.to_csv('submission.csv', index=False)