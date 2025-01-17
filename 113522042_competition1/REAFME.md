# Kaggle Competition 1 113522042
## Project Structure
- Data Files
    - Final_Code_and_data
        - train_data.csv
        - test_data.csv
        - sample_submission.csv 
        - cat_submission.csv
        - final_submission.csv (最終提交，由預測1和2組成)
        - cat.ipynb (預測1)
        - xgboost.ipynb (預測2)
    - Other_Code_logs_png
        - catboost_optuna.py 
        - other.ipynb
        - pca_ipynb
        - catboost_bayesian_logs
            - gender、hold、level、play_years best_params.csv (各自最佳參數)
            - gender、hold、level、play_years submission.csv (基於最佳參數的訓練預測結果)
        - PCA_feature_find (一堆投影到二維平面的圖片)
            - gender、hold、level、play_years
                - output.png
        - xgboost_optuna_visualizations (視覺化紀錄xgboost自動挖掘參數過程)
            - gender、hold、level、play_years
                - target_optimization_history.png
                - target_parallel_coordinate.png
                - target_param_importances.png
                - target_slice.png
                - best_params.csv
## Requirements
- 輸入以下指令
- conda create -n kaggle_test python=3.10 scikit-learn==1.5.1 numpy==1.26.4 pandas==2.2.2 xgboost==2.1.2 catboost==1.2.3
- 注意!該環境只能讓"Final_Code_and_Data"內的程式正常執行並產出預測，其餘的程式若要執行須自行安裝套件

## 程式碼運行及說明
- 提交最終成果方式
    0. cd Final_Code_and_Data (執行notebook前須先select kernel (kaggle_test))
    1. 執行 "cat.ipynb"，裡面會使用catboost訓練四個模型對四個target進行預測 
    2. 得到catboost的輸出 "cat_submission"
    3. 執行 "xgboost.ipynb", 裡面會訓練四個模型對四個target做預測。接著會替換level的預測為cat_submission中的結果
    4. 得到 "final_submission"，提交上去

- 程式碼詳細說明
    - PCA組成分析
        - 執行 "pca.ipynb"
        - 可以在selected_features加入特徵候選
        - 可以調整balance_dataset中的target_column，選定target進行特徵分析
        - 可以調整analyze_random_feature_combination中的num_features，指定從候選特徵中隨機選取的數目
    - Catboost-optuna
        - 執行 "catboost_optuna.py"
        - 以Catboost模型為基礎，自動化尋找最佳參數組合
        - 可以於y_train_full設定目標target
        - 可以於study.optimize設定要嘗試尋找的trials數以及timeout
        - 該程式會輸出最佳的參數組合及使用該參數進行的訓練與預測結果，如hold_racket_handed.csv
        - 相關資料皆存放於catboost_bayesian_logs資料夾中
    - Catboost
        - 執行 "cat.ipynb"
        - 該份程式碼可以基於catboost進行特徵重要性篩選，有需要可以使用
        - 基於catboost_optuna.py產生之各別最佳參數組合，針對四個target進行訓練及預測
        - 輸出檔盟名為"cat_submission"
    - Xgboost
        - 執行 "xgboost.ipynb"
        - 該份程式會先機於xgboost將特徵重要性小於0.01者刪除
        - 訓練四個classifier並對四個target進行預測
        - 在輸出成final_submission.csv前會先讀入上述catboost所輸出的預測，綜合兩者的輸出結果，提交kaggle上表現最好的
    - Other
        - 執行 "Other.ipynb"
        - 該份程式馬提供了smote資料欠採樣，可以進行嘗試
        - 該份程式馬提供特徵重要性基於不同threshold進行排序及視覺化的結果
        - 該份程式馬提供random_search，應用於xgboost

## 注意事項
- 大部分程式及資料只是過程，最終的提交只與"Final_Code_and_data"有關
- 部分檔案不見了，像是xgboost-optuna、一些紀錄score跟參數的csv以及之前組的包含人工特徵的程式碼，但結果上應沒有顯著差異
- conda環境之前以為不會用掉也刪掉了，因此沒辦法提供yml檔，可以創建環境跟安裝必要的套件來執行基本要求
