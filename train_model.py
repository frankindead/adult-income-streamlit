import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

DATA_PATH = "/Users/frankindead/Desktop/Python_2025_for_andan/Home Assignments/data.adult.csv"
TARGET_COL = ">50K,<=50K"

def main():
    df = pd.read_csv(DATA_PATH)

    df = df.replace("?", np.nan).dropna() # заменяем ? на np.nan и удаляем пропуски

    y = df[TARGET_COL].replace({">50K": 1, "<=50K": 0}).astype(int) # целевая переменная
    X = df.drop(columns=TARGET_COL)

    cat_features = X.select_dtypes(include=["object"]).columns.tolist() # типы признаков
    num_features = X.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )

    model = GradientBoostingClassifier( # выбрали лучшую модель для приложения
        n_estimators=88,     # определяем выявленные лучшие гиперпараметры модели
        max_features=None, 
        criterion="friedman_mse", 
        random_state=42
    )

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ]) # определяем пайплайн

    pipe.fit(X, y) # фиттим под наши данные

    Path("models").mkdir(exist_ok=True)
    joblib.dump(pipe, "models/model.joblib") # сохраняем модель в папку models

    meta = {
        "target_col": TARGET_COL,
        "num_features": num_features,
        "cat_features": cat_features,
        "all_columns": X.columns.tolist()
    }
    joblib.dump(meta, "models/meta.joblib")

    print("Модель градиентного бустинга сохранена в models/model.joblib")

if __name__ == "__main__":
    main()
