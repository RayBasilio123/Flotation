"""
run_2stage_experiments.py
-------------------------
Executa uma grade de experimentos com a estratégia em dois estágios
(utilizando o TwoStageSilicaModel definido no módulo m2stage_silica.py).

Algoritmos testados
===================
* Random Forest  (scikit‑learn)
* XGBoost        (xgboost)
* CatBoost       (catboost)

Para cada algoritmo são avaliadas diferentes configurações de
(n_estimators / iterations, max_depth / depth, learning_rate).

© 2025 — Rodrigo Silva & Assistente IA
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flot_two_stage import TwoStageSilicaModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# TENTATIVA DE IMPORTAR DEPENDÊNCIAS EXTERNAS
# ----------------------------------------------------------------------
try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:  # xgboost não instalado
    XGBClassifier = XGBRegressor = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:  # catboost não instalado
    CatBoostClassifier = CatBoostRegressor = None

# ----------------------------------------------------------------------
# 1) CARREGA DADOS
# ----------------------------------------------------------------------
df_path = Path(
    r"C:\Users\rcpsi\OneDrive\Documents\langchain\Flotation\flotation_prediction\working_df_15_05_2025.csv"
)
df = pd.read_csv(df_path)

TARGET = "conc_silica"
y = df[TARGET]
X = df.drop(columns=[TARGET, "inicio", "Unnamed: 0"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=None
)

# ----------------------------------------------------------------------
# 2) DEFINE GRADE DE HIPERPARÂMETROS
# ----------------------------------------------------------------------
GRID = {
    "rf": [
        # (n_estimators, max_depth)
        (200, 20),
        (200, 2),
        (200, 5),
        (400, 20),
        (400, 2),
        (400, 5),
        (600, 2),
        (600, 5),
        (600, 20),
    ],
    "xgb": [
        # (n_estimators, max_depth, learning_rate)
        (300, 6, 0.05),
        (500, 8, 0.05),
        (700, 10, 0.03),
    ]
    if XGBClassifier is not None
    else [],
    "cat": [
        # (iterations, depth, learning_rate)
        (400, 6, 0.05),
        (600, 8, 0.03),
        (800, 10, 0.03),
    ]
    if CatBoostClassifier is not None
    else [],
}

# ----------------------------------------------------------------------
# 3) FUNÇÕES AUXILIARES
# ----------------------------------------------------------------------
def rmse(a, b):
    return np.sqrt(mean_squared_error(a, b))


def make_estimators(algo: str, hp: tuple):
    """
    Cria (classifier, reg_high, reg_norm) de acordo com o algoritmo/hp.
    """
    if algo == "rf":
        n_estimators, max_depth = hp
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=42,
        )
        reg_high = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=42,
        )
        reg_norm = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=42,
        )
        hp_dict = {"n_estimators": n_estimators, "max_depth": max_depth}

    elif algo == "xgb":
        n_estimators, max_depth, lr = hp
        clf = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=lr,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            objective="binary:logistic",
        )
        reg_high = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=lr,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
        )
        reg_norm = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=lr,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
        )
        hp_dict = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": lr,
        }

    elif algo == "cat":
        iterations, depth, lr = hp
        clf = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=lr,
            loss_function="Logloss",
            random_state=42,
            verbose=False,
        )
        reg_high = CatBoostRegressor(
            iterations=iterations,
            depth=depth,
            learning_rate=lr,
            loss_function="RMSE",
            random_state=42,
            verbose=False,
        )
        reg_norm = CatBoostRegressor(
            iterations=iterations,
            depth=depth,
            learning_rate=lr,
            loss_function="RMSE",
            random_state=42,
            verbose=False,
        )
        hp_dict = {"iterations": iterations, "depth": depth, "learning_rate": lr}

    else:
        raise ValueError(f"Algoritmo '{algo}' não reconhecido.")

    return clf, reg_high, reg_norm, hp_dict


# ----------------------------------------------------------------------
# 4) LOOP DE EXPERIMENTOS
# ----------------------------------------------------------------------
results = []

for algo, hp_list in GRID.items():
    for hp in hp_list:
        clf, reg_h, reg_n, hp_dict = make_estimators(algo, hp)

        model = TwoStageSilicaModel(
            classifier=clf,
            reg_high=reg_h,
            reg_norm=reg_n,
            high_quantile=0.9,
            random_state=42,
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        res = {
            "algo": algo,
            **hp_dict,
            "rmse": rmse(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
            "mape": mean_absolute_percentage_error(y_test, y_pred)
        }
        results.append(res)

        print(
            f"[{algo.upper()}] {json.dumps(hp_dict)} "
            f"→ RMSE={res['rmse']:.4f} | MAE={res['mae']:.4f} | R2={res['r2']:.4f} | MAPE={res['mape']:.4f}"
        )

# ----------------------------------------------------------------------
# 5) RESULTADOS ORDENADOS POR RMSE
# ----------------------------------------------------------------------
df_res = pd.DataFrame(results).sort_values("rmse")
print("\nTOP 10 MODELOS (menor RMSE):")
print(df_res.head(10).to_string(index=False))

# ----------------------------------------------------------------------
# 6) (OPCIONAL) SALVA MELHOR MODELO
# ----------------------------------------------------------------------
best_row = df_res.iloc[0]
best_algo = best_row["algo"]
best_hp = {
    k: best_row[k]
    for k in best_row.index
    if k not in {"algo", "rmse", "mae", "r2"}
}

clf, reg_h, reg_n, _ = make_estimators(best_algo, tuple(best_hp.values()))
best_model = TwoStageSilicaModel(
    classifier=clf,
    reg_high=reg_h,
    reg_norm=reg_n,
    high_quantile=0.9,
    random_state=42,
).fit(X, y)

joblib.dump(best_model, "two_stage_silica_best.pkl")
print("\nMelhor modelo salvo em two_stage_silica_best.pkl")
