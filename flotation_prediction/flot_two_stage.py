"""
m2stage_silica.py
-----------------
Módulo para previsão da concentração de sílica em dois estágios
(classificação + regressão), de forma a facilitar a troca
tanto do classificador quanto dos regressores.

© 2025 — Rodrigo Silva & Assistente IA
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split


class TwoStageSilicaModel(BaseEstimator, RegressorMixin):
    """
    Estratégia em dois estágios para prever `conc_silica`:

    1) Classificador binário identifica se o ponto será "ALTO"
       (conc_silica > quantil de referência, default p90) ou "NORMAL".
    2) Dois regresssores especializados estimam o valor numérico:
       - `reg_high_` para os casos "ALTO".
       - `reg_norm_` para os casos "NORMAL".
    """

    def __init__(
        self,
        classifier,
        reg_high,
        reg_norm,
        high_quantile: float = 0.9,
        random_state: int | None = None,
    ):
        """
        Parameters
        ----------
        classifier : estimador sklearn
            Algoritmo binário que deve implementar fit/predict (e preferencialmente predict_proba).

        reg_high : estimador sklearn
            Regressor para os pontos classificados como "ALTO".

        reg_norm : estimador sklearn
            Regressor para os pontos classificados como "NORMAL".

        high_quantile : float, default 0.9
            Quantil usado como limiar para definir "ALTO".

        random_state : int | None
            Para reprodutibilidade (quando aplicável nos estimadores).
        """
        self.classifier = classifier
        self.reg_high = reg_high
        self.reg_norm = reg_norm
        self.high_quantile = high_quantile
        self.random_state = random_state

    # ---------------------------------------------------------------------
    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray):
        """
        Treina os três modelos:
        - classificador
        - regressor para casos altos
        - regressor para casos normais
        """
        # Garantir arrays numpy
        X = np.asarray(X)
        y = np.asarray(y).ravel()

        # 1) Define limiar de "ALTO"
        self.threshold_ = np.quantile(y, self.high_quantile)

        # 2) Gera rótulo binário
        y_bin = (y > self.threshold_).astype(int)

        # 3) Clona estimadores para evitar efeitos colaterais
        clf = clone(self.classifier)
        reg_h = clone(self.reg_high)
        reg_n = clone(self.reg_norm)

        # 4) Treina classificador
        clf.fit(X, y_bin)

        # 5) Separa subconjuntos
        X_high, y_high = X[y_bin == 1], y[y_bin == 1]
        X_norm, y_norm = X[y_bin == 0], y[y_bin == 0]

        # 6) Treina regressões
        if len(y_high) == 0 or len(y_norm) == 0:
            raise ValueError(
                "Não há exemplos suficientes em uma das classes após a divisão pelo quantil."
            )

        reg_h.fit(X_high, y_high)
        reg_n.fit(X_norm, y_norm)

        # 7) Guarda modelos treinados
        self.classifier_ = clf
        self.reg_high_ = reg_h
        self.reg_norm_ = reg_n
        return self

    # ---------------------------------------------------------------------
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Prediz conc_silica usando:
        - classificador para escolher qual regressor aplicar.
        """
        X = np.asarray(X)

        # Predição binária
        y_bin_pred = self.classifier_.predict(X)

        # Predição numérica condicionada
        y_pred_high = self.reg_high_.predict(X)
        y_pred_norm = self.reg_norm_.predict(X)

        # Combina de acordo com o classificador
        return np.where(y_bin_pred == 1, y_pred_high, y_pred_norm)

    # ---------------------------------------------------------------------
    def score(self, X, y, metric="rmse") -> float:
        """
        Avalia desempenho com métrica escolhida.
        Métricas implementadas: 'rmse', 'mae', 'r2'.
        """
        y = np.asarray(y).ravel()
        y_hat = self.predict(X)

        if metric == "rmse":
            return np.sqrt(mean_squared_error(y, y_hat))
        if metric == "mae":
            return mean_absolute_error(y, y_hat)
        if metric == "r2":
            return r2_score(y, y_hat)
        raise ValueError("Métrica desconhecida. Escolha 'rmse', 'mae' ou 'r2'.")


# =============================================================================
# Exemplo de uso
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    # -----------------------------------------------------------------
    # 1) Carrega dados (ajuste o path conforme necessário)
    df_path = Path(r"C:\Users\rcpsi\OneDrive\Documents\langchain\Flotation\flotation_prediction\working_df_15_05_2025.csv")
    df = pd.read_csv(df_path)

    # Define target
    TARGET = "conc_silica"
    y = df[TARGET]
    X = df.drop(columns=[TARGET,'inicio','Unnamed: 0'])

    print(X.columns)

    # -----------------------------------------------------------------
    # 2) Divide em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )

    # -----------------------------------------------------------------
    # 3) Define modelos (fácil de trocar!)
    clf = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=300, random_state=42)),
        ]
    )

    reg_high = RandomForestRegressor(
        n_estimators=300, max_depth=None, random_state=42
    )
    reg_norm = RandomForestRegressor(
        n_estimators=300, max_depth=None, random_state=42
    )

    # -----------------------------------------------------------------
    # 4) Treina modelo em dois estágios
    model = TwoStageSilicaModel(
        classifier=clf,
        reg_high=reg_high,
        reg_norm=reg_norm,
        high_quantile=0.9,
        random_state=42,
    )

    model.fit(X_train, y_train)

    # -----------------------------------------------------------------
    # 5) Avaliação rápida
    print("RMSE (test):", model.score(X_test, y_test, metric="rmse"))
    print("MAE  (test):", model.score(X_test, y_test, metric="mae"))
    print("R²   (test):", model.score(X_test, y_test, metric="r2"))

    # Opcional: salvar modelo treinado com joblib/pickle.
