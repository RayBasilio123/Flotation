"""
compare_models_with_clusters.py
-----------------------------------
Treina 3-4 modelos (Random Forest, Gradient Boosting, XGBoost*, Linear),
usando a coluna 'cluster' como feature (one-hot), calcula MSE, R², MAPE
e GERA GRÁFICOS de valor real × predito ao longo do tempo.

Uso típico (Windows CMD ou PowerShell - tudo em UMA linha):

python compare_models_with_clusters.py ^
  cluster_outputs\data_dbscan_ago_set_with_cluster.csv ^
  cluster_outputs\data_dbscan_mar_jun_with_cluster.csv ^
  cluster_outputs\data_kmeans_ago_set_with_cluster.csv ^
  cluster_outputs\data_kmeans_mar_jun_with_cluster.csv ^
  --target conc_silica ^
  --datecol data ^
  --output resultados.csv

*XGBoost será pulado se a biblioteca não estiver instalada.
"""

import argparse, sys, os, pathlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # backend não-interativo
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ---------- utilidades ----------------------------------------------------- #
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

def load_data(csv_path):
    df = pd.read_csv(csv_path)

    if 'cluster_label' in df.columns:
        df = df.rename(columns={'cluster_label': 'cluster'})

    if 'cluster' not in df.columns:
        print(f"Erro: coluna 'cluster' não encontrada em {csv_path}", file=sys.stderr)
        sys.exit(1)

    # remover colunas totalmente vazias
    empty_cols = df.columns[df.isna().all()].tolist()
    if empty_cols:
        print(f"  • removendo colunas sem valores: {empty_cols}")
        df = df.drop(columns=empty_cols)

    return df

def build_preprocessor(X, datecol):
    num_cols = X.select_dtypes('number').columns.tolist()
    if 'cluster' in num_cols:
        num_cols.remove('cluster')
    cat_cols = ['cluster']

    # garantir que a coluna de data não entre no modelo
    if datecol in num_cols:
        num_cols.remove(datecol)

    pre = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])
    return pre

def extract_info_from_filename(filename):
    fname = pathlib.Path(filename).name.lower()
    cluster_method = 'dbscan' if 'dbscan' in fname else 'kmeans' if 'kmeans' in fname else 'desconhecido'
    periodo = 'ago_set' if 'ago_set' in fname else 'mar_jun' if 'mar_jun' in fname else 'desconhecido'
    return cluster_method.upper(), periodo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', nargs='+', help='um ou mais CSVs com coluna cluster')
    parser.add_argument('--target', required=True, help='Variável alvo (y)')
    parser.add_argument('--datecol', required=True, help='Coluna de data para o eixo X dos gráficos')
    parser.add_argument('--output', help='Caminho do CSV para salvar os resultados')
    args = parser.parse_args()

    # modelos
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=300, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
    }
    if HAS_XGB:
        models['XGBoost'] = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

    # garantir pasta de plots
    plots_dir = pathlib.Path("plots")
    plots_dir.mkdir(exist_ok=True)

    results = []

    for csv_path in args.csv:
        print(f"\nProcessando: {csv_path}")
        df = load_data(csv_path)

        # converter data
        if args.datecol not in df.columns:
            print(f"Erro: coluna de data '{args.datecol}' não existe em {csv_path}", file=sys.stderr)
            sys.exit(1)
        df[args.datecol] = pd.to_datetime(df[args.datecol], errors='coerce')

        # descartar linhas sem target ou sem data
        df = df.dropna(subset=[args.target, args.datecol])

        y = df[args.target]
        dates = df[args.datecol]

        # X sem target, sem data
        X = df.drop(columns=[args.target, args.datecol])
        if 'conc_fe' in X.columns:
            X = X.drop(columns=['conc_fe'])
        cluster_method, periodo = extract_info_from_filename(csv_path)
        pre = build_preprocessor(X, args.datecol)

        for model_name, mdl in models.items():
            pipe = Pipeline([('prep', pre), ('mdl', mdl)])
            Xtr, Xte, ytr, yte, dates_tr, dates_te = train_test_split(
                X, y, dates, test_size=0.3, random_state=42, shuffle=True
            )
            pipe.fit(Xtr, ytr)
            ypred = pipe.predict(Xte)

            # métricas
            results.append({
                'Modelo': model_name,
                'Clusterização': cluster_method,
                'Período': periodo,
                'MSE': mean_squared_error(yte, ypred),
                'R²': r2_score(yte, ypred),
                'MAPE': mape(yte, ypred)
            })

            # -------- gráfico valor real × predito ------------------------- #
            plot_df = pd.DataFrame({
                'Data': dates_te,
                'Real': yte,
                'Predito': ypred
            }).sort_values('Data')

            # limitar a uma semana
            start_date = plot_df['Data'].min()
            end_date = start_date + pd.Timedelta(days=7)
            plot_df = plot_df[(plot_df['Data'] >= start_date) & (plot_df['Data'] < end_date)]

            if plot_df.empty:
                print(f"  • [AVISO] Nenhum dado dentro do intervalo de uma semana para gráfico.")
                continue

            plt.figure(figsize=(10, 4))
            plt.plot(plot_df['Data'], plot_df['Real'], label='Real')
            plt.plot(plot_df['Data'], plot_df['Predito'], label='Predito')
            plt.title(f"{model_name} | {cluster_method} | {periodo} (1 semana)")
            plt.xlabel('Data')
            plt.ylabel(args.target)
            plt.legend()
            plt.tight_layout()

            plot_name = f"plot_{cluster_method}_{periodo}_{model_name}.png"
            plt.savefig(plots_dir / plot_name, dpi=120)
            plt.close()
            print(f"  • gráfico salvo em plots/{plot_name}")
    # --------- salvar/mostrar tabela final -------------------------------- #
    res_df = pd.DataFrame(results).sort_values(['Modelo', 'MAPE'])

    if args.output:
        res_df.to_csv(args.output, index=False)
        print(f"\nResultados salvos em: {args.output}")
    else:
        print("\nResultados (teste 30%)")
        print(res_df.to_string(index=False))

if __name__ == '__main__':
    main()
