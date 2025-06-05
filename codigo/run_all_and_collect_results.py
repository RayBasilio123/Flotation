import argparse, sys, pandas as pd, numpy as np
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

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

def load_data(paths):
    dfs = [pd.read_csv(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)

    if 'cluster_label' in df.columns:
        df.rename(columns={'cluster_label': 'cluster'}, inplace=True)
    if 'cluster' not in df.columns:
        print("Erro: coluna 'cluster' não encontrada.", file=sys.stderr)
        sys.exit(1)

    empty_cols = df.columns[df.isna().all()].tolist()
    if empty_cols:
        print(f"Removendo colunas sem valores: {empty_cols}")
        df = df.drop(columns=empty_cols)

    return df

def build_preprocessor(X):
    num_cols = X.select_dtypes('number').columns.tolist()
    if 'cluster' in num_cols:
        num_cols.remove('cluster')
    cat_cols = ['cluster']
    pre = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])
    return pre, num_cols, cat_cols

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', nargs='+', help='um ou mais CSVs com coluna cluster')
    parser.add_argument('--target', default='conc_silica', help='Variável alvo')
    parser.add_argument('--output', help='Arquivo para salvar CSV com resultados (opcional)')
    args = parser.parse_args()

    df = load_data(args.csv)
    df = df.dropna(subset=[args.target])
    y = df[args.target]
    X = df.drop(columns=[args.target])
    if 'conc_fe' in X.columns:
        X = X.drop(columns=['conc_fe'])

    pre, num_cols, cat_cols = build_preprocessor(X)

    models = {
        'RandomForest': RandomForestRegressor(n_estimators=300, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
    }
    if HAS_XGB:
        models['XGBoost'] = XGBRegressor(n_estimators=400, learning_rate=0.05,
                                         subsample=0.8, colsample_bytree=0.8,
                                         random_state=42)

    results = []
    for name, mdl in models.items():
        pipe = Pipeline([('prep', pre), ('mdl', mdl)])
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
        pipe.fit(Xtr, ytr)
        ypred = pipe.predict(Xte)
        results.append({
            'Modelo': name,
            'MSE': mean_squared_error(yte, ypred),
            'R²': r2_score(yte, ypred),
            'MAPE': mape(yte, ypred)
        })

    res_df = pd.DataFrame(results).sort_values('MAPE')

    if args.output:
        res_df.to_csv(args.output, index=False)
    else:
        print("\nResultados (teste 30%)")
        print(res_df.to_string(index=False))

if __name__ == '__main__':
    main()
