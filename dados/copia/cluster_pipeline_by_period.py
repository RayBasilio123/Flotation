"""cluster_pipeline_by_period.py

Executa clustering (K‑Means / DBSCAN) em dois períodos:
  1. março – junho
  2. agosto – setembro

Detecta automaticamente 'inicio_dt' ou 'inicio' como coluna de data.

Salva:
  • data_<modelo>_<periodo>_with_cluster.csv
  • summary_<modelo>_<periodo>.csv
  • model_<modelo>_<periodo>.joblib
  • info_<modelo>_<periodo>.txt
  • plot_<modelo>_<periodo>.png
  • elbow_<modelo>_<periodo>.png
"""

import argparse, os, textwrap, warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import joblib
from kneed import KneeLocator

def preprocess_csv(path, sep=';', month_range=(3,6)):
    df = pd.read_csv(path, sep=sep)
    if 'inicio_dt' in df.columns:
        date_col = 'inicio_dt'
    elif 'inicio' in df.columns:
        date_col = 'inicio'
    else:
        raise ValueError("Nenhuma coluna de data encontrada ('inicio_dt' ou 'inicio').")
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    ini, fim = month_range
    df = df[df[date_col].dt.month.between(ini, fim)].copy()
    for col in df.columns.difference([date_col, 'fim_dt']):
        if df[col].dtype == object:
            df[col] = (df[col].str.replace(',', '.').str.strip().replace('', np.nan))
        df[col] = pd.to_numeric(df[col], errors='coerce')
    num_df = df.select_dtypes('number').dropna(axis=1, how='all').dropna()
    df = df.loc[num_df.index].copy()
    return df, num_df, date_col

def cluster_summary_stats(df, cluster_col='cluster'):
    num_df = df.select_dtypes('number').copy()
    if cluster_col not in num_df.columns:
        num_df[cluster_col] = df[cluster_col]
    summary = (num_df.groupby(cluster_col)
                     .agg(['count','mean','std','median'])
                     .round(3)
                     .reset_index())
    summary.columns = [
        col if col[1]=='' else f"{col[0]}_{col[1]}"
        for col in summary.columns
    ]
    return summary

def plot_clusters(num_df, labels, title, output_path=None, scale=True, noise=-1):
    X = StandardScaler().fit_transform(num_df) if scale else num_df.values
    pcs = PCA(n_components=2, random_state=42).fit_transform(X)
    cmap = plt.colormaps.get_cmap('tab10')
    colors = ['#808080' if lab == noise else cmap(lab % 10) for lab in labels]
    plt.figure(figsize=(8,6))
    plt.scatter(pcs[:,0], pcs[:,1], c=colors, edgecolors='k', s=30, alpha=.8)
    plt.xlabel('PC1'); plt.ylabel('PC2')
    plt.title(title); plt.grid(ls='--', alpha=.3); plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        print('Saved plot to', output_path)
    plt.show()

def plot_elbow_kmeans(num_df, out_path='elbow_kmeans.png', max_k=10):
    X = StandardScaler().fit_transform(num_df)
    inertias = []
    ks = range(1, max_k+1)
    for k in ks:
        model = KMeans(n_clusters=k, random_state=42).fit(X)
        inertias.append(model.inertia_)
    plt.figure()
    plt.plot(ks, inertias, 'bo-')
    plt.xlabel('Número de clusters (k)')
    plt.ylabel('Inércia')
    plt.title('Método do Cotovelo – KMeans')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    print('Saved elbow plot to', out_path)
    plt.show()

def plot_elbow_dbscan(num_df, out_path='elbow_dbscan.png', min_samples=20):
    X = StandardScaler().fit_transform(num_df)
    neigh = NearestNeighbors(n_neighbors=min_samples)
    nbrs = neigh.fit(X)
    distances, _ = nbrs.kneighbors(X)
    k_distances = np.sort(distances[:, -1])
    plt.figure()
    plt.plot(k_distances)
    plt.xlabel('Amostras ordenadas')
    plt.ylabel(f'{min_samples}-ésima menor distância')
    plt.title('Método do Cotovelo – DBSCAN')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    print('Saved elbow plot to', out_path)
    plt.show()

    # Sugerir eps com KneeLocator
    x = np.arange(len(k_distances))
    kneedle = KneeLocator(x, k_distances, curve='convex', direction='increasing')
    eps_sugerido = round(k_distances[kneedle.knee] if kneedle.knee is not None else np.median(k_distances), 3)
    print(f"→ EPS sugerido automaticamente: {eps_sugerido}")
    return eps_sugerido

def save_metadata(path, algo, tag, args, n_clusters=None):
    with open(path, 'w') as f:
        f.write(f"Algoritmo: {algo}\n")
        f.write(f"Período: {tag}\n")
        f.write("Parâmetros:\n")
        if algo == 'kmeans':
            f.write(f"  k: {args.k}\n")
        elif algo == 'dbscan':
            f.write(f"  eps: {args.eps}\n")
            f.write(f"  min_samples: {args.min_samples}\n")
        if n_clusters is not None:
            f.write(f"Número de clusters detectados: {n_clusters}\n")

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""        Clustering separado por período (mar‑jun, ago‑set) com saída de:
          • CSV completo + rótulo de cluster
          • CSV de estatísticas
          • gráfico PCA e gráfico do cotovelo
          • modelo treinado e metadados
        """))
    parser.add_argument('csv', help='arquivo CSV ;')
    parser.add_argument('--algo', choices=['kmeans','dbscan'], default='kmeans')
    parser.add_argument('--k', type=int, default=4, help='k para K-Means')
    parser.add_argument('--eps', type=float, default=None)
    parser.add_argument('--min_samples', type=int, default=20)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--out_dir', default='cluster_outputs')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    periods = [('mar_jun',(3,6)), ('ago_set',(8,9))]

    for tag, rng in periods:
        print(f"\n>>> Período {tag.replace('_',' → ').upper()}")
        df, num_df, date_col = preprocess_csv(args.csv, month_range=rng)

        if df.empty or num_df.empty:
            warnings.warn(f"Nenhum dado válido para o período {tag}."); continue

        print(f"→ Usando coluna de data: {date_col}")
        print(f"→ Registros usados: {len(df)}")
        print(f"→ {date_col} de {df[date_col].min().date()} até {df[date_col].max().date()}")

        elbow_path = os.path.join(args.out_dir, f'elbow_{args.algo}_{tag}.png')
        if args.algo == 'kmeans':
            plot_elbow_kmeans(num_df, out_path=elbow_path)
            eps_value = None
        else:
            eps_sugerido = plot_elbow_dbscan(num_df, out_path=elbow_path, min_samples=args.min_samples)
            eps_value = args.eps if args.eps is not None else eps_sugerido

        X = StandardScaler().fit_transform(num_df)
        if args.algo == 'kmeans':
            model = KMeans(args.k, random_state=42).fit(X)
        else:
            model = DBSCAN(eps=eps_value, min_samples=args.min_samples).fit(X)

        labels = model.labels_
        df['cluster'] = labels

        for col in ['inicio_dt', 'inicio', 'fim_dt']:
            if col not in df.columns:
                df[col] = pd.NaT

        prefix = f'{args.algo}_{tag}'
        df.to_csv(os.path.join(args.out_dir, f'data_{prefix}_with_cluster.csv'), index=False)
        summary = cluster_summary_stats(df)
        summary.to_csv(os.path.join(args.out_dir, f'summary_{prefix}.csv'), index=False)
        joblib.dump(model, os.path.join(args.out_dir, f'model_{prefix}.joblib'))

        n_clusters = args.k if args.algo == 'kmeans' else len(set(labels)) - (1 if -1 in labels else 0)
        save_metadata(os.path.join(args.out_dir, f'info_{prefix}.txt'), args.algo, tag, args, n_clusters=n_clusters)

        if args.plot:
            plot_path = os.path.join(args.out_dir, f'plot_{prefix}.png')
            plot_clusters(num_df, labels, f'{args.algo.upper()} – {tag}', output_path=plot_path)

if __name__ == '__main__':
    main()