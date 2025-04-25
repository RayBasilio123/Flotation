"""
flot_clustering.py

Module for dimensionality reduction and visualization of flotation data embeddings,
colored by continuous or categorical target values.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 for 3D projection
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE, MDS, Isomap, LocallyLinearEmbedding
from sklearn.decomposition import PCA
import umap


def get_numeric_features(df: pd.DataFrame, exclude: list[str] = None) -> list[str]:
    """
    Return list of numeric column names, excluding any in `exclude`.
    """
    if exclude is None:
        exclude = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if col not in exclude]


def categorize_silica_quantiles(
    df: pd.DataFrame,
    source_col: str = 'conc_silica',
    bins: list[float] = None,
    labels: list[int] = None
) -> pd.Series:
    """
    Compute percentile rank of `source_col` and bin into categories.

    Default bins: (-inf,1%), (1%,5%), (5%,10%), (10%,inf). Labels: [0,1,2,3].
    """
    quant = df[source_col].rank(pct=True)
    if bins is None:
        bins = [-np.inf, 0.01, 0.05, 0.10, np.inf]
    if labels is None:
        labels = [0, 1, 2, 3]
    categories = pd.cut(quant, bins=bins, labels=labels).astype(int)
    return categories


def compute_tsne(
    X: np.ndarray,
    n_components: int = 2,
    random_state: int = None,
    **kwargs
) -> np.ndarray:
    """
    Compute t-SNE embedding of data matrix X.
    """
    model = TSNE(n_components=n_components, random_state=random_state, **kwargs)
    return model.fit_transform(X)


def compute_pca(
    X: np.ndarray,
    n_components: int = 2,
    random_state: int = None
) -> np.ndarray:
    """
    Compute PCA embedding of data matrix X.
    """
    model = PCA(n_components=n_components)
    return model.fit_transform(X)


def compute_umap(
    X: np.ndarray,
    n_components: int = 2,
    random_state: int = None,
    **kwargs
) -> np.ndarray:
    """
    Compute UMAP embedding of data matrix X.
    """
    model = umap.UMAP(n_components=n_components, random_state=random_state, **kwargs)
    return model.fit_transform(X)


def compute_mds(
    X: np.ndarray,
    n_components: int = 2,
    random_state: int = None,
    **kwargs
) -> np.ndarray:
    """
    Compute MDS embedding of data matrix X.
    """
    model = MDS(n_components=n_components, random_state=random_state, **kwargs)
    return model.fit_transform(X)


def compute_isomap(
    X: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 5,
    random_state: int = None,
    **kwargs
) -> np.ndarray:
    """
    Compute Isomap embedding of data matrix X.
    """
    model = Isomap(n_components=n_components, n_neighbors=n_neighbors, **kwargs)
    return model.fit_transform(X)


def compute_lle(
    X: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 5,
    random_state: int = None,
    **kwargs
) -> np.ndarray:
    """
    Compute Locally Linear Embedding of data matrix X.
    """
    model = LocallyLinearEmbedding(n_components=n_components, n_neighbors=n_neighbors, **kwargs)
    return model.fit_transform(X)


def plot_2d(
    embedding: np.ndarray,
    values,
    title: str,
    xlabel: str = 'Component 1',
    ylabel: str = 'Component 2',
    cmap: str = 'viridis',
    colorbar_label: str = None
):
    """
    2D scatter plot of embedding colored by `values`.
    """
    plt.figure()
    sc = plt.scatter(embedding[:, 0], embedding[:, 1], c=values, cmap=cmap, s=10)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if colorbar_label:
        plt.colorbar(sc, label=colorbar_label)
    else:
        plt.colorbar(sc)
    plt.tight_layout()
    plt.show()


def plot_3d(
    embedding: np.ndarray,
    values,
    title: str,
    xlabel: str = 'Component 1',
    ylabel: str = 'Component 2',
    zlabel: str = 'Component 3',
    cmap: str = 'viridis',
    colorbar_label: str = None
):
    """
    3D scatter plot of embedding colored by `values`.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(
        embedding[:, 0], embedding[:, 1], embedding[:, 2],
        c=values, cmap=cmap, s=10
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    if colorbar_label:
        fig.colorbar(sc, ax=ax, label=colorbar_label)
    else:
        fig.colorbar(sc, ax=ax)
    plt.tight_layout()
    plt.show()


def embedding_visualization(
    df: pd.DataFrame,
    source_col: str = 'conc_silica',
    exclude_cols: list[str] = None,
    methods: list[str] = None,
    random_state: int = None
):
    """
    Run specified dimensionality reduction methods and plot embeddings.

    Parameters
    ----------
    df : pd.DataFrame
    source_col : str
        Column to color points by (continuous values).
    exclude_cols : list[str]
        Columns to exclude from features.
    methods : list[str]
        List of methods: ['pca','umap','tsne','mds','isomap','lle'].
    random_state : int
    """
    if exclude_cols is None:
        exclude_cols = []
    if methods is None:
        methods = ['pca', 'umap', 'tsne', 'mds', 'isomap', 'lle']

    # prepare data
    features = get_numeric_features(df, exclude=exclude_cols + [source_col])
    X = df[features].values
    values = df[source_col].values

    # mapping method names to functions
    compute_funcs = {
        'pca': compute_pca,
        'umap': compute_umap,
        'tsne': compute_tsne,
        'mds': compute_mds,
        'isomap': compute_isomap,
        'lle': compute_lle
    }

    # run and plot each
    for m in methods:
        func = compute_funcs.get(m)
        if func is None:
            continue
        # 2D
        emb2 = func(X, n_components=2, random_state=random_state)
        plot_2d(
            emb2, values,
            title=f'{m.upper()} 2D embedding',
            colorbar_label=source_col
        )
        # 3D (if supported)
        try:
            emb3 = func(X, n_components=3, random_state=random_state)
            plot_3d(
                emb3, values,
                title=f'{m.upper()} 3D embedding',
                colorbar_label=source_col
            )
        except TypeError:
            # some methods may not accept random_state or 3 components
            pass


def characterize_embedding_clusters(
    df: pd.DataFrame,
    embedding: np.ndarray,
    source_col: str = 'conc_silica',
    method: str = 'kmeans',
    n_clusters: int = 4,
    eps: float = 1.0,
    min_samples: int = 5,
    features: list[str] = None,
    random_state: int = 42
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Cluster the embedding (e.g. UMAP or t-SNE) and summarize each cluster
    in terms of the original numeric variables.

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame (must align row-wise with embedding).
    embedding : np.ndarray
        Low-dimensional embedding of shape (n_samples, n_components).
    source_col : str
        Name of the target column (used only to exclude from features).
    method : {'kmeans','dbscan'}
        Clustering algorithm to use.
    n_clusters : int
        Number of clusters for k-means.
    eps : float
        Epsilon parameter for DBSCAN (if method='dbscan').
    min_samples : int
        min_samples parameter for DBSCAN (if method='dbscan').
    features : list of str, optional
        List of numeric columns to include. If None, all numeric columns
        except source_col are used.
    random_state : int
        Random state for reproducibility.

    Returns
    -------
    stats : pd.DataFrame
        Per-cluster summary (count, mean, std, median) of each feature.
    labels : np.ndarray
        Cluster labels for each sample.
    """
    from sklearn.cluster import DBSCAN

    # Choose clustering method
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels = clusterer.fit_predict(embedding)
    elif method == 'dbscan':
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clusterer.fit_predict(embedding)
    else:
        raise ValueError(f"Unknown method: {method!r}")

    # Prepare DataFrame with labels
    df2 = df.copy().reset_index(drop=True)
    df2['cluster'] = labels

    # Select features if not provided
    if features is None:
        numeric_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
        features = [c for c in numeric_cols if c not in [source_col, 'cluster']]

    # Compute summary statistics
    stats = df2.groupby('cluster')[features].agg(['count', 'mean', 'std', 'median'])
    return stats, labels

def cluster_feature_importance(
    df: pd.DataFrame,
    labels: np.ndarray,
    source_col: str = 'conc_silica',
    features: list[str] = None,
    random_state: int = 42
) -> pd.Series:
    """
    Train a RandomForestClassifier to distinguish clusters, returning feature importances.

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame (must align with labels).
    labels : array-like
        Cluster labels for each sample.
    source_col : str
        Name of the target column (to exclude).
    features : list of str, optional
        Feature columns to use. If None, all numeric except source_col and 'cluster'.
    random_state : int
        Random state for reproducibility.

    Returns
    -------
    importances : pd.Series
        Feature importances indexed by feature name, sorted descending.
    """
    df2 = df.copy().reset_index(drop=True)
    df2['cluster'] = labels

    # Exclude noise points (DBSCAN label = -1)
    mask = df2['cluster'] >= 0
    df2 = df2.loc[mask]

    # Select features
    if features is None:
        numeric_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
        features = [c for c in numeric_cols if c not in [source_col, 'cluster']]

    X = df2[features]
    y = df2['cluster']

    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X, y)

    importances = pd.Series(clf.feature_importances_, index=features)
    return importances.sort_values(ascending=False)

# Usage example:

# Assuming you already have an embedding (e.g., `emb2`) and your cleaned df `df_clean`:

# stats, labels = characterize_embedding_clusters(
#     df_clean, emb2, source_col='conc_silica',
#     method='kmeans', n_clusters=4
# )
# print(stats)

# importances = cluster_feature_importance(df_clean, labels, source_col='conc_silica')
# print(importances)

