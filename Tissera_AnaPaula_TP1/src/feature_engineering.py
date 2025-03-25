import numpy as np
import pandas as pd
from src.utils import pca_with_svd

def pca_latlon(df):
    df['lat_sin'] = np.sin(df['lat'])
    df['lat_cos'] = np.cos(df['lat'])
    df['lon_sin'] = np.sin(df['lon'])
    df['lon_cos'] = np.cos(df['lon'])
    
    X_pca = df[['lat_sin', 'lat_cos', 'lon_sin', 'lon_cos']].values
    Z, U_d, S_d, VT_d = pca_with_svd(X_pca, d=2)

    df['pca_latlon_1'] = Z[:, 0]
    df['pca_latlon_2'] = Z[:, 1]
    df = df.drop(columns=['lat_sin', 'lat_cos', 'lon_sin', 'lon_cos'])
    
    explained_variance = (S_d ** 2) / np.sum(S_d ** 2)
    print(f"Varianza explicada por cada componente: {explained_variance}")
    
    # la segunda componente no explica casi varianza, por lo que nos vamos a quedar con la primera
    df = df.drop(columns=['pca_latlon_2'])
    
    return df

# k-means
from src.utils import haversine

def assign_clusters(points, centroids):
    labels = np.zeros(len(points), dtype=int)
    for i, point in enumerate(points):
        distances = [haversine(*point, *centroid) for centroid in centroids]
        labels[i] = np.argmin(distances)
    return labels

def compute_centroids(points, labels, k):
    new_centroids = np.zeros((k, 2))
    for i in range(k):
        cluster_points = points[labels == i]
        if len(cluster_points) > 0:
            new_centroids[i] = np.mean(cluster_points, axis=0)
    return new_centroids

def kmeans(points, k=2, max_iters=100, tol=1e-4):
    centroids = points[np.random.choice(len(points), k, replace=False)]
    for _ in range(max_iters):
        labels = assign_clusters(points, centroids)
        new_centroids = compute_centroids(points, labels, k)
        if np.max(np.abs(new_centroids - centroids)) < tol:
            break
        centroids = new_centroids
    return labels, centroids

def assign_to_cluster(point, centroids):
    min_dist = float('inf')
    best_cluster = -1
    for cluster_id, centroid in centroids.items():
        dist = haversine(*point, *centroid)
        if dist < min_dist:
            min_dist = dist
            best_cluster = cluster_id
    return best_cluster

def distance_to_centroid(df, lat_lon, centroids_dict):
    return np.array([haversine(*lat_lon[i], *centroids_dict[df[i]]) for i in range(len(lat_lon))])

    
def generate_power_features(df, num_features=300, max_power=12, seed=42):
    np.random.seed(seed)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if 'price' in numeric_cols:
        numeric_cols.remove('price')
    
    new_features = {}
    
    for _ in range(num_features):
        col = np.random.choice(numeric_cols) 
        power = np.random.randint(2, max_power + 1)
        new_col_name = f"{col}_pow_{power}"
        new_features[new_col_name] = df[col] ** power
    
    return pd.DataFrame(new_features)
