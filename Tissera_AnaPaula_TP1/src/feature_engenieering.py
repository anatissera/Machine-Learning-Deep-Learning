from funciones_auxiliares import pca_with_svd
import numpy as np


def pca_latlon(df):
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
