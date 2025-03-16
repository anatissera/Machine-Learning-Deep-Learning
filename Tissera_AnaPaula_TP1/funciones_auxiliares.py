import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def euclidean_distances(X):
    XXT = X @ X.T
    norms = np.diag(XXT)
    distances = np.sqrt(norms[:, np.newaxis] + norms[np.newaxis, :] - 2 * XXT)
    return distances

def normalize_dataset(dataset):
    return (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))

def normalize_var(variable):
    return (variable - np.mean(variable)) / np.std(variable)

def compute_covariance_matrix(dataset):
    covariance_matrix = np.cov(dataset, rowvar=False)
    return covariance_matrix

def plot_covariance_matrix(cov_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cov_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Matriz de Covarianza')
    plt.show()
    
def pca_with_svd(X, d):
    A = normalize_dataset(X)
    
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    
    U_d = U[:, :d]
    S_d = np.diag(S[:d])
    VT_d = Vt[:d, :]
    V_d = V[:, :d]
    
    # Componentes principales
    Z = np.dot(U_d, S_d)
    
    return Z, U_d, S_d, VT_d

def similarity_matrix(X, deviation):
    matrix = normalize_dataset(X)  # Centrar la matriz
    sim_matrix = np.exp(-euclidean_distances(matrix) / (2 * deviation**2))
    return sim_matrix

def plot_similarity_matrix(matrix, deviation, dim):
    
    sim_matrix = similarity_matrix(matrix, deviation)
    plt.figure()
    # plt.imshow(sim_matrix, cmap='vidris', interpolation='nearest')
    plt.imshow(sim_matrix, cmap='viridis')
    plt.colorbar()
    plt.title(f"Matriz de Similaridad para $d =$ {dim} con desviación {deviation}")
    plt.show()
    