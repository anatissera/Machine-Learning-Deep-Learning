import numpy as np
import matplotlib.pyplot as plt
from src.metrics import calculate_mse, calculate_rmse, calculate_mae, calculate_r2

class LinearRegression:
    def __init__(self, X_train, y_train, X_val, y_val, feature_names=[], to_scale=["area"], to_standardize=["age", "rooms"], 
                 lambda_l1=0, lambda_l2=0, already_scaled_data=False, train_stats=None):
        
        self.l1 = lambda_l1 
        self.l2 = lambda_l2
        self.recieved_scaled_data = already_scaled_data
        
        if already_scaled_data:
            self.train_stats = train_stats
            self.X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
            self.y_train = y_train
            self.X_val = np.hstack((np.ones((X_val.shape[0], 1)), X_val))
            self.y_val = y_val
        else:
            self.feature_names = feature_names
            self.f_to_scale = to_scale
            self.f_to_standardize = to_standardize
            
            self.stats = {}
            self._compute_statistics(X_train, y_train)
            
            self.X_train = self._transform(X_train)
            self.y_train = self._scale_target(y_train)
            self.X_val = self._transform(X_val)
            self.y_val = self._scale_target(y_val)
        
        self.coef = None
        self.historial_perdida_train = []
        self.historial_perdida_val = []
            
        
    def _compute_statistics(self, X, y):
        self.stats = {}
        for i, feature in enumerate(self.feature_names):
            if feature in self.f_to_standardize:
                self.stats[feature] = {"mean": X[:, i].mean(), "std": X[:, i].std()}
            elif feature in self.f_to_scale:
                self.stats[feature] = {"min": X[:, i].min(), "max": X[:, i].max()}
        self.stats["price"] = {"min": y.min(), "max": y.max()}
    
    def _transform(self, X):
        X_transformed = X.copy()
        for i, feature in enumerate(self.feature_names):
            if feature in self.f_to_standardize:
                X_transformed[:, i] = (X[:, i] - self.stats[feature]["mean"]) / self.stats[feature]["std"]
            elif feature in self.f_to_scale:
                X_transformed[:, i] = (X[:, i] - self.stats[feature]["min"]) / (self.stats[feature]["max"] - self.stats[feature]["min"])
        return np.hstack((np.ones((X.shape[0], 1)), X_transformed))
    
    def _scale_target(self, y):
        return (y - self.stats["price"]["min"]) / (self.stats["price"]["max"] - self.stats["price"]["min"])
    
    def _inverse_transform_target(self, y_scaled):
        return y_scaled * (self.stats["price"]["max"] - self.stats["price"]["min"]) + self.stats["price"]["min"]

    def entrenar_pseudoinversa(self):
        n = self.X_train.shape[1]
        lambda_identity = self.l2 * np.eye(n)
        lambda_identity[0, 0] = 0  # No regularizar el término de sesgo (bias)
       
        self.coef = np.linalg.inv(self.X_train.T @ self.X_train + lambda_identity) @ self.X_train.T @ self.y_train


    def entrenar_descenso_gradiente(self, lr=0.01, epochs=1000, to_print = True, early_stopping=True, tol=1e-5, paciencia=10):
        m, n = self.X_train.shape
        self.coef = np.zeros(n)
        self.historial_perdida_train = []
        self.historial_perdida_val = []
        mejor_perdida = float('inf')
        paciencia_contador = 0

        for epoch in range(epochs):
            predicciones_train = self.X_train @ self.coef
            error_train = predicciones_train - self.y_train
            gradiente = (1 / m) * (self.X_train.T @ error_train)

            if self.l2 > 0:
                gradiente[1:] += self.l2 * self.coef[1:]
                
            if self.l1 > 0:
                gradiente[1:] += self.l1 * np.sign(self.coef[1:]) / (1 + self.l2)

            self.coef -= lr * gradiente
            
            perdida_train = np.mean(error_train ** 2)
            perdida_val = np.mean((self.X_val @ self.coef - self.y_val) ** 2)
            
            self.historial_perdida_train.append(perdida_train)
            self.historial_perdida_val.append(perdida_val)
            
            if early_stopping:
                if perdida_val < mejor_perdida - tol:
                    mejor_perdida = perdida_val
                    paciencia_contador = 0
                else:
                    paciencia_contador += 1
                    if paciencia_contador >= paciencia:
                        if to_print:
                            print(f"Early stopping en epoch {epoch+1}")
                        break

            if np.isnan(self.coef).any():
                print("Error: Coeficientes nan detectados, reducir la tasa de aprendizaje.")
                return


    def predecir(self, X):
        if self.recieved_scaled_data:
            if X.shape[1] + 1 == self.X_train.shape[1]:
                X = np.hstack((np.ones((X.shape[0], 1)), X))
            predicciones = X @ self.coef
            
            if self.train_stats and 'price_min' in self.train_stats and 'price_max' in self.train_stats:
                predicciones = predicciones * (self.train_stats['price_max'] - self.train_stats['price_min']) + self.train_stats['price_min']
            
            return predicciones
        
        else:
            X_transformed = self._transform(X)
            predicciones_scaled = X_transformed @ self.coef
            return self._inverse_transform_target(predicciones_scaled)
    
    def evaluar(self, X_test, y_test, set='test', to_print = True, metric = "mse"):	
        y_pred = self.predecir(X_test)
        if metric == "mse":
            mse = calculate_mse(y_test, y_pred)
            if to_print:
                print(f"Error cuadrático medio (MSE) en {set}: {mse:.4f}")
            return mse
        elif metric == "rmse":
            rmse = calculate_rmse(y_test, y_pred)
            if to_print:
                print(f"Raíz del error cuadrático medio (RMSE) en {set}: {rmse:.4f}")
            return rmse
        elif metric == "mae":
            mae = calculate_mae(y_test, y_pred)
            if to_print:
                print(f"Error absoluto medio (MAE) en {set}: {mae:.4f}")
            return mae
        elif metric == 'R2':
            r2 = calculate_r2(y_test, y_pred)
            if to_print:
                print(f"Coeficiente de determinación R^2 en {set}: {r2:.4f}")
            return r2

            
    def graficar_regresion_pseudoinversa(self, X, y, nombres):
        if X.shape[1] == 1:
            plt.scatter(X, y, color='blue', label='Real data')
            X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            y_pred = self.predecir(X_line)
            plt.plot(X_line, y_pred, color='green', label='Pseudo-inverse')
            plt.xlabel(nombres[0])
            plt.ylabel('Target price')
            plt.legend()
            plt.title('Linear Regression - Pseudo-inverse')
            plt.show()
    
    def graficar_regresion_descenso_gradiente(self, X, y, nombres):
        if X.shape[1] == 1:
            plt.scatter(X, y, color='blue', label='Real data')
            X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            y_pred = self.predecir(X_line)
            plt.plot(X_line, y_pred, color='red', label='Gradient Descent')
            plt.xlabel(nombres[0])
            plt.ylabel('Target price')
            plt.legend()
            plt.title('Linear Regression - Gradient Descent')
            plt.show()

    def graficar_perdida(self):
        if self.historial_perdida_train and self.historial_perdida_val:
            plt.plot(self.historial_perdida_train, label='Train')
            plt.plot(self.historial_perdida_val, label='Validation')
            plt.xlabel('Epochs')
            plt.ylabel('Medium Squared Error (MSE)')
            plt.title('Loss during training')
            plt.legend()
            plt.show()
        else:
            print("No hay datos de entrenamiento con descenso por gradiente.")



# modelo para estimar rooms
from src.preprocessing import one_hot_encoding, softmax
from src.data_splitting import divide_train_test
from src.utils import complete_data

def multinomial_logistic(X, y, lr=0.1, epochs=30000, patience=500, val_size=0.1, standardize_cols=None, scale_cols=None):
    """Entrena un modelo de regresión logística multinomial con normalización/escalado según corresponda."""
    
    num_samples, num_features = X.shape
    num_classes = np.unique(y).size
    
    val_split = int(num_samples * (1 - val_size))
    X_train, X_val = X[:val_split], X[val_split:]
    y_train, y_val = y[:val_split], y[val_split:]

    mean_train = np.mean(X_train[:, standardize_cols], axis=0) if standardize_cols else None
    std_train = np.std(X_train[:, standardize_cols], axis=0) + 1e-8 if standardize_cols else None
    min_train = np.min(X_train[:, scale_cols], axis=0) if scale_cols else None
    max_train = np.max(X_train[:, scale_cols], axis=0) if scale_cols else None

    if standardize_cols:
        X_train[:, standardize_cols] = (X_train[:, standardize_cols] - mean_train) / std_train
        X_val[:, standardize_cols] = (X_val[:, standardize_cols] - mean_train) / std_train

    if scale_cols:
        X_train[:, scale_cols] = (X_train[:, scale_cols] - min_train) / (max_train - min_train + 1e-8)
        X_val[:, scale_cols] = (X_val[:, scale_cols] - min_train) / (max_train - min_train + 1e-8)

    y_train_one_hot = one_hot_encoding(y_train, num_classes)
    y_val_one_hot = one_hot_encoding(y_val, num_classes)

    W = np.random.randn(num_features, num_classes) * 0.01
    b = np.zeros((1, num_classes))

    best_loss = np.inf
    best_W, best_b = W.copy(), b.copy()
    wait = 0 
    
    for epoch in range(epochs):
        logits = np.dot(X_train, W) + b
        probs = softmax(logits)

        loss_train = -np.mean(y_train_one_hot * np.log(probs + 1e-8))

        logits_val = np.dot(X_val, W) + b
        probs_val = softmax(logits_val)
        loss_val = -np.mean(y_val_one_hot * np.log(probs_val + 1e-8))
        
        grad_W = np.dot(X_train.T, (probs - y_train_one_hot)) / X_train.shape[0]
        grad_b = np.mean(probs - y_train_one_hot, axis=0, keepdims=True)

        W -= lr * grad_W
        b -= lr * grad_b

        # Early stopping
        if loss_val < best_loss:
            best_loss = loss_val
            best_W, best_b = W.copy(), b.copy()
            wait = 0 
        else:
            wait += 1

        if wait >= patience:
            print(f"Early stopping en época {epoch}, mejor pérdida de validación: {best_loss:.4f}")
            break

        if epoch % 5000 == 0:
            print(f"Época {epoch}, Pérdida Train: {loss_train:.4f}, Pérdida Val: {loss_val:.4f}")

    return best_W, best_b, mean_train, std_train, min_train, max_train


def log_predict(X, W, b):
    logits = np.dot(X, W) + b
    probs = softmax(logits)
    return np.argmax(probs, axis=1)

def precision(y_real, y_pred):
    return np.mean(y_real == y_pred)

def complete_missing_rooms_values(df, W, b, mean_train, std_train, min_train, max_train):
   
    df_faltantes = df[df['rooms'].isna()].copy()
    X_faltantes = df_faltantes[['area']].values
    X_faltantes = (X_faltantes - min_train) / (max_train - min_train + 1e-8)  # Min-Max Scaling
    
    y_pred_faltantes = log_predict(X_faltantes, W, b)
    df.loc[df['rooms'].isna(), 'rooms'] = y_pred_faltantes
    print(f"{len(df_faltantes)} valores faltantes en 'rooms' completados.")
    
    return df


def predict_rooms_train_test(df):
    
    datos = complete_data(df, ['rooms'])
    X = datos[['area']].values
    y = datos['rooms'].values.astype(int)

    y = y - np.min(y)

    X_train, X_test, y_train, y_test = divide_train_test(X, y)

    W, b, mean_train, std_train, min_train, max_train = multinomial_logistic(
        X_train, y_train, lr=0.1, epochs=30000, standardize_cols=[], scale_cols=[0]  # Escalado solo para 'area'
    )
    
    X_test = (X_test - min_train) / (max_train - min_train + 1e-8)  # Aplicar Min-Max Scaling en test

    y_pred = log_predict(X_test, W, b)
    accuracy = precision(y_test, y_pred)
    print(f"Precisión en el conjunto de prueba de train_df: {accuracy:.4f}")
    
    return W, b, mean_train, std_train, min_train, max_train


def predict_rooms_no_split(df):
 
    datos = complete_data(df, ['rooms'])
    X = datos[['area']].values
    y = datos['rooms'].values.astype(int)

    y = y - np.min(y)

    W, b, mean_train, std_train, min_train, max_train = multinomial_logistic(
        X, y, lr=0.1, epochs=30000, standardize_cols=[], scale_cols=[0]
    )
    
    return W, b, mean_train, std_train, min_train, max_train


# modelo para estimar age
from src.data_splitting import train_val_test_split
from src.utils import generate_polynomial_features, add_bias
from src.metrics import calculate_rmse

def normalize_or_scale(X, mean_train, std_train, min_train, max_train, standardize_cols, scale_cols):

    X_transformed = X.copy()
    if standardize_cols:
        X_transformed[:, standardize_cols] = (X[:, standardize_cols] - mean_train) / std_train
    if scale_cols:
        X_transformed[:, scale_cols] = (X[:, scale_cols] - min_train) / (max_train - min_train + 1e-8)
    return X_transformed

def train_regression_for_age(df_train, features, standardize_cols, scale_cols, grado=1):

    X_train = generate_polynomial_features(df_train[features].values, grado)
    
    mean_train = np.mean(X_train[:, standardize_cols], axis=0)
    std_train = np.std(X_train[:, standardize_cols], axis=0) + 1e-8
    min_train = np.min(X_train[:, scale_cols], axis=0)
    max_train = np.max(X_train[:, scale_cols], axis=0)
    
    X_train = normalize_or_scale(X_train, mean_train, std_train, min_train, max_train, standardize_cols, scale_cols)
    X_train = add_bias(X_train)
    y_train = df_train['age'].values
    
    theta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train
    return theta, mean_train, std_train, min_train, max_train

def reg_predict_age(X, theta, mean_train, std_train, min_train, max_train, standardize_cols, scale_cols, grado=1):
  
    X_poly = generate_polynomial_features(X, grado)
    X_poly = normalize_or_scale(X_poly, mean_train, std_train, min_train, max_train, standardize_cols, scale_cols)
    X_poly = add_bias(X_poly)
    return X_poly @ theta

def evaluate_reg_model_age(df, theta, mean_train, std_train, min_train, max_train, features, standardize_cols, scale_cols, grado):
  
    X = generate_polynomial_features(df[features].values, grado)
    X = normalize_or_scale(X, mean_train, std_train, min_train, max_train, standardize_cols, scale_cols)
    X = add_bias(X)
    y_pred = X @ theta
    return calculate_rmse(df['age'].values, y_pred)

def complete_missing_age_values(df, theta, mean_train, std_train, min_train, max_train, features, standardize_cols, scale_cols, grado=1):
  
    mask_missing = df['age'].isna()
    if mask_missing.sum() == 0:
        print("No hay valores faltantes en 'age'.")
        return df
    
    X_faltantes = df.loc[mask_missing, features].values
    df.loc[mask_missing, 'age'] = np.round(reg_predict_age(X_faltantes, theta, mean_train, std_train, min_train, max_train, standardize_cols, scale_cols, grado)).astype(int)
    
    print(f"{mask_missing.sum()} valores faltantes en 'age' completados.")
    return df

def evaluate_and_impute(df, features, standardize_cols, scale_cols, grado=1):
   
    df_train, df_val, df_test = train_val_test_split(df)
    
    theta, mean_train, std_train, min_train, max_train = train_regression_for_age(df_train, features, standardize_cols, scale_cols, grado)
    
    print(f"Train RMSE: {evaluate_reg_model_age(df_train, theta, mean_train, std_train, min_train, max_train, features, standardize_cols, scale_cols, grado):.4f}")
    print(f"Validation RMSE: {evaluate_reg_model_age(df_val, theta, mean_train, std_train, min_train, max_train, features, standardize_cols, scale_cols, grado):.4f}")
    print(f"Test RMSE: {evaluate_reg_model_age(df_test, theta, mean_train, std_train, min_train, max_train, features, standardize_cols, scale_cols, grado):.4f}")
    
    theta_full, mean_full, std_full, min_full, max_full = train_regression_for_age(df.dropna(subset=['age']), features, standardize_cols, scale_cols, grado)
    df = complete_missing_age_values(df, theta_full, mean_full, std_full, min_full, max_full, features, standardize_cols, scale_cols, grado)
    
    return df

# cross validation

def cross_validate(X_train, y_train, X_val, y_val, X_test, y_test, feature_names, to_scale, to_standarize, k=5, lambdas=np.logspace(-2, 2, 20), metric='mse', L2=True, training_method='pinv'):
    if lambdas is None or np.size(lambdas) == 0:
        raise ValueError("La lista de lambdas está vacía. Debes proporcionar valores.")

    n = X_train.shape[0]

    if k > n:
        raise ValueError(f"El número de folds (k={k}) no puede ser mayor que el número de muestras ({n}).")

    fold_size = n // k
    best_lambda = None
    best_score = float('inf')
    global_scores = []
    
    for lambda_ in lambdas:
        scores = []
        
        for i in range(k):
            X_ho = X_train[i * fold_size: (i + 1) * fold_size]  # Holdout (validación interna)
            y_ho = y_train[i * fold_size: (i + 1) * fold_size]
            X_tr = np.vstack((X_train[:i * fold_size], X_train[(i + 1) * fold_size:]))  # Train interno
            y_tr = np.hstack((y_train[:i * fold_size], y_train[(i + 1) * fold_size:]))

            lambda_l1 = lambda_ if not L2 else 0
            lambda_l2 = lambda_ if L2 else 0
            
            model = LinearRegression(X_tr, y_tr, X_ho, y_ho, 
                                     feature_names, to_scale, to_standarize, 
                                     lambda_l1=lambda_l1, lambda_l2=lambda_l2)
            
            if training_method == "gd":
                model.entrenar_descenso_gradiente(lr=0.01, epochs=5000, to_print=False)
            elif training_method == "pinv":
                model.entrenar_pseudoinversa()
            else:
                raise ValueError("Método de entrenamiento no válido. Usa 'pinv' o 'gd'.")

            mse = model.evaluar(X_ho, y_ho, set="validation", to_print=False)
            scores.append(mse)

        mean_score = np.mean(scores)
        global_scores.append(mean_score)
        
        if mean_score < best_score:
            best_score = mean_score
            best_lambda = lambda_
    
    final_model = LinearRegression(X_train, y_train, X_val, y_val, 
                                   feature_names, to_scale, to_standarize, 
                                   lambda_l1=best_lambda if not L2 else 0, 
                                   lambda_l2=best_lambda if L2 else 0)
    
    if training_method == "gd":
        final_model.entrenar_descenso_gradiente(lr=0.001, epochs=5000, to_print=False)
    elif training_method == "pinv":
        final_model.entrenar_pseudoinversa()

    print(f"\nMejor lambda encontrado: {best_lambda:.4f}")
    final_model.evaluar(X_val, y_val, set="validación final")
    final_model.evaluar(X_test, y_test, set="test")
    
    return best_lambda, global_scores
