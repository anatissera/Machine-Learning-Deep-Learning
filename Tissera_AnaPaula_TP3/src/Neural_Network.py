import numpy as np
import matplotlib.pyplot as plt
EPS = 1e-15

class NeuralNetwork:
    def __init__(
        self,
        layer_sizes,
        learning_rate=0.01,
        seed=None,
        optimizer='gd',       # 'gd', 'sgd', 'mb', 'adam'
        batch_size=None,      # usado solo si optimizer='mb'
        l2_lambda=0.0,        # coeficiente L2
        dropout_p=0.0,        # probabilidad de dropout
        use_batchnorm=False,  # activar batch normalization
        early_stopping=False, # usar early stopping
        patience=5            # épocas de paciencia
    ):
        """
        Inicializa la red neuronal y sus opciones de entrenamiento.
        """
        if seed is not None:
            np.random.seed(seed)
        self.layer_sizes = layer_sizes
        self.initial_lr = learning_rate
        self.learning_rate = learning_rate
        self.L = len(layer_sizes) - 1  # número de capas con parámetros

        self.optimizer = optimizer
        self.batch_size = batch_size
        self.l2_lambda = l2_lambda
        self.dropout_p = dropout_p
        self.use_batchnorm = use_batchnorm
        self.early_stopping = early_stopping
        self.patience = patience

        # Para Adam
        self.adam_m = {}
        self.adam_v = {}
        self.adam_t = 0

        self._initialize_weights()

    def _initialize_weights(self):
        """Inicializa pesos y biases con He initialization."""
        self.weights = {}
        self.biases = {}
        self.gamma = {}
        self.beta = {}
        for l in range(1, self.L+1):
            in_size = self.layer_sizes[l-1]
            out_size = self.layer_sizes[l]
            self.weights[l] = np.random.randn(out_size, in_size) * np.sqrt(2. / in_size)
            self.biases[l] = np.zeros((out_size, 1))
            if self.use_batchnorm and l < self.L:
                self.gamma[l] = np.ones((out_size,1))
                self.beta[l] = np.zeros((out_size,1))
            if self.optimizer == 'adam':
                self.adam_m[l] = np.zeros_like(self.weights[l])
                self.adam_v[l] = np.zeros_like(self.weights[l])

    def _relu(self, Z):
        return np.maximum(0, Z)

    def _relu_derivative(self, Z):
        return (Z > 0).astype(float)

    def _softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return expZ / expZ.sum(axis=0, keepdims=True)

    def _batchnorm_forward(self, Z, l):
        """BatchNorm forward de la capa l."""
        mu = Z.mean(axis=1, keepdims=True)
        var = Z.var(axis=1, keepdims=True)
        Z_norm = (Z - mu) / np.sqrt(var + EPS)
        out = self.gamma[l] * Z_norm + self.beta[l]
        cache = (Z, Z_norm, mu, var, l)
        return out, cache

    def _batchnorm_backward(self, dOut, cache):
        """BatchNorm backward de la capa l."""
        Z, Z_norm, mu, var, l = cache
        m = Z.shape[1]
        dgamma = (dOut * Z_norm).sum(axis=1, keepdims=True)
        dbeta = dOut.sum(axis=1, keepdims=True)
        dZ_norm = dOut * self.gamma[l]
        dvar = (dZ_norm * (Z - mu) * -0.5 * (var + EPS)**(-1.5)).sum(axis=1, keepdims=True)
        dmu = (dZ_norm * -1/np.sqrt(var+EPS)).sum(axis=1, keepdims=True) + \
              dvar * ((-2 * (Z - mu)).mean(axis=1, keepdims=True))
        dZ = dZ_norm/np.sqrt(var+EPS) + dvar*2*(Z-mu)/m + dmu/m
        return dZ, dgamma, dbeta

    def forward(self, X, train=True):
        """Propagación hacia adelante."""
        A = X.T
        self.caches = {'A0': A}
        if self.use_batchnorm:
            self.bn_caches = {}
        for l in range(1, self.L+1):
            W, b = self.weights[l], self.biases[l]
            Z = W.dot(A) + b
            if self.use_batchnorm and l < self.L:
                Z, bn_cache = self._batchnorm_forward(Z, l)
                self.bn_caches[l] = bn_cache
            A = self._relu(Z) if l < self.L else self._softmax(Z)
            if train and self.dropout_p > 0 and l < self.L:
                mask = (np.random.rand(*A.shape) < (1 - self.dropout_p)) / (1 - self.dropout_p)
                A *= mask
                self.caches[f'D{l}'] = mask
            self.caches[f'Z{l}'], self.caches[f'A{l}'] = Z, A
        return A

    def compute_loss(self, Y_hat, Y):
        """Cross-entropy loss para clasificación multi-clase (+ L2 opcional)."""
        m = Y.shape[0]
        loss = -np.sum(np.log(Y_hat.T[Y == 1] + EPS)) / m
        if self.l2_lambda > 0:
            sum_w = sum((W**2).sum() for W in self.weights.values())
            loss += self.l2_lambda/(2*m) * sum_w
        return loss

    def _update_params(self, grads):
        """Actualiza parámetros con SGD o Adam."""
        m = self.batch_m
        if self.optimizer == 'adam':
            self.adam_t += 1
            for l in range(1, self.L+1):
                gW = grads[f'dW{l}'] + (self.l2_lambda/m)*self.weights[l]
                self.adam_m[l] = 0.9*self.adam_m[l] + 0.1*gW
                self.adam_v[l] = 0.999*self.adam_v[l] + 0.001*(gW**2)
                m_hat = self.adam_m[l] / (1 - 0.9**self.adam_t)
                v_hat = self.adam_v[l] / (1 - 0.999**self.adam_t)
                self.weights[l] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + EPS)
                self.biases[l]  -= self.learning_rate * grads[f'db{l}']
        else:
            for l in range(1, self.L+1):
                self.weights[l] -= self.learning_rate * grads[f'dW{l}']
                self.biases[l]  -= self.learning_rate * grads[f'db{l}']
        if self.use_batchnorm:
            for l in range(1, self.L):
                if f'dgamma{l}' in grads:
                    self.gamma[l] -= self.learning_rate * grads[f'dgamma{l}']
                    self.beta[l]  -= self.learning_rate * grads[f'dbeta{l}']

    def backward(self, Y):
        """Propagación hacia atrás para actualizar pesos y biases."""
        self.batch_m = Y.shape[0]
        grads = {}
        Y_hat = self.caches[f'A{self.L}']  # a⁽L⁾
        # δ⁽L⁾ = a⁽L⁾ – y
        dZ = Y_hat - Y.T
        for l in reversed(range(1, self.L+1)):
            A_prev = self.caches[f'A{l-1}']  # a⁽ℓ-1⁾
            if self.use_batchnorm and l < self.L:
                dZ, dgamma, dbeta = self._batchnorm_backward(dZ, self.bn_caches[l])
                grads[f'dgamma{l}'], grads[f'dbeta{l}'] = dgamma, dbeta
            # ∇W⁽ℓ⁾ = (1/m) ⋅ δ⁽ℓ⁾ ⋅ (a⁽ℓ-1⁾)ᵀ
            grads[f'dW{l}'] = (1/self.batch_m) * dZ.dot(A_prev.T)
            # ∇b⁽ℓ⁾ = (1/m) ⋅ sum_j δ⁽ℓ⁾_j
            grads[f'db{l}'] = (1/self.batch_m) * np.sum(dZ, axis=1, keepdims=True)
            if l > 1:
                Z_prev = self.caches[f'Z{l-1}']  # Z⁽ℓ-1⁾
                dA_prev = self.weights[l].T.dot(dZ)  # δ⁽ℓ-1⁾ = (W⁽ℓ⁾)ᵀ ⋅ δ⁽ℓ⁾
                if self.dropout_p > 0:
                    dA_prev *= self.caches[f'D{l-1}']  # máscara de dropout
                dZ = dA_prev * self._relu_derivative(Z_prev)  # δ⁽ℓ-1⁾ = dA ⊙ ReLU′(Z⁽ℓ-1⁾)
        self._update_params(grads)

    def get_linear_schedule(self, final_lr, max_epochs):
        """
        Crea función de tasa lineal con saturación:
        lr(t) = max(final_lr, initial_lr * (1 - t/max_epochs))
        """
        def lr_fn(t):
            return max(final_lr, self.initial_lr * (1 - t/max_epochs))
        return lr_fn

    def get_exponential_schedule(self, decay_rate):
        """
        Crea función de tasa exponencial:
        lr(t) = initial_lr * exp(-decay_rate * t)
        """
        def lr_fn(t):
            return self.initial_lr * np.exp(-decay_rate * t)
        return lr_fn

    def normal_bp(self, X_train, Y_train, X_val=None, Y_val=None,
                  epochs=100, plot=True, lr_schedule=None):
        """Entrena la red usando batch GD, SGD o mini-batch según optimizer y batch_size."""
        best_loss = np.inf
        wait = 0
        best_params = None
        train_losses, val_losses = [], []
        self.learning_rate = self.initial_lr
        for epoch in range(epochs):
            if lr_schedule:
                self.learning_rate = lr_schedule(epoch)
            # definir batches según optimizer
            m = X_train.shape[0]
            if self.optimizer in ['gd', 'adam']:
                batches = [np.arange(m)]
            elif self.optimizer == 'sgd':
                batches = [np.array([i]) for i in range(m)]
            else:  # 'mb'
                bs = self.batch_size or m
                perm = np.random.permutation(m)
                batches = [perm[i:i+bs] for i in range(0, m, bs)]
            # entrenar por lotes
            for batch in batches:
                Xb, Yb = X_train[batch], Y_train[batch]
                self.forward(Xb, train=True)
                self.backward(Yb)
            # pérdidas
            tr_loss = self.compute_loss(self.forward(X_train, train=False), Y_train)
            train_losses.append(tr_loss)
            if X_val is not None and Y_val is not None:
                vl_loss = self.compute_loss(self.forward(X_val, train=False), Y_val)
                val_losses.append(vl_loss)
                if self.early_stopping:
                    if vl_loss < best_loss:
                        best_loss, wait, best_params = vl_loss, 0, (self.weights.copy(), self.biases.copy())
                    else:
                        wait += 1
                        if wait >= self.patience:
                            self.weights, self.biases = best_params
                            # print(f"Early stopping en época {epoch+1}")
                            return train_losses, val_losses
        # plot de pérdidas
        if plot:
            from src.plot import plot_loss
            plot_loss(epochs, train_losses, val_losses)
        return train_losses, val_losses
