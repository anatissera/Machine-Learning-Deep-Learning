# Neural_Network.py (convertido a CuPy)

import cupy as cp
import matplotlib.pyplot as plt
from tqdm import trange, tqdm   

EPS = 1e-15

class NeuralNetwork:
    def __init__(
        self,
        layer_sizes,
        learning_rate=0.01,
        seed=None,
        optimizer='gd',       # 'gd' o 'mb'
        batch_size=None,      # tamaño de mini-batch para 'mb'; si None usa 1 (SGD)
        l2_lambda=0.0,
        dropout_p=0.0,
        use_batchnorm=False,
        early_stopping=False,
        patience=5
    ):
        if seed is not None:
            cp.random.seed(seed)
        self.layer_sizes = layer_sizes
        self.initial_lr = learning_rate
        self.learning_rate = learning_rate
        self.L = len(layer_sizes) - 1

        self.optimizer = optimizer
        self.batch_size = batch_size
        self.l2_lambda = l2_lambda
        self.dropout_p = dropout_p
        self.use_batchnorm = use_batchnorm
        self.early_stopping = early_stopping
        self.patience = patience

        self.adam_m = {}
        self.adam_v = {}
        self.adam_t = 0

        self._initialize_weights()

    def _initialize_weights(self):
        self.weights = {}
        self.biases = {}
        self.gamma = {}
        self.beta = {}
        for l in range(1, self.L+1):
            in_size = self.layer_sizes[l-1]
            out_size = self.layer_sizes[l]
            self.weights[l] = cp.random.randn(out_size, in_size) * cp.sqrt(2. / in_size)
            self.biases[l] = cp.zeros((out_size, 1))
            if self.use_batchnorm and l < self.L:
                self.gamma[l] = cp.ones((out_size,1))
                self.beta[l]  = cp.zeros((out_size,1))
            if self.optimizer == 'adam':
                self.adam_m[l] = cp.zeros_like(self.weights[l])
                self.adam_v[l] = cp.zeros_like(self.weights[l])

    def _relu(self, Z):
        return cp.maximum(0, Z)

    def _relu_derivative(self, Z):
        return (Z > 0).astype(cp.float32)

    def _softmax(self, Z):
        expZ = cp.exp(Z - cp.max(Z, axis=0, keepdims=True))
        return expZ / expZ.sum(axis=0, keepdims=True)
    
    def _batchnorm_forward(self, Z, l):
        """BatchNorm forward de la capa l."""
        mu  = Z.mean(axis=1, keepdims=True)
        var = Z.var(axis=1, keepdims=True)
        Z_norm = (Z - mu) / cp.sqrt(var + EPS)
        out = self.gamma[l] * Z_norm + self.beta[l]
        cache = (Z, Z_norm, mu, var, l)
        return out, cache

    def _batchnorm_backward(self, dOut, cache):
        """BatchNorm backward de la capa l."""
        Z, Z_norm, mu, var, l = cache
        m = Z.shape[1]
        dgamma = (dOut * Z_norm).sum(axis=1, keepdims=True)
        dbeta  = dOut.sum(axis=1, keepdims=True)
        dZ_norm = dOut * self.gamma[l]
        dvar = (dZ_norm * (Z - mu) * -0.5 * (var + EPS)**(-1.5)).sum(axis=1, keepdims=True)
        dmu = (dZ_norm * -1/cp.sqrt(var+EPS)).sum(axis=1, keepdims=True) + \
              dvar * ((-2 * (Z - mu)).mean(axis=1, keepdims=True))
        dZ = dZ_norm/cp.sqrt(var+EPS) + dvar*2*(Z-mu)/m + dmu/m
        return dZ, dgamma, dbeta

    def forward(self, X, train=True):
        X_cp = cp.array(X)
        A = X_cp.T
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
                mask = (cp.random.rand(*A.shape) < (1 - self.dropout_p)) / (1 - self.dropout_p)
                A *= mask
                self.caches[f'D{l}'] = mask
            self.caches[f'Z{l}'], self.caches[f'A{l}'] = Z, A

        return A

    def compute_loss(self, Y_hat, Y):
        Y_cp = cp.array(Y)
        m = Y_cp.shape[0]
        probs = Y_hat.T[Y_cp == 1]
        loss = -cp.sum(cp.log(probs + EPS)) / m
        if self.l2_lambda > 0:
            sum_w = sum((W**2).sum() for W in self.weights.values())
            loss += self.l2_lambda/(2*m) * sum_w
        return float(loss.item())
    
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
                self.weights[l] -= self.learning_rate * m_hat / (cp.sqrt(v_hat) + EPS)
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
        Y_cp = cp.array(Y)
        self.batch_m = Y_cp.shape[0]
        grads = {}
        Y_hat = self.caches[f'A{self.L}']  # a⁽L⁾

        # δ⁽L⁾ = a⁽L⁾ – y
        dZ = Y_hat - Y_cp.T
        for l in reversed(range(1, self.L+1)):
            A_prev = self.caches[f'A{l-1}']
            if self.use_batchnorm and l < self.L:
                dZ, dgamma, dbeta = self._batchnorm_backward(dZ, self.bn_caches[l])
                grads[f'dgamma{l}'], grads[f'dbeta{l}'] = dgamma, dbeta

            grads[f'dW{l}'] = (1/self.batch_m) * dZ.dot(A_prev.T)
            grads[f'db{l}'] = (1/self.batch_m) * cp.sum(dZ, axis=1, keepdims=True)

            if l > 1:
                Z_prev = self.caches[f'Z{l-1}']
                W_orig = self.weights[l]
                dA_prev = W_orig.T.dot(dZ)
                if self.dropout_p > 0:
                    dA_prev *= self.caches[f'D{l-1}']
                dZ = dA_prev * self._relu_derivative(Z_prev)

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
            return self.initial_lr * cp.exp(-decay_rate * t)
        return lr_fn

    def train_bp(self, X_train, Y_train, X_val=None, Y_val=None,
                  epochs=3000, plot=True, lr_schedule=None):
        """Entrena la red usando Batch GD o Mini-batch SGD (incluye SGD si batch_size=1)."""
        best_loss = float('inf')
        wait = 0
        best_params = None
        train_losses, val_losses = [], []
        self.learning_rate = self.initial_lr

        for epoch in trange(epochs, desc="Epochs"):
            if lr_schedule:
                self.learning_rate = lr_schedule(epoch)

            m = X_train.shape[0]
            if self.optimizer in ['gd', 'adam']:
                batches = [cp.arange(m)]
            else:
                # Mini-batch SGD (batch_size=None -> SGD con bs=1)
                bs = self.batch_size or 1
                perm = cp.random.permutation(m)
                batches = [perm[i:i+bs] for i in range(0, m, bs)]

            for batch in tqdm(batches, desc=f" Epoch {epoch+1} batches", leave=False):
                Xb, Yb = X_train[batch], Y_train[batch]
                self.forward(Xb, train=True)
                self.backward(Yb)

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
                            return train_losses, val_losses

        if plot:
            from src.plot import plot_loss
            plot_loss(epochs, train_losses, val_losses)

        return train_losses, val_losses

