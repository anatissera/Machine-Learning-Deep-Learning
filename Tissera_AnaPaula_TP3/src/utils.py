
import torch
import torch.nn as nn
import torch.optim as optim
import cupy as cp

from .Neural_Network import NeuralNetwork
from .metrics import Metrics

SEED = 42

def greedy_search(architectures, hyperparams, X_train, Y_train_oh, X_val, Y_val_oh, y_train, y_val, param_order, epochs=200, seed=SEED):
    """Realiza búsqueda greedy de hiperparámetros optimizando la accuracy en validación."""
    
    best_config = {}
    best_score = -float('inf')

    def eval_config(config):
        arch = config['arch']
        opt = config['optimizer']
        lr = config['learning_rate']
        bs = config['batch_size']
        l2 = config['l2_lambda']
        dp = config['dropout_p']
        bn = config['use_batchnorm']
        es = config['early_stopping']
        pat = config['patience']
        sched = config['lr_schedule']
        decay = config.get('decay_rate', None)
        lr_min = config.get('lr_min', None)

        if opt in ['gd', 'adam'] and bs is not None:
            return -float('inf') 
        if opt == 'mb' and bs is None:
            bs = 1

        nn = NeuralNetwork(
            layer_sizes=arch,
            learning_rate=lr,
            seed=seed,
            optimizer=opt,
            batch_size=bs,
            l2_lambda=l2,
            dropout_p=dp,
            use_batchnorm=bn,
            early_stopping=es,
            patience=pat,
            lr_min=lr_min
        )

        lr_fn = None
        if sched == 'linear':
            lr_fn = nn.get_linear_schedule(final_lr=lr_min or 0.001, max_epochs=epochs)
        elif sched == 'exp' and decay is not None:
            lr_fn = nn.get_exponential_schedule(decay_rate=decay, final_lr=lr_min)

        nn.train_bp(
            X_train, Y_train_oh,
            X_val=X_val, Y_val=Y_val_oh,
            epochs=epochs,
            plot=False,
            lr_schedule=lr_fn,
        )

        # eval en validación
        Yhat_v = nn.forward(X_val, train=False)
        ypred_v = cp.argmax(Yhat_v, axis=0)
        yproba_v = Yhat_v.T
        m_v = Metrics(y_true=y_val, y_pred=ypred_v, y_proba=yproba_v)
        return m_v.accuracy()

    # Itera sobre los parámetros de forma greedy
    for param in param_order:
        best_val = None
        param_best_score = -float('inf')

        if param == 'arch':
            candidates = architectures
        else:
            candidates = hyperparams[param]

        for val in candidates:
            if param == 'patience' and not best_config.get('early_stopping', False):
                break

            test_config = best_config.copy()
            test_config[param] = val

            for p in ['arch', 'optimizer', 'learning_rate', 'batch_size',
                      'l2_lambda', 'dropout_p', 'use_batchnorm',
                      'early_stopping', 'patience', 'lr_schedule', 'decay_rate', 'lr_min']:
                if p not in test_config:
                    if p == 'arch':
                        test_config[p] = architectures[0]
                    else:
                        test_config[p] = hyperparams[p][0]

            score = eval_config(test_config)

            if score > param_best_score:
                param_best_score = score
                best_val = val

        best_config[param] = best_val
        print(f"Best {param}: {best_val} (val_acc={param_best_score:.4f})")

        if param == param_order[-1]:
            best_score = param_best_score

    return best_config, best_score


def get_dataloader(X, y, batch_size):
    """
    Si batch_size es None, usa todo el set como un único batch (full‐batch).
    """
    X_t = torch.tensor(cp.asnumpy(X), dtype=torch.float32)
    y_t = torch.tensor(cp.asnumpy(y), dtype=torch.long)
    ds  = torch.utils.data.TensorDataset(X_t, y_t)
    bs  = batch_size or len(ds) 
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True)

def build_scheduler(optimizer, conf, n_batches):
    """
    Crea un scheduler según conf['lr_schedule'], usando conf['decay_rate'] y conf['lr_min'].
    """
    kind = conf['lr_schedule']
    if kind == 'linear':
        return optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=conf.get('lr_min', 0.0) / conf['learning_rate'],
            total_iters=conf['patience'] * n_batches
        )
    if kind == 'exp':
        return optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=conf.get('decay_rate', 1.0)
        )
    return None

def train_model(ModelClass, conf, epochs, X_train, y_train, X_val=None, y_val=None):
    """Entrena un modelo PyTorch con scheduler y early stopping, devolviendo el mejor estado."""
    
    train_dl = get_dataloader(X_train, y_train, conf['batch_size'])
    val_dl   = get_dataloader(X_val,   y_val,   conf['batch_size'])

    model     = ModelClass(conf)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=conf['learning_rate'],
                           weight_decay=conf['l2_lambda'])
    scheduler = build_scheduler(optimizer, conf, len(train_dl))

    best_val_loss = float('inf')
    patience_ctr  = 0
    best_state    = None

    train_losses = []
    val_losses   = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_train = 0.0
        for Xb, yb in train_dl:
            optimizer.zero_grad()
            logits = model(Xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            running_train += loss.item() * Xb.size(0)
        train_losses.append(running_train / len(train_dl.dataset))

        # validación
        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for Xb, yb in val_dl:
                logits = model(Xb)
                l      = criterion(logits, yb)
                running_val += l.item() * Xb.size(0)
        val_epoch_loss = running_val / len(val_dl.dataset)
        val_losses.append(val_epoch_loss)

        # early stopping
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            patience_ctr  = 0
            best_state    = model.state_dict()
        else:
            patience_ctr += 1
            if conf['early_stopping'] and patience_ctr >= conf['patience']:
                break

    model.load_state_dict(best_state)
    return model, train_losses, val_losses
