
import torch
import torch.nn as nn
import torch.optim as optim
import cupy as cp


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
