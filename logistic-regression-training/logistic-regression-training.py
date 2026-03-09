import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def loss_fn(ys, preds):
    ys = ys.flatten()
    preds = preds.flatten()
    n = len(ys)
    eps = 1e-15
    preds = np.clip(preds, eps, 1-eps)
    loss = -1/n * np.sum([y*np.log(p) + (1-y)*np.log(1-p) for y, p in zip(ys, preds)])
    # print(f"Loss value: {loss}")
    return loss

def compute_grads(X, y, preds):
    grad_w = (X.T @ (preds-y)) / len(y)
    grad_b = np.mean(preds - y).item()
    # print(f"Grad w: {grad_w}\nGrad b: {grad_b}")
    return grad_w, grad_b

def update_weights(w, b, grad_w, grad_b, lr):
    w-=(lr*grad_w)
    b-=(lr*grad_b)
    return w, b

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    ## Define tensors
    # print(f"==== Define tensors ====")
    X = np.array(X)
    y = np.array(y).reshape(-1,1)
    w = np.zeros((len(X[0]), 1))
    b = 0

    for step in range(steps):
        ## Compute Sigma function
        # print(f"---- Compute Sigma function ----")
        # print(f"X: {X.shape}, W: {w.shape}, b: {b.shape}, y: {y.shape}")
        sigmoid = _sigmoid(X@w + b)
        # print(f"Sigmoid: {sigmoid}")
        preds = np.where(sigmoid>=0.5, 1, 0)
        # print(f"Preds: {preds}")
        
        ## Compute loss function
        # print(f"---- Compute loss function ----")
        loss = loss_fn(y, sigmoid)

        ## Compute grads
        # print(f"---- Compute grads ----")
        grad_w, grad_b = compute_grads(X, y, sigmoid)

        ## Update weights
        # print(f"---- Update params ----")
        w, b = update_weights(w, b, grad_w, grad_b, lr)
        # print(f"Updated: new w = {w}, new b = {b}")
    return w.reshape(-1), b