import numpy as np
import matplotlib.pyplot as plt
import csv

rng = np.random.default_rng(0)

def relu(z):
    return np.maximum(0.0, z)

def relu_grad(z):
    return (z > 0).astype(z.dtype)

def generate_dataset(d=50, n=200, pi=0.0, rho_pos=0.5, seed=None):
    local = np.random.default_rng(seed) if seed is not None else rng
    eta = local.normal(0, 1, size=(d,))
    eta = eta / np.linalg.norm(eta)
    y_star = local.choice([-1.0, 1.0], size=(n,), p=[1-rho_pos, rho_pos])
    eps = local.normal(0, 1, size=(n, d))
    X = np.sqrt(d) * np.outer(y_star, eta) + eps
    flips = (y_star == 1.0) & (local.uniform(size=n) < pi)
    y_noisy = y_star.copy()
    y_noisy[flips] *= -1.0
    return X, y_noisy, y_star, eta

def init_params(d, H, seed=None):
    local = np.random.default_rng(seed) if seed is not None else rng
    W1 = local.normal(0, 1/np.sqrt(d), size=(d, H))
    b1 = np.zeros(H)
    a = local.normal(0, 1/np.sqrt(H), size=(H,))
    return W1, b1, a

def forward(X, W1, b1, a):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    out = A1 @ a
    return out, Z1, A1

def train_two_layer(X, y, H, lam=0.0, lr=2e-2, epochs=120, seed=None):
    n, d = X.shape
    W1, b1, a = init_params(d, H, seed=seed)
    for _ in range(epochs):
        f, Z1, A1 = forward(X, W1, b1, a)
        df = (f - y) / n
        grad_a = A1.T @ df + lam * a
        dA1 = np.outer(df, a)
        dZ1 = dA1 * relu_grad(Z1)
        grad_W1 = X.T @ dZ1 + lam * W1
        grad_b1 = dZ1.sum(axis=0) + lam * b1
        a  -= lr * grad_a
        W1 -= lr * grad_W1
        b1 -= lr * grad_b1
    return W1, b1, a

def predict(X, params):
    W1, b1, a = params
    f, _, _ = forward(X, W1, b1, a)
    return np.sign(f), f

def misclass_rate(y_true, y_pred):
    return np.mean(y_true != y_pred)

def run_sweep_noise(d=50, n=200, H_list=None, noises=(0.0, 0.1, 0.3), lam=0.0):
    if H_list is None:
        H_list = [1, 2, 3, 4, 6, 8, 12, 16, 24]
    results = {pi: [] for pi in noises}
    for pi in noises:
        for H in H_list:
            # one seed for speed
            Xtr, ytr, ytr_clean, _ = generate_dataset(d=d, n=n, pi=pi, seed=10)
            Xte, yte_noisy, yte_clean, _ = generate_dataset(d=d, n=3000, pi=0.0, seed=100)
            params = train_two_layer(Xtr, ytr, H, lam=lam, lr=2e-2, epochs=120, seed=1000+H)
            yhat, _ = predict(Xte, params)
            err = misclass_rate(yte_clean, yhat)
            results[pi].append(err)
    return H_list, results

def run_sweep_regularization(d=50, n=200, H_list=None, pi=0.3, lambdas=(0.0, 0.01)):
    if H_list is None:
        H_list = [1, 2, 3, 4, 6, 8, 12, 16, 24]
    results = {lam: [] for lam in lambdas}
    for lam in lambdas:
        for H in H_list:
            Xtr, ytr, ytr_clean, _ = generate_dataset(d=d, n=n, pi=pi, seed=20)
            Xte, yte_noisy, yte_clean, _ = generate_dataset(d=d, n=3000, pi=0.0, seed=200)
            params = train_two_layer(Xtr, ytr, H, lam=lam, lr=2e-2, epochs=120, seed=2000+H)
            yhat, _ = predict(Xte, params)
            err = misclass_rate(yte_clean, yhat)
            results[lam].append(err)
    return H_list, results

# Run
H_list, noise_results = run_sweep_noise()
H_list_reg, reg_results = run_sweep_regularization()

# Save CSVs
noise_csv_path = "/mnt/data/dd_noise_results.csv"
with open(noise_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    header = ["H"] + [f"pi={pi}" for pi in noise_results.keys()]
    writer.writerow(header)
    for i, H in enumerate(H_list):
        row = [H] + [noise_results[pi][i] for pi in noise_results.keys()]
        writer.writerow(row)

reg_csv_path = "/mnt/data/dd_reg_results.csv"
with open(reg_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    header = ["H"] + [f"lambda={lam}" for lam in reg_results.keys()]
    writer.writerow(header)
    for i, H in enumerate(H_list_reg):
        row = [H] + [reg_results[lam][i] for lam in reg_results.keys()]
        writer.writerow(row)

# Figure 1
plt.figure(figsize=(7,5))
for pi, errs in noise_results.items():
    plt.plot(H_list, errs, marker="o", label=f"pi={pi}")
plt.xlabel("Hidden units H (model size)")
plt.ylabel("Test error (misclassification rate)")
plt.title("Double Descent under Structured Label Noise")
plt.legend()
plt.grid(True, alpha=0.3)
fig1_path = "/mnt/data/figure1_double_descent_noise.png"
plt.savefig(fig1_path, dpi=200, bbox_inches="tight")
plt.close()

# Figure 2
plt.figure(figsize=(7,5))
for lam, errs in reg_results.items():
    plt.plot(H_list_reg, errs, marker="o", label=f"lambda={lam}")
plt.xlabel("Hidden units H (model size)")
plt.ylabel("Test error (misclassification rate)")
plt.title("Effect of L2 Regularization (pi=0.3)")
plt.legend()
plt.grid(True, alpha=0.3)
fig2_path = "/mnt/data/figure2_regularization_effect.png"
plt.savefig(fig2_path, dpi=200, bbox_inches="tight")
plt.close()

noise_csv_path, reg_csv_path, fig1_path, fig2_path
