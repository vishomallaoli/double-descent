#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv

# --------- Defaults ---------
DEFAULT_H = [1, 2, 3, 4, 6, 8, 12, 16, 24]
DEFAULT_NOISES = [0.0, 0.1, 0.3]
DEFAULT_LAMBDAS = [0.0, 0.01]

rng = np.random.default_rng(0)

def relu(z): return np.maximum(0.0, z)
def relu_grad(z): return (z > 0).astype(z.dtype)

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

def run_sweep_noise(d, n, H_list, noises, lam, lr, epochs):
    results = {pi: [] for pi in noises}
    for pi in noises:
        for H in H_list:
            Xtr, ytr, _, _ = generate_dataset(d=d, n=n, pi=pi, seed=10)
            Xte, _, yte_clean, _ = generate_dataset(d=d, n=3000, pi=0.0, seed=100)
            params = train_two_layer(Xtr, ytr, H, lam=lam, lr=lr, epochs=epochs, seed=1000+H)
            yhat, _ = predict(Xte, params)
            err = misclass_rate(yte_clean, yhat)
            results[pi].append(err)
    return results

def run_sweep_regularization(d, n, H_list, pi, lambdas, lr, epochs):
    results = {lam: [] for lam in lambdas}
    for lam in lambdas:
        for H in H_list:
            Xtr, ytr, _, _ = generate_dataset(d=d, n=n, pi=pi, seed=20)
            Xte, _, yte_clean, _ = generate_dataset(d=d, n=3000, pi=0.0, seed=200)
            params = train_two_layer(Xtr, ytr, H, lam=lam, lr=lr, epochs=epochs, seed=2000+H)
            yhat, _ = predict(Xte, params)
            err = misclass_rate(yte_clean, yhat)
            results[lam].append(err)
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=50)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=2e-2)
    ap.add_argument("--noise-levels", nargs="+", type=float, default=DEFAULT_NOISES)
    ap.add_argument("--lambdas", nargs="+", type=float, default=DEFAULT_LAMBDAS)
    ap.add_argument("--h-grid", nargs="+", type=int, default=DEFAULT_H)
    ap.add_argument("--reg-noise", type=float, default=0.3, help="noise level used for the regularization plot")
    ap.add_argument("--out-fig-dir", type=str, default="paper/figs")
    ap.add_argument("--out-csv-dir", type=str, default="results")
    args = ap.parse_args()

    fig_dir = Path(args.out_fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = Path(args.out_csv_dir); csv_dir.mkdir(parents=True, exist_ok=True)

    # Noise sweep
    noise_results = run_sweep_noise(
        d=args.d, n=args.n, H_list=args.h_grid, noises=args.noise_levels,
        lam=0.0, lr=args.lr, epochs=args.epochs
    )
    # Save CSV
    noise_csv = csv_dir / "dd_noise_results.csv"
    with noise_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["H"] + [f"pi={pi}" for pi in args.noise_levels])
        for i, H in enumerate(args.h_grid):
            row = [H] + [noise_results[pi][i] for pi in args.noise_levels]
            writer.writerow(row)
    # Plot
    plt.figure(figsize=(7,5))
    for pi in args.noise_levels:
        plt.plot(args.h_grid, noise_results[pi], marker="o", label=f"pi={pi}")
    plt.xlabel("Hidden units H (model size)")
    plt.ylabel("Test error (misclassification rate)")
    plt.title("Double Descent under Structured Label Noise")
    plt.legend(); plt.grid(True, alpha=0.3)
    fig1 = fig_dir / "figure1_double_descent_noise.png"
    plt.savefig(fig1, dpi=200, bbox_inches="tight"); plt.close()

    # Regularization sweep
    reg_results = run_sweep_regularization(
        d=args.d, n=args.n, H_list=args.h_grid, pi=args.reg_noise,
        lambdas=args.lambdas, lr=args.lr, epochs=args.epochs
    )
    # Save CSV
    reg_csv = csv_dir / "dd_reg_results.csv"
    with reg_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["H"] + [f"lambda={lam}" for lam in args.lambdas])
        for i, H in enumerate(args.h_grid):
            row = [H] + [reg_results[lam][i] for lam in args.lambdas]
            writer.writerow(row)
    # Plot
    plt.figure(figsize=(7,5))
    for lam in args.lambdas:
        plt.plot(args.h_grid, reg_results[lam], marker="o", label=f"lambda={lam}")
    plt.xlabel("Hidden units H (model size)")
    plt.ylabel("Test error (misclassification rate)")
    plt.title(f"Effect of L2 Regularization (pi={args.reg_noise})")
    plt.legend(); plt.grid(True, alpha=0.3)
    fig2 = fig_dir / "figure2_regularization_effect.png"
    plt.savefig(fig2, dpi=200, bbox_inches="tight"); plt.close()

    print(f"Saved:\n - {fig1}\n - {fig2}\n - {noise_csv}\n - {reg_csv}")

if __name__ == "__main__":
    main()
