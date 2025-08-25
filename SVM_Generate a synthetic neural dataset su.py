# Generate a synthetic neural dataset suitable for an SVM population decoder
# in the style of Marques et al. (2018), and provide working code to train
# and evaluate the decoder (accuracy vs number of neurons; and optional
# time-resolved decoding). Files will be saved to /mnt/data for download.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

rng = np.random.default_rng(7)

# -----------------------------
# 1) Simulation parameters
# -----------------------------
n_neurons = 120                 # total neurons in the pool
coherences = np.array([16, 40, 100])
directions = np.array([-1, +1]) # -1 = nasal (left), +1 = temporal (right)
trials_per_cond = 45            # per coherence x direction
T = 30                          # frames per trial for "time-resolved" (1.5 s at 20 Hz)

# Bias toward temporal-preferring DS neurons (like the paper)
p_temporal = 0.65

# -----------------------------
# 2) Build neuron parameters
# -----------------------------
# Each neuron has: preferred direction p in {-1, +1}, coherence gain g(c),
# amplitude A, and noise level. We also add weak shared noise to induce correlations.
neuron_pref = rng.choice(directions, size=n_neurons, p=[1-p_temporal, p_temporal])

# Direction selectivity strength ~ Beta distribution (skewed to modest values)
ds_strength = rng.beta(2.2, 2.2, size=n_neurons)  # 0..1
base_amp = rng.uniform(0.4, 1.2, size=n_neurons)  # scaling of ΔF/F
A = base_amp * (0.5 + ds_strength)               # final amplitude scale
baseline = rng.uniform(0.0, 0.2, size=n_neurons)  # small offsets

# Per-neuron noise (trial-to-trial) and temporal noise
trial_noise_sd = 0.25 * A                         # heteroscedastic
temporal_noise_sd = 0.15 * A                      # frame-wise

# Shared state gain (to introduce correlations)
shared_gain_sd = 0.12

# Coherence gain function: monotonic saturating (similar shape to paper's response vs coherence)
def coh_gain(c):
    # c in {16,40,100} → roughly [0.35, 0.7, 1.0]
    return (c/100.0)**0.65

coh_to_gain = {int(c): coh_gain(c) for c in coherences}

# Timecourse kernel (fast rise, slow decay) to shape ΔF/F over frames
t = np.linspace(0, 1.5, T)
tau_r, tau_d = 0.15, 0.60
h = (1 - np.exp(-t/tau_r)) * np.exp(-t/tau_d)
h /= h.max()

# -----------------------------
# 3) Simulate trials
# -----------------------------
rows = []
# Also store a time-resolved cube: trials x neurons x frames
n_trials = len(coherences) * len(directions) * trials_per_cond
cube = np.zeros((n_trials, n_neurons, T), dtype=np.float32)

trial_idx = 0
for c in coherences:
    g = coh_to_gain[int(c)]
    for d in directions:  # label: -1 nasal (left), +1 temporal (right)
        for _ in range(trials_per_cond):
            # shared state (e.g., arousal) adds correlated noise
            shared = rng.normal(0, shared_gain_sd)

            # per-neuron mean response for this condition (before noise)
            # preferred direction gets positive modulation; null gets suppression.
            # We'll implement a slight null suppression so the task is nontrivial.
            sign = (neuron_pref == d).astype(float)*1.0 - (neuron_pref != d).astype(float)*0.6
            mean_amp = baseline + A * g * sign

            # time-resolved trace: mean_amp scaled by kernel h plus temporal noise
            noise_t = rng.normal(0, temporal_noise_sd[:, None], size=(n_neurons, T))
            trace = (mean_amp[:, None] * h[None, :]) + shared + noise_t

            # trial-level mean during stimulus (approx mean ΔF/F)
            mean_df = trace[:, int(0.2*T):].mean(axis=1)  # ignore first 20% frames

            # add trial noise (independent across neurons)
            mean_df += rng.normal(0, trial_noise_sd)

            # Save in table
            row = {
                "trial_id": trial_idx,
                "signed_coherence": int(d * c),          # negative for nasal
                "coherence": int(c),
                "direction_label": int(d),               # -1 nasal, +1 temporal
                "choice_right_label": int(d == +1)       # 0=Left/Nasal, 1=Right/Temporal
            }
            # attach neuron features
            for i in range(n_neurons):
                row[f"n{i+1:03d}"] = float(mean_df[i])
            rows.append(row)

            # Save cube
            cube[trial_idx, :, :] = trace.astype(np.float32)
            trial_idx += 1

df = pd.DataFrame(rows)

# Save datasets
base = Path("/mnt/data")
mean_csv = base / "svm_sample_neural_meanResponses.csv"
df.to_csv(mean_csv, index=False)

npz_path = base / "svm_sample_neural_timeSeries.npz"
np.savez_compressed(npz_path, traces=cube, coherences=df["coherence"].values,
                    signed_coh=df["signed_coherence"].values,
                    dir_label=df["direction_label"].values,
                    choice_right=df["choice_right_label"].values)

# -----------------------------
# 4) Decoder utilities (with and without scikit-learn)
# -----------------------------
USE_SK = False
try:
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import StandardScaler
    USE_SK = True
except Exception as e:
    USE_SK = False

def simple_linear_svm_hinge(X, y, lr=0.1, C=1.0, epochs=200, batch=128, seed=0):
    """
    Minimal SGD hinge-loss linear SVM (L2-regularised). Returns weight vector and bias.
    y should be in {-1, +1}
    """
    rng2 = np.random.default_rng(seed)
    N, D = X.shape
    w = np.zeros(D)
    b = 0.0
    lam = 1.0 / C  # regularization strength

    for ep in range(epochs):
        idx = rng2.permutation(N)
        for s in range(0, N, batch):
            j = idx[s:s+batch]
            Xb = X[j]; yb = y[j]
            margins = yb * (Xb @ w + b)  # shape (B,)
            # hinge gradient: if margin < 1, contribute; else 0
            mask = (margins < 1).astype(float)
            # gradient wrt w: lam*w - (1/B)*sum( y_i * x_i * I[margin<1] )
            grad_w = lam * w - (Xb.T @ (yb * mask)) / len(j)
            grad_b = - np.mean(yb * mask)
            w -= lr * grad_w
            b -= lr * grad_b
        # small learning rate decay
        lr *= 0.98
    return w, b

def evaluate_decoder(mean_table, neuron_indices, coherence_value, repeats=50, test_frac=0.2, C=1.0):
    """Train/test on one coherence at a time, random resplits; return mean accuracy and SEM."""
    accs = []
    m = mean_table[mean_table["coherence"] == coherence_value].copy()
    X = m[[f"n{i+1:03d}" for i in neuron_indices]].to_numpy()
    y = (m["choice_right_label"].to_numpy().astype(int) * 2) - 1  # {-1,+1}

    for r in range(repeats):
        idx = rng.permutation(len(y))
        n_test = int(test_frac * len(y))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        # Standardise features (as in many SVM pipelines)
        mu = Xtr.mean(axis=0, keepdims=True)
        sd = Xtr.std(axis=0, keepdims=True) + 1e-6
        Xtrn = (Xtr - mu) / sd
        Xten = (Xte - mu) / sd

        if USE_SK:
            clf = LinearSVC(C=C, loss="hinge", max_iter=5000)
            clf.fit(Xtrn, ytr)
            yhat = clf.predict(Xten)
        else:
            w, b = simple_linear_svm_hinge(Xtrn, ytr, lr=0.2, C=C, epochs=400, batch=64, seed=r)
            yhat = np.sign(Xten @ w + b)
        acc = np.mean(yhat == yte)
        accs.append(acc)
    return float(np.mean(accs)), float(np.std(accs)/np.sqrt(len(accs)))

# -----------------------------
# 5) Reproduce Fig 6A-style curves: accuracy vs #neurons, per coherence
# -----------------------------
Ns = [1, 2, 5, 10, 20, 50, 80, 100]
means = {c: [] for c in coherences}
sems  = {c: [] for c in coherences}

# Randomly sample neuron sets of size N (without replacement)
for N in Ns:
    # choose one set for plotting stability; inner function re-splits train/test repeatedly
    neuron_idx = rng.choice(np.arange(n_neurons), size=N, replace=False)
    for c in coherences:
        m, s = evaluate_decoder(df, neuron_idx, coherence_value=int(c), repeats=60, test_frac=0.25, C=1.0)
        means[c].append(m*100.0)  # percent
        sems[c].append(s*100.0)

# Plot
plt.figure(figsize=(6.5, 4.2))
for c in coherences:
    plt.errorbar(Ns, means[c], yerr=sems[c], label=f"{int(c)}%", marker='o')
plt.axhline(50, linestyle="--")  # chance
plt.xlabel("Number of neurons")
plt.ylabel("Decoder accuracy (%)")
plt.title("Linear SVM decoder — accuracy vs population size (simulated data)")
plt.legend()
out_curve = base / "svm_decoder_accuracy_vs_neurons.png"
plt.tight_layout()
plt.savefig(out_curve, dpi=150)
plt.close()

# -----------------------------
# 6) Time-resolved decoding (optional): 10 and 50 neurons
# -----------------------------
def time_resolved_accuracy(cube, labels_right, neuron_indices, coherence_value, repeats=10, test_frac=0.2, C=1.0):
    """Return mean accuracy over time (frames) for a given coherence and neuron set."""
    # subset trials by coherence
    mask = df["coherence"].values == coherence_value
    Xfull = cube[mask][:, neuron_indices, :]  # trials x neurons x T
    y = ((labels_right[mask].astype(int)) * 2) - 1  # {-1,+1}
    accs = []
    for r in range(repeats):
        idx = rng.permutation(len(y))
        n_test = int(test_frac * len(y))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        Xtr, Xte = Xfull[train_idx], Xfull[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        # standardise per-neuron using training mean/std computed across trials and time (like flattening time)
        mu = Xtr.reshape(Xtr.shape[0], -1).mean(axis=0)  # not used directly
        # For simplicity, compute mean/std per neuron across train trials and all frames
        mu_n = Xtr.mean(axis=(0,2), keepdims=True)
        sd_n = Xtr.std(axis=(0,2), keepdims=True) + 1e-6
        # time loop
        acc_t = []
        for tt in range(T):
            Xtr_t = (Xtr[:,:,tt] - mu_n[:,:,0]) / sd_n[:,:,0]
            Xte_t = (Xte[:,:,tt] - mu_n[:,:,0]) / sd_n[:,:,0]
            if USE_SK:
                clf = LinearSVC(C=C, loss="hinge", max_iter=3000)
                clf.fit(Xtr_t, ytr)
                yhat = clf.predict(Xte_t)
            else:
                w, b = simple_linear_svm_hinge(Xtr_t, ytr, lr=0.2, C=C, epochs=400, batch=64, seed=r)
                yhat = np.sign(Xte_t @ w + b)
            acc_t.append(np.mean(yhat == yte))
        accs.append(acc_t)
    return np.mean(accs, axis=0)

# Load time series & labels
traces = cube  # already in memory
labels_right = df["choice_right_label"].values

for N in [10, 50]:
    neuron_idx = rng.choice(np.arange(n_neurons), size=N, replace=False)
    acc_time_40 = time_resolved_accuracy(traces, labels_right, neuron_idx, coherence_value=40, repeats=20)
    acc_time_100 = time_resolved_accuracy(traces, labels_right, neuron_idx, coherence_value=100, repeats=20)

    plt.figure(figsize=(6.2, 3.8))
    plt.plot(t, acc_time_40*100, label="40%")
    plt.plot(t, acc_time_100*100, label="100%")
    plt.axhline(50, linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Decoder accuracy (%)")
    plt.title(f"Time-resolved SVM accuracy — {N} neurons (simulated)")
    plt.legend()
    out_time = base / f"svm_time_resolved_{N}neurons.png"
    plt.tight_layout()
    plt.savefig(out_time, dpi=150)
    plt.close()

mean_csv, npz_path, out_curve, base / "svm_time_resolved_10neurons.png", base / "svm_time_resolved_50neurons.png"
