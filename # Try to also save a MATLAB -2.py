# Try to also save a MATLAB .mat version of the mean-responses table for convenience
from scipy.io import savemat
import numpy as np
import pandas as pd

df = pd.read_csv("/mnt/data/svm_sample_neural_meanResponses.csv")
data = {
    "trial_id": df["trial_id"].to_numpy(),
    "signed_coherence": df["signed_coherence"].to_numpy(),
    "coherence": df["coherence"].to_numpy(),
    "direction_label": df["direction_label"].to_numpy(),
    "choice_right_label": df["choice_right_label"].to_numpy(),
    "features": df[[c for c in df.columns if c.startswith("n")]].to_numpy()
}
savemat("/mnt/data/svm_sample_neural_meanResponses.mat", data)
print("Saved .mat file.")
