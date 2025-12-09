import matplotlib.pyplot as plt
import numpy as np

# ================================
# BASELINE METRICS (from papers)
# ================================

# -----------------------
# KIBA
# -----------------------
kiba_models = ["KronRLS", "SimBoost", "DeepDTA", "CMGT (Ours)"]
kiba_rmse = [0.64, 0.47, 0.44, 0.77]
kiba_ci   = [0.782, 0.836, 0.863, 0.68]
kiba_pearson = [None, None, None, 0.40]  # baseline papers did not report Pearson

# -----------------------
# DAVIS
# -----------------------
davis_models = ["KronRLS", "SimBoost", "DeepDTA", "CMGT (Ours)"]
davis_rmse = [0.62, 0.53, 0.51, 0.73]
davis_ci   = [0.871, 0.872, 0.878, 0.783]
davis_pearson = [None, None, None, 0.584]


# ================================
# Helper function to plot bars
# ================================
def plot_bar(models, values, title, ylabel, filename):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, values, color=["gray", "gray", "gray", "green"])
    
    # highlight "ours"
    bars[-1].set_color("crimson")

    # value labels
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=10)

    plt.title(title, fontsize=16)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(rotation=20, fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# ================================
# Generate charts
# ================================

# --- KIBA ---
plot_bar(kiba_models, kiba_rmse, "KIBA Dataset - RMSE Comparison", "RMSE", "kiba_rmse.png")
plot_bar(kiba_models, kiba_ci, "KIBA Dataset - CI Comparison", "Concordance Index", "kiba_ci.png")

# For Pearson, only ours is available → plot single bar
plot_bar(["CMGT (Ours)"], [0.40], "KIBA Dataset - Pearson Correlation", "Pearson", "kiba_pearson.png")


# --- DAVIS ---
plot_bar(davis_models, davis_rmse, "DAVIS Dataset - RMSE Comparison", "RMSE", "davis_rmse.png")
plot_bar(davis_models, davis_ci, "DAVIS Dataset - CI Comparison", "Concordance Index", "davis_ci.png")
plot_bar(["CMGT (Ours)"], [0.584], "DAVIS Dataset - Pearson Correlation", "Pearson", "davis_pearson.png")

print("\n✔ All charts generated successfully!")
print("Files saved as:")
print("    kiba_rmse.png, kiba_ci.png, kiba_pearson.png")
print("    davis_rmse.png, davis_ci.png, davis_pearson.png")
