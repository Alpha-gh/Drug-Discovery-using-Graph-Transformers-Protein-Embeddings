import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def plot_scatter(y_true, y_pred, title, save_path):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Compute Pearson
    r, _ = pearsonr(y_true, y_pred)

    plt.figure(figsize=(7, 7))

    # Scatter points
    plt.scatter(y_true, y_pred, alpha=0.4, s=10)

    # Best fit line
    p = np.polyfit(y_true, y_pred, 1)
    x_line = np.linspace(min(y_true), max(y_true), 100)
    y_line = np.polyval(p, x_line)
    plt.plot(x_line, y_line, color="red", linewidth=2)

    plt.title(f"{title}\nPearson = {r:.4f}", fontsize=14)
    plt.xlabel("Actual Affinity")
    plt.ylabel("Predicted Affinity")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Saved plot to {save_path}")


# ============================================================
# Example usage:
# Call plot_scatter(y_true, y_pred, ...)
# after evaluation in your existing code.
# ============================================================

if __name__ == "__main__":
    # Example dummy values — replace with your saved predictions!
    y_true = np.load("y_true1.npy")      # ← from evaluation script
    y_pred = np.load("y_pred1.npy")      # ← from evaluation script

    plot_scatter(y_true, y_pred,
                 title="CMGT - KIBA Dataset",
                 save_path="scatter_kiba.png")
