import numpy as np
import matplotlib.pyplot as plt

def plot_from_npy(path, title, xlabel, ylabel, out_path):
    mat = np.load(path)
    plt.figure(figsize=(8,6))
    plt.imshow(mat, aspect="auto", cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

# Example:
# plot_from_npy("attention_outputs/a2p_last.npy", "Atom → Protein", "Protein residues", "Ligand atoms", "a2p_heatmap.png")
