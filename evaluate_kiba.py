import os
import numpy as np
import torch
from scipy.stats import pearsonr
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from lifelines.utils import concordance_index

from traincmgt import (
    DTIDataset, collate_fn, CMGTModel
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# -----------------------------
# Concordance Index
# -----------------------------
'''def concordance_index(y_true, y_pred):
    n = 0
    h_sum = 0.0
    for i in range(len(y_true)):
        for j in range(i+1, len(y_true)):
            if y_true[i] != y_true[j]:
                n += 1
                if (y_pred[i] > y_pred[j] and y_true[i] > y_true[j]) or \
                   (y_pred[i] < y_pred[j] and y_true[i] < y_true[j]):
                    h_sum += 1
                elif y_pred[i] == y_pred[j]:
                    h_sum += 0.5
    return h_sum / n if n > 0 else 0.0'''


# -----------------------------
# Evaluate model
# -----------------------------
@torch.no_grad()
def evaluate(model_path, csv_path, mol_dir, prot_dir, batch_size=16):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    dataset = DTIDataset(csv_path, mol_dir, prot_dir, preload=True)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )

    sample = dataset[0]
    atom_dim = sample["atom_feats"].shape[1]
    prot_dim = sample["prot_emb"].shape[1]

    model = CMGTModel(atom_dim, prot_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    y_true, y_pred = [], []

    for batch in tqdm(loader, desc="Evaluating"):
        AF = batch["atom_feats"].to(device)
        CO = batch["coords"].to(device)
        PM = batch["prot_emb"].to(device)
        a_mask = batch["a_mask"].to(device)
        p_mask = batch["p_mask"].to(device)
        y = batch["y"].to(device)

        preds = model(AF, CO, PM, a_mask, p_mask)
        y_true.append(y.cpu().numpy())
        y_pred.append(preds.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
    print(rmse)
    pr, _ = pearsonr(y_true, y_pred)
    ci = concordance_index(y_true, y_pred)
    np.save("y_true1.npy", y_true)
    np.save("y_pred1.npy", y_pred)

    print("\n=== FINAL METRICS ===")
    print(f"RMSE     = {rmse:.4f}")
    print(f"Pearson  = {pr:.4f}")
    print(f"CI       = {ci:.4f}")

    return rmse, pr, ci


if __name__ == "__main__":
    evaluate(
        model_path="best_cmgt_kiba.pt",        # ← Davis model OR kiba
        csv_path="kiba_full.csv",
        mol_dir="mol_feats_kiba",
        prot_dir="prot_embs_kiba",
        batch_size=8
    )
