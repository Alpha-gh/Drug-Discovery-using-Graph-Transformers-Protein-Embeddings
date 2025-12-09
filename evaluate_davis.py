import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from traincmgt import DTIDataset, collate_fn, CMGTModel
from scipy.stats import pearsonr
from lifelines.utils import concordance_index

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def force_pkd(y):
    """If eval data contains raw Kd instead of pKd, convert automatically."""
    if np.max(y) > 50:     # raw Kd is in range 10–10,000
        print("⚠ Detected raw Kd — converting to pKd for evaluation.")
        return -np.log10(y * 1e-9)
    return y


@torch.no_grad()
def evaluate(model_path, csv_path, mol_dir, prot_dir, batch_size=8):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    dataset = DTIDataset(csv_path, mol_dir, prot_dir, preload=True)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    sample = dataset[0]
    atom_dim = sample["atom_feats"].shape[1]
    prot_dim = sample["prot_emb"].shape[1]

    model = CMGTModel(atom_dim, prot_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    y_true, y_pred = [], []

    for batch in tqdm(loader):
        AF = batch["atom_feats"].to(device)
        CO = batch["coords"].to(device)
        PM = batch["prot_emb"].to(device)
        a_mask = batch["a_mask"].to(device)
        p_mask = batch["p_mask"].to(device)
        y = batch["y"].cpu().numpy()

        preds = model(AF, CO, PM, a_mask, p_mask).cpu().numpy()

        y_true.append(y)
        y_pred.append(preds)

    # Merge all batches
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # 🔥 Fix RAW Kd → pKd automatically
    y_true = force_pkd(y_true)

    # Metrics
    rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
    pear, _ = pearsonr(y_true, y_pred)
    ci = concordance_index(y_true, y_pred)
    np.save("y_true.npy", y_true)
    np.save("y_pred.npy", y_pred)

    print("\n=== FINAL METRICS ===")
    print(f"RMSE     = {rmse:.4f}")
    print(f"Pearson  = {pear:.4f}")
    print(f"CI       = {ci:.4f}")


if __name__ == "__main__":
    evaluate(
        model_path="best_cmgt_davis.pt",
        csv_path="davis_full.csv",
        mol_dir="mol_feats",
        prot_dir="prot_embs",
        batch_size=8
    )
