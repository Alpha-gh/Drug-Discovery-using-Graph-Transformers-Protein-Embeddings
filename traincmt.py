import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# Use TF32 on CUDA (fast & stable, unlike FP16 on your card)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ============================================================
#  DTI Dataset  (optimized)
# ============================================================

class DTIDataset(Dataset):
    def __init__(self, csv_path, mol_dir, prot_dir, preload=True):
        self.df = pd.read_csv(csv_path)
        self.mol_dir = mol_dir
        self.prot_dir = prot_dir

        self.preload = preload
        self.lig_cache = {}
        self.prot_cache = {}

        if self.preload:
            print("Preloading ligand and protein features into RAM...")
            lig_ids = self.df["ligand_id"].unique()
            prot_ids = self.df["protein_id"].unique()

            for lid in lig_ids:
                path = os.path.join(self.mol_dir, f"{lid}.npz")
                mol = np.load(path, allow_pickle=True)
                self.lig_cache[lid] = {
                    "atom_feats": mol["atom_feats"].astype(np.float32),
                    "coords": mol["coords"].astype(np.float32),
                }

            for pid in prot_ids:
                path = os.path.join(self.prot_dir, f"{pid}.npz")
                prot = np.load(path, allow_pickle=True)
                self.prot_cache[pid] = prot["emb"].astype(np.float32)

            print(f"Preloaded {len(lig_ids)} ligands and {len(prot_ids)} proteins.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ligand_id = row["ligand_id"]
        protein_id = row["protein_id"]
        kd = float(row["affinity"])        # raw Kd in nM
        if kd <= 0:
            kd = 1.0                       # avoid log errors

        y = -np.log10(kd * 1e-9)           # convert nM → M → pKd
        y = np.float32(y)

        if self.preload:
            mol = self.lig_cache[ligand_id]
            prot_emb = self.prot_cache[protein_id]
            atom_feats = mol["atom_feats"]
            coords = mol["coords"]
        else:
            mol = np.load(os.path.join(self.mol_dir, f"{ligand_id}.npz"),
                          allow_pickle=True)
            prot = np.load(os.path.join(self.prot_dir, f"{protein_id}.npz"),
                           allow_pickle=True)
            atom_feats = mol["atom_feats"].astype(np.float32)
            coords = mol["coords"].astype(np.float32)
            prot_emb = prot["emb"].astype(np.float32)

        return {
            "atom_feats": atom_feats,
            "coords": coords,
            "prot_emb": prot_emb,
            "y": np.float32(y),
        }


def collate_fn(batch):
    B = len(batch)

    a_lens = [b["atom_feats"].shape[0] for b in batch]
    p_lens = [b["prot_emb"].shape[0] for b in batch]

    maxA = max(a_lens)
    maxP = max(p_lens)

    Fa = batch[0]["atom_feats"].shape[1]
    Fp = batch[0]["prot_emb"].shape[1]

    AF = torch.zeros(B, maxA, Fa, dtype=torch.float32)
    CO = torch.zeros(B, maxA, 3, dtype=torch.float32)
    PM = torch.zeros(B, maxP, Fp, dtype=torch.float32)
    a_mask = torch.zeros(B, maxA, dtype=torch.bool)
    p_mask = torch.zeros(B, maxP, dtype=torch.bool)
    ys = torch.zeros(B, dtype=torch.float32)

    for i, b in enumerate(batch):
        na = a_lens[i]
        np_ = p_lens[i]

        AF[i, :na, :] = torch.from_numpy(b["atom_feats"])
        CO[i, :na, :] = torch.from_numpy(b["coords"])
        PM[i, :np_, :] = torch.from_numpy(b["prot_emb"])

        a_mask[i, :na] = True
        p_mask[i, :np_] = True
        ys[i] = torch.tensor(b["y"], dtype=torch.float32)

    return {
        "atom_feats": AF,
        "coords": CO,
        "prot_emb": PM,
        "a_mask": a_mask,
        "p_mask": p_mask,
        "y": ys,
    }

# ============================================================
#  CMGT Model  (stabilized: fewer heads)
# ============================================================

class FourierDistanceEncoding(nn.Module):
    def __init__(self, d_model, num_bands=16, max_freq=10.0):
        super().__init__()
        self.num_bands = num_bands
        self.max_freq = max_freq
        self.proj = nn.Linear(2 * num_bands, d_model)

    def forward(self, coords, mask):
        B, N, _ = coords.shape
        c1 = coords.unsqueeze(2)
        c2 = coords.unsqueeze(1)
        d = torch.norm(c1 - c2, dim=-1)

        m = mask.unsqueeze(1) * mask.unsqueeze(2)
        d = d * m.float()

        denom = m.sum(-1).clamp(min=1.0)
        agg = d.sum(-1) / denom

        freqs = torch.linspace(
            1.0, self.max_freq, self.num_bands, device=coords.device
        ).view(1, 1, -1)
        v = agg.unsqueeze(-1) * freqs
        feats = torch.cat([torch.sin(v), torch.cos(v)], dim=-1)
        return self.proj(feats)


class SimpleTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead=4, dim_ff=512):   # nhead 8 -> 4
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, key_padding_mask=None):
        x_t = x.transpose(0, 1)
        att_out, _ = self.attn(x_t, x_t, x_t, key_padding_mask=key_padding_mask)
        x = x + att_out.transpose(0, 1)
        x = self.norm1(x)

        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class GraphTransformerEncoder(nn.Module):
    def __init__(self, in_dim, d_model=256, n_layers=3):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        self.layers = nn.ModuleList(
            [SimpleTransformerLayer(d_model) for _ in range(n_layers)]
        )
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, atom_feats, mask):
        x = self.in_proj(atom_feats)
        kpm = ~mask
        for layer in self.layers:
            x = layer(x, key_padding_mask=kpm)
        return self.out_proj(x)


class ResidueAtomCrossAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # heads 8 -> 4
        self.a2p = nn.MultiheadAttention(d_model, 4)
        self.p2a = nn.MultiheadAttention(d_model, 4)
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_p = nn.LayerNorm(d_model)

    def forward(self, atom_rep, prot_rep, a_mask, p_mask):
        atom_rep_t = atom_rep.transpose(0, 1)
        prot_rep_t = prot_rep.transpose(0, 1)

        a_kpm = ~a_mask
        p_kpm = ~p_mask

        att_a, _ = self.a2p(
            atom_rep_t, prot_rep_t, prot_rep_t, key_padding_mask=p_kpm
        )
        atom_rep = self.norm_a(atom_rep + att_a.transpose(0, 1))

        att_p, _ = self.p2a(
            prot_rep_t, atom_rep_t, atom_rep_t, key_padding_mask=a_kpm
        )
        prot_rep = self.norm_p(prot_rep + att_p.transpose(0, 1))

        return atom_rep, prot_rep


class CMGTModel(nn.Module):
    def __init__(self, atom_feat_dim, prot_emb_dim, d_model=256):
        super().__init__()
        self.graph_encoder = GraphTransformerEncoder(atom_feat_dim, d_model)
        self.prot_proj = nn.Linear(prot_emb_dim, d_model)

        self.geo = FourierDistanceEncoding(d_model)
        self.geo_merge = nn.Linear(d_model * 2, d_model)

        self.cross = nn.ModuleList(
            [ResidueAtomCrossAttention(d_model) for _ in range(2)]
        )

        self.pool_fc = nn.Linear(d_model * 2, d_model)
        self.head = nn.Linear(d_model, 1)

    def masked_mean(self, x, mask):
        m = mask.unsqueeze(-1).float()
        return (x * m).sum(1) / m.sum(1).clamp(min=1.0)

    def forward(self, atom_feats, coords, prot_emb, a_mask, p_mask):
        atom_rep = self.graph_encoder(atom_feats, a_mask)
        geo = self.geo(coords, a_mask)
        atom_rep = self.geo_merge(torch.cat([atom_rep, geo], dim=-1))

        prot_rep = self.prot_proj(prot_emb)

        for layer in self.cross:
            atom_rep, prot_rep = layer(atom_rep, prot_rep, a_mask, p_mask)

        atom_pool = self.masked_mean(atom_rep, a_mask)
        prot_pool = self.masked_mean(prot_rep, p_mask)

        joint = torch.cat([atom_pool, prot_pool], dim=-1)
        out = self.head(self.pool_fc(joint))
        return out.squeeze(-1)

# ============================================================
#  Training / Evaluation  (FP32 + grad clipping)
# ============================================================

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    criterion = nn.MSELoss()

    for batch in tqdm(loader, desc="Train"):
        AF = batch["atom_feats"].to(device, non_blocking=True)
        CO = batch["coords"].to(device, non_blocking=True)
        PM = batch["prot_emb"].to(device, non_blocking=True)
        a_mask = batch["a_mask"].to(device, non_blocking=True)
        p_mask = batch["p_mask"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        optimizer.zero_grad()
        preds = model(AF, CO, PM, a_mask, p_mask)
        loss = criterion(preds, y)
        loss.backward()

        # gradient clipping to avoid NaNs / explosions
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)

        optimizer.step()
        total_loss += loss.item() * AF.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    ys, yps = [], []
    for batch in tqdm(loader, desc="Val"):
        AF = batch["atom_feats"].to(device, non_blocking=True)
        CO = batch["coords"].to(device, non_blocking=True)
        PM = batch["prot_emb"].to(device, non_blocking=True)
        a_mask = batch["a_mask"].to(device, non_blocking=True)
        p_mask = batch["p_mask"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        preds = model(AF, CO, PM, a_mask, p_mask)
        ys.append(y.cpu().numpy())
        yps.append(preds.cpu().numpy())

    ys = np.concatenate(ys)
    yps = np.concatenate(yps)
    rmse = np.sqrt(((ys - yps) ** 2).mean())
    return rmse


def main():
    csv_path = "davis_full.csv"
    mol_dir = "mol_feats"
    prot_dir = "prot_embs"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = DTIDataset(csv_path, mol_dir, prot_dir, preload=True)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)

    train_ds, val_ds = random_split(dataset, [n_train, n_total - n_train])

    batch_size = 4      # safe for 8 GB GPU with this model

    pin_mem = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=pin_mem,
    )

    sample = dataset[0]
    atom_feat_dim = sample["atom_feats"].shape[1]
    prot_emb_dim = sample["prot_emb"].shape[1]

    model = CMGTModel(atom_feat_dim, prot_emb_dim).to(device)

    # LR 3e-4 -> 1e-4 for stability on KIBA
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    best_rmse = 1e9
    for epoch in range(1, 31):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_rmse = eval_epoch(model, val_loader, device)
        print(f"Epoch {epoch}: Loss={train_loss:.6f}  RMSE={val_rmse:.6f}")

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(model.state_dict(), "best_cmgt_davis.pt")
            print(f"Saved best model!  (RMSE={best_rmse:.6f})")


if __name__ == "__main__":
    main()
