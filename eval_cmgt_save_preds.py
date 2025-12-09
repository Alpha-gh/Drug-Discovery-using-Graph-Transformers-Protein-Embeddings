# eval_cmgt_save_preds.py

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== Dataset (copy of your training one, DAVIS pKd version) =====
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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ligand_id = row["ligand_id"]
        protein_id = row["protein_id"]
        kd = float(row["affinity"])
        if kd <= 0:
            kd = 1.0
        y = -np.log10(kd * 1e-9)  # pKd

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

        return atom_feats, coords, prot_emb, np.float32(y)


def collate_fn(batch):
    atom_feats, coords, prot_embs, ys = zip(*batch)
    B = len(batch)
    a_lens = [a.shape[0] for a in atom_feats]
    p_lens = [p.shape[0] for p in prot_embs]
    maxA, maxP = max(a_lens), max(p_lens)
    Fa, Fp = atom_feats[0].shape[1], prot_embs[0].shape[1]

    AF = torch.zeros(B, maxA, Fa, dtype=torch.float32)
    CO = torch.zeros(B, maxA, 3, dtype=torch.float32)
    PM = torch.zeros(B, maxP, Fp, dtype=torch.float32)
    a_mask = torch.zeros(B, maxA, dtype=torch.bool)
    p_mask = torch.zeros(B, maxP, dtype=torch.bool)
    Y = torch.tensor(ys, dtype=torch.float32)

    for i in range(B):
        na, np_ = a_lens[i], p_lens[i]
        AF[i, :na] = torch.from_numpy(atom_feats[i])
        CO[i, :na] = torch.from_numpy(coords[i])
        PM[i, :np_] = torch.from_numpy(prot_embs[i])
        a_mask[i, :na] = True
        p_mask[i, :np_] = True

    return AF, CO, PM, a_mask, p_mask, Y


# ===== CMGT Model (exact copy of your train code) =====
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
    def __init__(self, d_model, nhead=4, dim_ff=512):
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


@torch.no_grad()
def run_eval(model_path, csv_path, mol_dir, prot_dir, out_prefix, batch_size=16):
    ds = DTIDataset(csv_path, mol_dir, prot_dir, preload=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    sample = ds[0]
    atom_dim = sample[0].shape[1]
    prot_dim = sample[2].shape[1]

    model = CMGTModel(atom_dim, prot_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_true, all_pred = [], []

    for AF, CO, PM, a_mask, p_mask, Y in tqdm(dl, desc="Eval"):
        AF = AF.to(device)
        CO = CO.to(device)
        PM = PM.to(device)
        a_mask = a_mask.to(device)
        p_mask = p_mask.to(device)
        Y = Y.to(device)

        preds = model(AF, CO, PM, a_mask, p_mask)
        all_true.append(Y.cpu().numpy())
        all_pred.append(preds.cpu().numpy())

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)

    np.save(f"{out_prefix}_y_true.npy", all_true)
    np.save(f"{out_prefix}_y_pred.npy", all_pred)
    print(f"Saved: {out_prefix}_y_true.npy, {out_prefix}_y_pred.npy")


if __name__ == "__main__":
    # DAVIS
    '''run_eval(
        model_path="best_cmgt_davis.pt",
        csv_path="davis_full.csv",
        mol_dir="mol_feats",
        prot_dir="prot_embs",
        out_prefix="davis_cmgt",
        batch_size=14
    )'''

    #KIBA (only if you trained CMGT on KIBA with same architecture)
    run_eval(
        model_path="best_cmgt_kiba.pt",
        csv_path="kiba_full.csv",
        mol_dir="mol_feats_kiba",
        prot_dir="prot_embs_kiba",
        out_prefix="kiba_cmgt",
        batch_size=14
    )
