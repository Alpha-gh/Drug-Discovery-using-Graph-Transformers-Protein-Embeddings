# getattention.py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# CONFIG – change these if you want KIBA later
# -------------------------------------------------------------------
MODEL_PATH = "best_cmgt_kiba.pt"
CSV_PATH   = "kiba_full.csv"
MOL_DIR    = "mol_feats_kiba"
PROT_DIR   = "prot_embs_kiba"
SAMPLE_IDX = 0          # which row from CSV to visualize

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
#  Dataset (same as train file; pKd conversion included)
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
        kd = float(row["affinity"])  # raw Kd in nM
        if kd <= 0:
            kd = 1.0
        y = -np.log10(kd * 1e-9)      # pKd
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
            "ligand_id": ligand_id,
            "protein_id": protein_id,
        }


# ============================================================
#  CMGT Model – EXACT COPY of your training code
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


# ============================================================
#  ATTENTION CAPTURE VIA HOOKS
# ============================================================

def main():
    print("Using device:", device)

    # ----- Load dataset & one sample -----
    dataset = DTIDataset(CSV_PATH, MOL_DIR, PROT_DIR, preload=True)
    sample = dataset[SAMPLE_IDX]
    atom_feats = torch.from_numpy(sample["atom_feats"]).unsqueeze(0)  # (1, Na, Fa)
    coords     = torch.from_numpy(sample["coords"]).unsqueeze(0)      # (1, Na, 3)
    prot_emb   = torch.from_numpy(sample["prot_emb"]).unsqueeze(0)    # (1, Np, Fp)

    Na = atom_feats.size(1)
    Np = prot_emb.size(1)
    a_mask = torch.ones(1, Na, dtype=torch.bool)
    p_mask = torch.ones(1, Np, dtype=torch.bool)

    # ----- Build model & load weights -----
    atom_feat_dim = atom_feats.size(2)
    prot_emb_dim  = prot_emb.size(2)
    model = CMGTModel(atom_feat_dim, prot_emb_dim).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # ----- Register hooks on cross-attention layers -----
    atom_to_prot_attns = []  # shape: (layers, 1, Na, Np)
    prot_to_atom_attns = []  # shape: (layers, 1, Np, Na)

    def hook_a2p(module, inp, out):
        # out is (attn_output, attn_weights)
        attn_w = out[1]           # (B, tgt_len, src_len) = (B, Na, Np)
        atom_to_prot_attns.append(attn_w.detach().cpu())

    def hook_p2a(module, inp, out):
        attn_w = out[1]           # (B, tgt_len, src_len) = (B, Np, Na)
        prot_to_atom_attns.append(attn_w.detach().cpu())

    for layer in model.cross:
        layer.a2p.register_forward_hook(hook_a2p)
        layer.p2a.register_forward_hook(hook_p2a)

    # ----- Forward pass -----
    with torch.no_grad():
        _ = model(
            atom_feats.to(device),
            coords.to(device),
            prot_emb.to(device),
            a_mask.to(device),
            p_mask.to(device),
        )

    # Take attention from last cross layer
    a2p_last = atom_to_prot_attns[-1].squeeze(0).numpy()  # (Na, Np)
    p2a_last = prot_to_atom_attns[-1].squeeze(0).numpy()  # (Np, Na)

    os.makedirs("attention_outputs", exist_ok=True)
    np.save("attention_outputs/a2p_last.npy", a2p_last)
    np.save("attention_outputs/p2a_last.npy", p2a_last)

    # ----- Plot heatmaps -----
    def plot_heatmap(mat, xlabel, ylabel, title, out_path):
        plt.figure(figsize=(8, 6))
        plt.imshow(mat, aspect="auto", cmap="viridis")
        plt.colorbar()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

    lig_id = sample["ligand_id"]
    prot_id = sample["protein_id"]

    plot_heatmap(
        a2p_last,
        xlabel="Protein residues (index)",
        ylabel="Ligand atoms (index)",
        title=f"Atom → Protein attention (ligand {lig_id}, protein {prot_id})",
        out_path="attention_outputs/atom_to_protein_heatmap.png",
    )

    plot_heatmap(
        p2a_last,
        xlabel="Ligand atoms (index)",
        ylabel="Protein residues (index)",
        title=f"Protein → Atom attention (ligand {lig_id}, protein {prot_id})",
        out_path="attention_outputs/protein_to_atom_heatmap.png",
    )

    print("\nSaved:")
    print("  attention_outputs/a2p_last.npy")
    print("  attention_outputs/p2a_last.npy")
    print("  attention_outputs/atom_to_protein_heatmap.png")
    print("  attention_outputs/protein_to_atom_heatmap.png")


if __name__ == "__main__":
    main()
