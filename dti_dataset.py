import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

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
        y = float(row["affinity"])

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