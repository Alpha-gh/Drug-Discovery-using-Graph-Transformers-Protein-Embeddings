# mol_features.py
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

ATOM_LIST = ["C","N","O","S","F","Cl","Br","I","P","H"]

def atom_features(atom):
    sym = atom.GetSymbol()
    one_hot = [1 if sym == a else 0 for a in ATOM_LIST]
    degree = atom.GetDegree()
    formal_charge = atom.GetFormalCharge()
    hybrid = int(atom.GetHybridization())
    aromatic = int(atom.GetIsAromatic())
    implicit_h = atom.GetNumImplicitHs()
    return np.array(one_hot + [degree, formal_charge, hybrid, aromatic, implicit_h], dtype=np.float32)

def smiles_to_graph(smiles, use_3d=True):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)

    if use_3d:
        try:
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            AllChem.UFFOptimizeMolecule(mol)
            conf = mol.GetConformer()
            coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=np.float32)
        except Exception:
            coords = np.zeros((mol.GetNumAtoms(), 3), dtype=np.float32)
    else:
        coords = np.zeros((mol.GetNumAtoms(), 3), dtype=np.float32)

    feats = np.stack([atom_features(a) for a in mol.GetAtoms()], axis=0)  # (N, F)
    return feats, coords

def precompute_molecules(csv_path, smiles_col, id_col, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    seen = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Molecules"):
        mid = row[id_col]
        smi = row[smiles_col]
        if mid in seen:
            continue
        try:
            atom_feats, coords = smiles_to_graph(smi)
            np.savez_compressed(os.path.join(out_dir, f"{mid}.npz"),
                                atom_feats=atom_feats,
                                coords=coords,
                                smiles=smi)
            seen[mid] = True
        except Exception as e:
            print("Failed:", mid, e)
    print("Done molecule features.")

if __name__ == "__main__":
    # example
    precompute_molecules("kiba_full.csv", smiles_col="smiles", id_col="ligand_id", out_dir="mol_feats_kiba")

