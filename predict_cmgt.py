# predict_cmgt.py
import torch
import numpy as np
from rdkit import Chem
from mol_features import smiles_to_graph
from model_cmgt import CMGTModel
from transformers import AutoTokenizer, AutoModel

@torch.no_grad()
def embed_protein_seq(seq, model_name="Rostlab/prot_bert", device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    seq_spaced = " ".join(list(seq))
    tokens = tokenizer(seq_spaced, return_tensors="pt")
    out = model(**{k: v.to(device) for k,v in tokens.items()})
    emb = out.last_hidden_state.squeeze(0)  # (L,d)
    return emb.cpu().numpy()

@torch.no_grad()
def predict_affinity(smiles, sequence, ckpt_path="best_cmgt_kiba.pt", d_model=256, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    atom_feats, coords = smiles_to_graph(smiles)
    prot_emb = embed_protein_seq(sequence, device=device)

    atom_feats = torch.from_numpy(atom_feats).unsqueeze(0).to(device)  # (1,Na,F)
    coords = torch.from_numpy(coords).unsqueeze(0).to(device)          # (1,Na,3)
    prot_emb = torch.from_numpy(prot_emb).unsqueeze(0).to(device)      # (1,Np,Fp)

    a_mask = torch.ones(1, atom_feats.size(1), dtype=torch.bool, device=device)
    p_mask = torch.ones(1, prot_emb.size(1), dtype=torch.bool, device=device)

    atom_feat_dim = atom_feats.size(-1)
    prot_emb_dim = prot_emb.size(-1)
    model = CMGTModel(atom_feat_dim, prot_emb_dim, d_model=d_model).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    pred = model(atom_feats, coords, prot_emb, a_mask, p_mask).item()
    return pred

if __name__ == "__main__":
    smiles = "CCOc1ccc2nc(S(N)(=O)=O)sc2c1"  # example
    sequence = "MGDKY..."                   # real protein seq here
    score = predict_affinity(smiles, sequence)
    print("Predicted affinity:", score)

