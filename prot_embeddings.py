# prot_embeddings.py
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel

def precompute_proteins(csv_path, seq_col, id_col, out_dir,
                        model_name="Rostlab/prot_bert", device="cpu"):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_safetensors=True,       
    ).to(device)
    model.eval()

    seen = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Proteins"):
        pid = row[id_col]
        seq = row[seq_col]
        if pid in seen:
            continue
        seen[pid] = True

        # ProtBERT expects spaces between amino acids
        seq_spaced = " ".join(list(seq))
        tokens = tokenizer(seq_spaced, return_tensors="pt")
        with torch.no_grad():
            out = model(**{k: v.to(device) for k, v in tokens.items()})
            emb = out.last_hidden_state.squeeze(0).cpu().numpy()    # (L, d)

        np.savez_compressed(os.path.join(out_dir, f"{pid}.npz"),
                            emb=emb,
                            sequence=seq)
    print("Done protein embeddings.")

if __name__ == "__main__":
    precompute_proteins("kiba_full.csv",
                        seq_col="sequence",
                        id_col="protein_id",
                        out_dir="prot_embs_kiba",
                        model_name="Rostlab/prot_bert",
                        device="cpu")
