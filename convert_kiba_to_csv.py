import numpy as np
import pandas as pd

ligands = [line.strip() for line in open("ligands_clean kiba.txt")]
proteins = [line.strip() for line in open("proteins_clean kiba.txt")]

Y = np.loadtxt("kiba_binding_affinity_v2.txt")

rows = []
for i, drug in enumerate(ligands):
    for j, protein in enumerate(proteins):
        affinity = Y[i][j]
        if affinity > 0:  # skip missing values
            rows.append([i, j, drug, protein, affinity])

df = pd.DataFrame(rows, columns=["ligand_id", "protein_id", "smiles", "sequence", "affinity"])
df.to_csv("kiba_full.csv", index=False)

print("Saved kiba_full.csv with rows:", len(df))
