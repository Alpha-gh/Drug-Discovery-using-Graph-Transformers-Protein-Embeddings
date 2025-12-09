import numpy as np
import pandas as pd

# Load files
ligands = [line.strip() for line in open("ligands_clean.txt")]
proteins = [line.strip() for line in open("proteins_clean.txt")]

Y = np.loadtxt("drug-target_interaction_affinities_Kd__Davis_et_al.txt")

rows = []
for i, drug in enumerate(ligands):
    for j, protein in enumerate(proteins):
        affinity = Y[i][j]
        if affinity > 0:  # skip missing values
            rows.append([i, j, drug, protein, affinity])

df = pd.DataFrame(rows, columns=["ligand_id", "protein_id", "smiles", "sequence", "affinity"])
df.to_csv("davis_full.csv", index=False)

print("Saved davis_full.csv with rows:", len(df))
