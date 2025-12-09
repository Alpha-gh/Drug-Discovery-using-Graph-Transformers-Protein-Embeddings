import json

# Input file (your downloaded JSON-like ligands_can.txt)
input_file = "ligands_can kiba.txt"

# Output file (clean SMILES-only list)
output_file = "ligands_clean kiba.txt"

# Read the entire line (the file has only ONE line)
with open(input_file, "r") as f:
    data = f.read().strip()

# Convert string → Python dict
ligand_dict = json.loads(data)

# Extract SMILES in order of appearance
smiles_list = list(ligand_dict.values())

# Write one SMILES per line
with open(output_file, "w") as f:
    for smi in smiles_list:
        f.write(smi + "\n")

print(f"Converted {len(smiles_list)} molecules into clean SMILES format.")
print("Output saved as ligands_clean.txt")
