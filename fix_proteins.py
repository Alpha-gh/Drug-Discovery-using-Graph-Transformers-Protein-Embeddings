import json

# Input JSON-like protein file
input_file = "proteins kiba.txt"

# Output clean protein sequence file
output_file = "proteins_clean kiba.txt"

# Step 1 — Read entire content
with open(input_file, "r") as f:
    data = f.read().strip()

# Step 2 — Convert JSON-format into a Python dict
protein_dict = json.loads(data)

# Step 3 — Extract only sequences (ignore keys)
seq_list = list(protein_dict.values())

# Step 4 — Write sequences, one per line
with open(output_file, "w") as f:
    for seq in seq_list:
        f.write(seq + "\n")

print(f"Extracted {len(seq_list)} protein sequences.")
print("Clean output saved as proteins_clean.txt")
