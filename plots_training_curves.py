import re
import matplotlib.pyplot as plt

LOG_FILE = "train_log_davis.txt"

epochs = []
losses = []
rmses  = []

with open(LOG_FILE, "r") as f:
    for line in f:
        m = re.search(r"Epoch\s+(\d+):\s+Loss=([\d\.eE+-]+)\s+RMSE=([\d\.eE+-]+)", line)
        if m:
            ep = int(m.group(1))
            lo = float(m.group(2))
            rm = float(m.group(3))
            epochs.append(ep)
            losses.append(lo)
            rmses.append(rm)

plt.figure(figsize=(7,5))
plt.plot(epochs, losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.title("Training Loss Curve (DAVIS)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("train_loss_davis.png", dpi=300)
plt.close()

plt.figure(figsize=(7,5))
plt.plot(epochs, rmses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Validation RMSE")
plt.title("Validation RMSE Curve (DAVIS)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("val_rmse_davis.png", dpi=300)
plt.close()

print("Saved: train_loss_davis.png, val_rmse_davis.png")
