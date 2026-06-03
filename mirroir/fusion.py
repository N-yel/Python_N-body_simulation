from PIL import Image
from pathlib import Path
import numpy as np

# Charger les deux images
img1 = Image.open(Path("Images") / Path("fusion2.png")).convert("RGB")
img2 = Image.open(Path("Images") / Path("champ_gravitationnel_20260531_175718.png")).convert("RGB")

arr1 = np.array(img1)
arr2 = np.array(img2)

# Là où img2 n'est pas blanc, on prend img2, sinon on prend img1
blanc_mask = (arr2[:, :, 0] == 255) & (arr2[:, :, 1] == 255) & (arr2[:, :, 2] == 255)

result = arr1.copy()
result[~blanc_mask] = arr2[~blanc_mask]

# Sauvegarder
Image.fromarray(result.astype(np.uint8)).save("fusion3.png")
print("Image sauvegardée : fusion3.png")