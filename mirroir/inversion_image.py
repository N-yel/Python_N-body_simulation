from PIL import Image
from pathlib import Path
import numpy as np

# Charger l'image
img = Image.open(Path("Images") / Path("champ_gravitationnel_20260531_014359.png")).convert("RGB")
arr = np.array(img)

# Prendre les 500 lignes du haut
top_half = arr[:500, :, :]

# Créer le miroir vertical (flip vertical)
mirrored = top_half[::-1, :, :]

# Échanger le vert et le rouge dans la partie miroir
mirrored_swapped = mirrored.copy()

# Masque des pixels verts (R faible, G élevé, B faible)
green_mask = (mirrored[:, :, 0] < 100) & (mirrored[:, :, 1] > 150) & (mirrored[:, :, 2] < 100)

# Masque des pixels rouges (R élevé, G faible, B faible)
red_mask = (mirrored[:, :, 0] > 150) & (mirrored[:, :, 1] < 100) & (mirrored[:, :, 2] < 100)

# Remplacer le vert par du rouge
mirrored_swapped[green_mask] = [255, 0, 0]

# Remplacer le rouge par du vert
mirrored_swapped[red_mask] = [0, 255, 0]

# Construire l'image finale : 500 lignes du haut + 500 lignes miroir en bas
result = np.concatenate([top_half, mirrored_swapped], axis=0)

# Sauvegarder
result_img = Image.fromarray(result.astype(np.uint8))
result_img.save("output.png")
print("Image sauvegardée sous 'output.png'")