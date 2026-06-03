import numpy as np
class Planete:
    def __init__(self, pos, vit, masse, rayon, couleur=(255,255,255), fixe=False, nom=""):
        self.pos  = np.array(pos,    dtype=float)
        self.vit  = np.array(vit,    dtype=float)
        self.acc  = np.zeros_like(self.pos)
        self.acc_old  = np.zeros_like(self.pos)
        self.masse  = masse
        self.rayon  = rayon
        self.couleur = np.array(couleur, dtype=int)
        self.fixe   = fixe
        self.nom    = nom
    
    # renvoie en texte la représentation d'une planète (utile pour le fichier csv)
    def __repr__(self):
        return f"pos={self.pos}, vit={self.vit}, m={self.m}, r={self.r}, dt={self.dt}, fixe={self.fixe}"

    def dist(self,p):
        return np.linalg.norm(self.pos - p.pos)
    
