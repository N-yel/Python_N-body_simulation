import numpy as np
from core import Planete, Scenario

G = 6.674e-11 # en m^3 kg^-1 s^-2

#1px = 1000 km = 1e6 m
# 1 unité de temps = 1h = 3600 s
G = G*(3600**2)/(1e6**3)

distance_tl = 384.4
terre = Planete(pos=np.array([0, 0], dtype=float),
                vit=np.array([0, 0], dtype=float),
                masse=5.972e24,
                rayon=6,
                couleur=(0,0,255),
                fixe=False,
                nom="terre")


lune = Planete(pos=np.array([distance_tl,0], dtype=float),
               vit=np.array([0, np.sqrt(G*terre.masse/384)], dtype=float),
               masse=7.35e22,
               couleur=(255,0,0),
               rayon=1.7,
               fixe=False,
               nom="lune")

planetes=[terre,lune]
scenar = Scenario(
    planetes=planetes,
    long=1000,
    haut=1000,
    G=G,
    t_max=2000,
    i_sat=-1,
    nom="terre_lune"
)
