import numpy as np
import models as models

long = 1000
haut = 1000
centre = [long/2, haut/2]

G = 6.674e-11 # en m^3 kg^-1 s^-2

#1px = 1000 km = 1e6 m
# 1 unité de temps = 1h = 3600 s
G = G*(3600**2)/(1e6**3)


t = 0
t_max = 10000
dt = 1
res = 0

# Pas de satellite
i_sat = 1

couleur = np.array([(255,0,0),(0,255,0),(0,0,255),(255,0,255),(255,255,0),(0,255,255),(42,42,42),(0,0,0)], dtype=int)

distance_tl = 384.4
terre = models.Planete(pos=np.array([centre[0], centre[1]], dtype=float),
                       vit=np.array([0, 0], dtype=float),
                       acc=np.array([0, 0], dtype=float),
                       m=5.972e24,
                       r=6,
                       dt=dt,
                       fixe=True)


lune = models.Planete(pos=np.array([centre[0] + distance_tl, centre[1]], dtype=float),
                      vit=np.array([0, np.sqrt(G*terre.m/384)], dtype=float),
                      acc=np.array([0, 0], dtype=float),
                      m=7.35e22,
                      r=1.7,
                      dt=dt,
                      fixe=False)

p2 = models.Planete(pos=np.array([180,386],dtype=float),
                    vit=np.array([-20,10],dtype=float),
                    acc=np.array([0,0],dtype=float),
                    m=50,
                    r=2,
                    dt=dt,
                    fixe=False)
planetes = np.array([terre, lune])


#on remarque que la période orbitale de la lune (temps qu'elle met à faire le tour de la terre ≈ 27 jour = 24*27 h) est correcte.


