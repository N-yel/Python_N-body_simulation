import numpy as np
import models

# dans ce code, on va mettre 3 planete fixe de même masse et même rayon, et un satellite, on va voir vers quelle planete il va
long = 1000
haut = 1000
centre = [long/2,haut/2]

# constante gravitationelle (je pense qu'on s'en fiche de la valeur)
G = 6.7e-11

t = 0
t_max = 5000
# petite variation de temps qui augmente la précision mais plus il est petit, plus l'algo demande du calcul
dt = 1
# res représente le coef de frottement: plus il est faible, plus il y aura conservation de l'énergie
res = 0

# convention : si i_sat = -1, alors il n'y a pas de satellite dans le sens ou on regarde l'évolution globale du système sans s'arreter si il y a une collision du satellite
i_sat = -1

couleur = np.array( [(255,0,0),(0,255,0),(0,0,255),(255,0,255),(255,255,0),(0,255,255),(42,42,42),(0,0,0)], dtype = int)

distance_tl = 234.3

terre = models.Planete(pos=np.array([centre[0],centre[1]],dtype=float),
                    vit=np.array([0,0],dtype=float),
                    acc=np.array([0,0],dtype=float),
                    m=5.972e24,
                    r=65/2,
                    dt=dt,
                    fixe=True)

lune = models.Planete(pos=np.array([centre[0]+distance_tl,centre[1]],dtype=float),
                    vit=np.array([np.sqrt(G*terre.m/(distance_tl)),0],dtype=float),
                    acc=np.array([0,0],dtype=float),
                    m=7.35e22,
                    r=10.18/2,
                    dt=dt,
                    fixe=False)

sat = models.Planete(pos=np.array([centre[0],centre[1]],dtype=float),
                    vit=np.array([0,0],dtype=float),
                    acc=np.array([0,0],dtype=float),
                    m=100,
                    r=10,
                    dt=dt,
                    fixe=True)


planetes = np.array([terre,lune])