import numpy as np
import models

# dans ce code, on va mettre 3 planete fixe de même masse et même rayon, et un satellite, on va voir vers quelle planete il va
long = 1000
haut = 1000
centre = [long/2,haut/2]

# constante gravitationelle (je pense qu'on s'en fiche de la valeur)
G = 1

t = 0
t_max = 1000
# petite variation de temps qui augmente la précision mais plus il est petit, plus l'algo demande du calcul
dt = 1
# res représente le coef de frottement: plus il est faible, plus il y aura conservation de l'énergie
res = 0

# convention : si i_sat = -1, alors il n'y a pas de satellite dans le sens ou on regarde l'évolution globale du système sans s'arreter si il y a une collision du satellite
i_sat = 3

p1 = models.Planete(pos=[centre[0]-100,centre[1]],
                    vit=[0,0],
                    acc=[0,0],
                    m=100,
                    r=10,
                    fixe=True,
                    couleur=(255,0,0))

p2 = models.Planete(pos=[centre[0]+100,centre[1]],
                    vit=[0,0],
                    acc=[0,0],
                    m=100,
                    r=10,
                    fixe=True,
                    couleur=(0,255,0))

p3 = models.Planete(pos=[centre[0],centre[1]],
                    vit=[0,0],
                    acc=[0,0],
                    m=100,
                    r=10,
                    fixe=True,
                    couleur=(0,0,255))

p4 = models.Planete(pos=[59,592],
                    vit=[0,0],
                    acc=[0,0],
                    m=10,
                    r=10,
                    fixe=False,
                    couleur=(255,0,255))