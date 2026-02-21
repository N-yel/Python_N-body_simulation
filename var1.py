"""
This file is a dataset for our simulation.
Comments are in French throughout the rest of the document.
"""
import numpy as np

# dans ce code, on va mettre 3 planete fixe de même masse et même rayon, et un satellite, on va voir vers quelle planete il va
long = 1000
haut = 1000
centre = [long/2,haut/2]
couleur = np.array( [(255,0,0),(0,255,0),(0,0,255),(255,0,255),(255,255,0),(0,255,255),(42,42,42),(0,0,0)], dtype = int)

# constante gravitationelle (je pense qu'on s'en fiche de la valeur)
G = 1

t = 0
t_max = 5000
# petite variation de temps qui augmente la précision mais plus il est petit, plus l'algo demande du calcul
dt = 1
# res représente le coef de frottement: plus il est faible, plus il y aura conservation de l'énergie
res = 0
#taille_zone représente la taille du rayon a partir du quel on considère que les points sont dans la zone d'une planete

save = 3
# convention : si i_sat = -1, alors il n'y a pas de satellite dans le sens ou on regarde l'évolution globale du système sans s'arreter si il y a une collision du satellite
i_sat = 3

# m: liste des masses de chaque planete
m = np.array([100,100,100,10], dtype= float)
# r: liste des rayons de chaque planete
r = np.array([10,10,10,5], dtype= float)
#systeme : contient la liste des coordonées pour chaque planete
sys = np.array([[centre[0]-100,centre[1]],[centre[0]+100,centre[1]],[centre[0],centre[1]], [456,503]], dtype= float)
#systemev : contient le vecteur vitesse de chaque planete
sysv = np.array([[0,0],[0,0],[0,0],[0,0]], dtype= float)
sysa = np.array([[0,0],[0,0],[0,0],[0,0]], dtype= float)
#fixe: contient un booléen pour chaque planete: si ce dernier est a true, alors on n'actualise pas sa position
fixe = np.array([True,True,True,False])
#trace: contient un booléen pour chaque planete: si ce dernier est a true on affichera ses points précédents
trace = [False,False,False,False]