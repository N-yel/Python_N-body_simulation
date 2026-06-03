"""Ce fichier à pour but de comparer la simulation avec les données réelles du système solaire (sur un intervalle de 200 ans)"""

import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import copy
import var_poo3_leapfrog as var
import fun_iter_leapfrog as fun
import pandas as pd


def extraire(chemin):
    with open(chemin, "r") as f:
        content = f.read()

    # on extrait le bloc entre $$SOE et $$EOE
    bloc = content.split("$$SOE")[1].split("$$EOE")[0]

    # on extrait les lignes contenant X, Y, Z (ce qui sera capturé par le findall sera ce qui est entre parenthèses)
    # les crochets définissent une liste de caractères acceptés. Dans [-\d.E+] il n'y aura pas nécéssairement un moins
    pattern = r"X\s*=\s*([-\d.E+]+)\s+Y\s*=\s*([-\d.E+]+)\s+Z\s*=\s*([-\d.E+]+)"

    resultats = re.findall(pattern, bloc)

    # On Convertit en floats
    return [(float(x), float(y), float(z)) for x, y, z in resultats]


dossier = Path("data_solar_system") / Path("200_ans_solar_system_helio")
fichiers = sorted(dossier.iterdir())  # ordre alphabétique
donnees_planetes = [extraire(chemin) for chemin in fichiers if chemin.suffix == ".txt"]
donnees_planetes = np.array(donnees_planetes)

#print(donnees_planetes.shape)

#rayons des planetes du système solaire dans l'ordre alphabétique
vrais_rayons = np.array([71492,3396,2440,24766,60268,695700,6378,25559,6051])
log_rayons = np.log10(vrais_rayons)

copie_planete = copy.deepcopy(var.planetes)


def calcul_traj(dt):
    var.planetes = copy.deepcopy(copie_planete)
    continuer = True
    t=0
    var.dt = dt
    traj = []
    while(t < var.t_max):
        if not continuer:
            break
        t+=var.dt
        #on ajoute la valeur dans la liste que si t est entier
        if round(t,5) % 1 == 0:
            traj.append(np.array([planete.pos.copy() for planete in var.planetes]))
        #fun.actual met a jour les coordonées, vitesses, accélérations, de chacune des planètes dans le tab var.sys auquel l'on ajoute des dimensions
        _, continuer = fun.actual(var.planetes,var.t,var.res,var.i_sat,var.G)
    return np.array(traj)



"""Norme de la différence par rapport à la distance au soleil"""
# donnees_planetes = np.transpose(donnees_planetes,(1,0,2))
# #print(f"shape traj_np = {traj_np_01.shape} et shape donnees_planetes = {donnees_planetes.shape}")
# #les deux tableaux sont de shape (7305,9,3)

# difference = np.linalg.norm(calcul_traj(var.dt) - donnees_planetes, axis=-1)
# #différence est de shape (7305,9)

# dist = np.linalg.norm(donnees_planetes,axis=-1)
# dist[:,5]=1


# diff_par_dist = difference/dist

# moyenne =diff_par_dist.mean(axis=-1)

# # Création du DataFrame avec numérotation t
# df = pd.DataFrame({
#     't': range(0, 7305),
#     'moyenne': moyenne
# })

# # Export en CSV
# df.to_csv('Erreur_moyenne_par_dist_dt=1.csv', index=False)

"""Visualisation"""

#on doit avoir une shape (3,N,len(t)) pour afficher
donnees_planetes = np.transpose(donnees_planetes,(2,0,1))

#on doit avoir une shape (3,N,len(t)) pour afficher
traj_np = np.transpose(calcul_traj(var.dt),(2,1,0))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim(-1e9/2,1e9/2)
ax.set_ylim(-1e9/2,1e9/2)
ax.set_zlim(-1e9/2,1e9/2)

ax.set_xlabel("x (km)")
ax.set_ylabel("y (km)")
ax.set_zlabel("z (km)")

ax.set_title("Comparaison réel/théorique")
ax.set_aspect("equal")

points1 = []
traces1 = []
for i in range(donnees_planetes.shape[1]):
    p, = ax.plot([], [],[], 'o',
                markersize=log_rayons[i],
                color= 'red')
    points1.append(p)
    trace, = ax.plot([], [],[], '-',
                        linewidth=1,
                        alpha=0.5,         # transparence de la trace
                        color='red')
    traces1.append(trace)


points2 = []
traces2 = []
for i in range(traj_np.shape[1]):
    p, = ax.plot([], [],[], 'o',
                markersize=np.log10(var.planetes[i].r),
                color= 'blue')
    points2.append(p)
    trace, = ax.plot([], [],[], '-',
                        linewidth=1,
                        alpha=0.5,         # transparence de la trace
                        color='blue')
    traces2.append(trace)


def update_data(frame):
    #dans anim.set_data, on doit avoir un tab de shape (2,N,len(t))
    for i in range(len(points1)):
        points1[i].set_data_3d(
            [donnees_planetes[0,i, frame]],
            [donnees_planetes[1,i, frame]],
            [donnees_planetes[2,i, frame]]
        )
        traces1[i].set_data_3d(
            donnees_planetes[0, i, :frame],
            donnees_planetes[1, i, :frame],
            donnees_planetes[2, i, :frame]
        )
        points2[i].set_data_3d(
            [traj_np[0,i, frame]],
            [traj_np[1,i, frame]],
            [traj_np[2,i, frame]]
        )
        traces2[i].set_data_3d(
            traj_np[0, i, :frame],
            traj_np[1, i, :frame],
            traj_np[2, i, :frame]
        )
    return points1 + traces1 + points2 + traces2


animation = FuncAnimation(
    fig=fig,
    func= update_data,
    frames= range(1,donnees_planetes.shape[2]),
    interval = 25,
    )

plt.show()