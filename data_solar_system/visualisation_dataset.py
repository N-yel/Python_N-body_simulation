import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


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
#donnees_planetes = [extraire(dossier / "soleil.txt")]

donnees_planetes = np.array(donnees_planetes)
#on doit avoir une shape (3,N,len(t)) pour afficher
donnees_planetes = np.transpose(donnees_planetes,(2,0,1))
#print(donnees_planetes.shape)

#rayons des planetes du système solaire dans l'ordre alphabétique
vrais_rayons = np.array([71492,3396,2440,24766,60268,695700,6378,25559,6051])

rayon_min_affichage = 5   # taille minimale en points
rayon_max_affichage = 20  # taille maximale en points

log_rayons = np.log10(vrais_rayons)
"""Tracé des données"""

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim(donnees_planetes[0,:].min(), donnees_planetes[0,:].max())
ax.set_ylim(donnees_planetes[1,:].min(), donnees_planetes[1,:].max())
ax.set_zlim(donnees_planetes[2,:].min(), donnees_planetes[2,:].max())

ax.set_title("Simulation N-corps")
ax.set_aspect("equal")

points = []
traces = []
for i in range(donnees_planetes.shape[1]):
    p, = ax.plot([], [],[], 'o',
                markersize=log_rayons[i],
                color= 'red')
    points.append(p)
    trace, = ax.plot([], [],[], '-',
                        linewidth=1,
                        alpha=0.5,         # transparence de la trace
                        color='red')
    traces.append(trace)





def update_data(frame):
    #dans anim.set_data, on doit avoir un tab de shape (2,N,len(t))
    for i in range(len(points)):
        points[i].set_data_3d(
            [donnees_planetes[0,i, frame]],
            [donnees_planetes[1,i, frame]],
            [donnees_planetes[2,i, frame]]
        )
        traces[i].set_data_3d(
            donnees_planetes[0, i, :frame],
            donnees_planetes[1, i, :frame],
            donnees_planetes[2, i, :frame]
        )
    return points + traces


animation = FuncAnimation(
    fig=fig,
    func= update_data,
    frames= range(1,donnees_planetes.shape[2]),
    interval = 25,
    )

plt.show()
