import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


with open(Path("data_solar_system") /"center_earth_moon_10ans.txt", "r") as f:
    content = f.read()

# on extrait le bloc entre $$SOE et $$EOE
bloc = content.split("$$SOE")[1].split("$$EOE")[0]

# on extrait les lignes contenant X, Y, Z (ce qui sera capturé par le findall sera ce qui est entre parenthèses)
# les crochets définissent une liste de caractères acceptés. Dans [-\d.E+] il n'y aura pas nécéssairement un moins
pattern = r"X\s*=\s*([-\d.E+]+)\s+Y\s*=\s*([-\d.E+]+)\s+Z\s*=\s*([-\d.E+]+)"

resultats = re.findall(pattern, bloc)

# 3. Convertir en floats
donnees_terre = [(float(x), float(y), float(z)) for x, y, z in resultats]

# print(donnees[:3])  # Aperçu


donnees_planetes = np.array([donnees_terre])
#on doit avoir une shape (3,N,len(t)) pour afficher
donnees_planetes = np.transpose(donnees_planetes,(2,0,1))
#print(donnees_planetes.shape)

"""Tracé des données"""

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim(donnees_planetes[0,:].min()-1, donnees_planetes[0,:].max()+1)
ax.set_ylim(donnees_planetes[1,:].min()-1, donnees_planetes[1,:].max()+1)
ax.set_zlim(donnees_planetes[2,:].min()-1, donnees_planetes[2,:].max()+1)

ax.set_title("Simulation N-corps")
ax.set_aspect("equal")

points = []
traces = []
for i in range(donnees_planetes.shape[1]):
    p, = ax.plot([], [],[], 'o',
                markersize=10,
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
    frames= donnees_planetes.shape[2],
    interval = 25,
    )

plt.show()
