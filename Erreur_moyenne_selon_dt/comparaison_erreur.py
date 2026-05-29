from pathlib import Path
import re
import matplotlib.pyplot as plt
dossier = Path("Erreur_moyenne_selon_dt")


fichiers = [
    "Erreur_moyenne_dt=0.01.csv",
    "Erreur_moyenne_dt=0.1.csv",
    "Erreur_moyenne_dt=0.2.csv",
    "Erreur_moyenne_dt=0.5.csv"
]

dossier = Path("Erreur_moyenne_selon_dt")
couleurs = ["blue", "red", "green", "orange","purple"]
dts = [0.01,0.1,0.2,0.5,1]


for i in range(len(fichiers)):
    t=[]
    error=[]
    with open(dossier /fichiers[i], "r", encoding="utf-8") as f:
        contenu = f.read()

    # Découpage par blocs séparés par des lignes vides
    blocs = contenu.strip().split("\n\n")

    for bloc in blocs:
        lignes = bloc.strip().split("\n")

    for ligne in lignes:
        parties = re.split(r',', ligne)
        if(parties[0] != 't'):
            t.append(int(parties[0]))
        if(parties[1] != 'moyenne'):
            error.append(float(parties[1]))
    plt.plot(t, error, color=couleurs[i], label=f"dt={dts[i]}")



# Mise en forme
plt.xlabel("t (10 jours)")
plt.ylabel("Erreur moyenne (km)")
plt.title("Erreur moyenne en fonction du temps")
plt.legend()
#plt.savefig("graphique_erreur_moyenne_par_dist_systeme_solaire.png")
plt.show()
