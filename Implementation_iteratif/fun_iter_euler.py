import numpy as np
import csv
import copy
from time import perf_counter
from pathlib import Path
from PIL import Image



update_list = np.vectorize (lambda planete : planete.update())

def actual(planetes,t,res,i_sat,G):
    # sys.shape = (len(sys),2) : on stocke un tableau couples de deux coordonées, on pose ci-dessous n = nombre de planètes
    n = planetes.shape[0]
    save = -1
    continuer = True
    
    # pour toutes les planètes du système, si elles ne sont pas fixes, on somme l'accélération liée aux autres planètes
    for k in range(n):
        sumk = np.array([0,0],dtype=float)
        if(not planetes[k].fixe):
            for i in range(n):
                if k!=i:
                    # d: distance entre la planète i et k
                    d = planetes[i].dist(planetes[k])

                    #cas particulier, la planète que l'on observe est le satelite, alors on sauvegarde la planète dans laquelle elle se crashe avec la variable save.
                    #  On ne traite pas le cas ou il y a plusieurs collisions en même temps sachant qu'il est plutot improbable en vue de ce que l'on veut simuler
                    if (k == i_sat):
                        if d<= planetes[i].r + planetes[i_sat].r:
                            #si la position initiale a laquelle on lache le satelite est dans une des planètes, alors la couleur renvoyée est noire
                            if (t==0) and d <= planetes[i].r:
                                save = -1
                                continuer = False
                            #si le satellite touche une planete, on renvoie la couleur associée à cette planète
                            else:
                                save = i
                                continuer = False

                    #si deux planètes autre que le satellite rentrent en collision, on n'arrète pas la simulation mais on n'augmentera pas l'acceleration pour éviter les départs vers l'infini (d -> 0 implique 1/d**3 -> \infty)
                    if (d > planetes[k].r + planetes[i].r):
                        #on somme la composante de l'accélération selon toute les autre planètes et on soustrait un frottement fluide pour faire baisser artificiellement l'energie (a voir)
                        sumk += planetes[i].calcul_force(planetes[k],res,G)
            planetes[k].acc = sumk

    update_list(planetes)
    return save, continuer



#renvoie la planète si il y en a une vers laquelle le satellite a aterri, sinon (42,42,42)
def crash(planetes,res,i_sat,G,t_max):
    t = 0
    continuer = True
    save = -2
    while t < t_max and continuer:
        save, continuer = actual(planetes,t,res,i_sat,G)
        t+=planetes[0].dt
    # print(f"la couleur censé être renvoyé sera {var.couleur[save]}")
    return save



#va servir à stocker le temps mis par chaque itération
def get_prochain_fichier():
    dossier = Path("suivi_temps")
    dossier.mkdir(exist_ok=True)  # Crée le dossier s'il n'existe pas
    i = 0
    while (dossier / f"temps_iterations{i}.csv").exists():
        i += 1
    
    return dossier / f"temps_iterations{i}.csv",i


def get_fichier_image(i):
    return Path(f"img_{i}.png")





#va renvoyer un tableau de dimension (long,haut) avec la couleur de la planète vers laquelle le satellite d'indice i_sat va atterir en fonction de sa position
def calculer(long,haut,planetes,res,i_sat,G,t_max,couleur):
    start0 = perf_counter()
    #img: va stocker en la couleur en fct des conditionss initiales
    save = np.full((long,haut),-2,dtype = int)
    
    fichier, idx = get_prochain_fichier()
    fichier_img = get_fichier_image(idx)

    fichier_vide = not fichier.exists() or fichier.stat().st_size == 0

    with open(fichier, "a", newline="") as f:
        f.write(f"# long={long}, haut={haut}, res={res}, i_sat={i_sat}\n")
        for k in range(len(planetes)):
            f.write(f"# planete_{k}: {planetes[k]}\n")

        writer = csv.writer(f)

        # Écrire l'en-tête uniquement si le fichier est vide
        if fichier_vide:
            writer.writerow(["iteration_i", "temps_secondes"])

        for i in range(long):
            start = perf_counter()
            for j in range(haut):
                #Calcul de la collision
                planetes_act = copy.deepcopy(planetes)
                planetes_act[i_sat].pos = np.array([i,j],dtype=float)

                save[i][j] = crash(planetes_act,res,i_sat,G,t_max)
                del planetes_act #je ne suis pas sur de l'utilité de cette commande. Mais dans le doute
            duree = perf_counter() - start
            writer.writerow([i, round(duree, 4)])  # stockage CSV
            f.flush()  # force l'écriture immédiate, utile si le programme plante

            #sauvegarde progressive de l'image
            arr = np.array(couleur[save], dtype=np.uint8)
            img = Image.fromarray(arr, mode="RGB")
            img.save(fichier_img)

            print(i,duree)
    print(f"fini en {round((perf_counter() - start0)//60)} min et {round((perf_counter() - start0)%60)} s")


    