import numpy as np
import var_poo as var
import models
from time import perf_counter

update_list = np.vectorize (lambda planete : planete.update())


def actual(planetes,t,res,i_sat):
    # sys.shape = (len(sys),2) : on stocke un tableau couples de deux coordonées, on pose ci-dessous n = nombre de planètes
    n = planetes.shape[0]
    save = None
    continuer = True

    #si cela fait trop longtemps, alors on renvoie (42,42,42) (une sorte de gris)
    if t>=9999:
        return -2 , False
    
    # pour toutes les planètes du système, si elles ne sont pas fixes, on somme l'accélération liée aux autres planètes
    for k in range(n):
        sumk = np.array([0,0],dtype=float)
        if(not planetes[k].fixe):
            for i in range(n):
                # d: distance entre la planète i et k
                d = planetes[i].dist(planetes[k])

                #cas particulier, la planète que l'on observe est le satelite, alors on sauvegarde la planète dans laquelle elle se crashe avec la variable save. On ne traite pas le cas ou il y a plusieurs collisions en même temps sachant qu'il est plutot improbable en vue de ce que l'on veut simuler
                if (k == i_sat) and (i != i_sat):
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
                if k!=i and (d > planetes[k].r + planetes[i].r):
                    #on somme la composante de l'accélération selon toute les autre planètes et on soustrait un frottement fluide pour faire baisser artificiellement l'energie (a voir)
                    sumk += planetes[i].calcul_force(planetes[k],res)
            planetes[k].acc = sumk

    update_list(planetes)
    return save, continuer



#renvoie la planète si il y en a une vers laquelle le satellite a aterri, sinon (42,42,42)
def crash(sys,sysv,dt,res,m,r,fixe,i_sat):
    t = 0
    continuer = True
    save = None
    while continuer:
        save, continuer = actual(sys,sysv,t,dt,res,m,r,fixe,i_sat)
        # print(f"positions = {sys}")
        # print(f"vitesses = {sysv}")
        t+=dt
    # print(f"la couleur censé être renvoyé sera {var.couleur[save]}")
    return save



#va renvoyer un tableau de dimension (long,haut) avec la couleur de la planète vers laquelle le satellite d'indice i_sat va atterir en fonction de sa position
def calculer(sys,sysv,long,haut,dt,res,i_sat,m,r,fixe):
    start0 = perf_counter()
    #img: va stocker en la couleur en fct des conditionss initiales
    img = [[(0,0,0) for _ in range(haut)] for _ in range(long)]
    for i in range(long): #on est censé mettre long ici
        start = perf_counter()
        for j in range(haut): # on est censé mettre haut ici si on veut tt le tableau
            copy_pos = sys.copy()
            copy_pos[i_sat] = [i,j]
            img[i][j] = var.couleur[crash(copy_pos,sysv.copy(),dt,res,m,r,fixe,i_sat)]
            #print(f"le pixel dans l'image sera {img[i][j]}")
        print(i,(perf_counter() - start)) 
    print(f"fini en {round((perf_counter() - start0)//60)} min et {round((perf_counter() - start0)%60)} s")
    return img