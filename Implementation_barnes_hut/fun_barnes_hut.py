import models_barnes_hut as models

def actual(planetes,t,res,i_sat,G,taille,theta):
    n = planetes.shape[0]
    save = -1
    continuer = True

    for k in range(n):
        if not planetes[k].fixe:
            planetes[k].update_pos()
        
    arbre = models.Quadtree([0,0],taille)
    arbre.ajouter_planetes(planetes)
    

    for k in range(n):
        if not planetes[k].fixe:
            planetes[k].acc = arbre.calcul_force(planetes[k],theta,G)/planetes[k].m
            planetes[k].update_vit()

    return save, continuer

