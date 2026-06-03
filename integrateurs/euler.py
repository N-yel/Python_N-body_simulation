from integrateurs.base import BaseIntegrator
import numpy as np

class Euler_integrator(BaseIntegrator):
    def __init__(self,scenario):
        self.scenario = scenario
        self.nom = "Euler"

    def calcul_force(self,k,i,res,G,d):
        """calcule la force qu'exerce p2 sur p1 divisé par la masse de p1"""
        if d==0:
            return np.zeros_like(self.scenario.planetes[k].pos)
        F = []
        for c in range(len(self.scenario.planetes[0].pos)):
            F.append(self.scenario.planetes[i].masse *G *(self.scenario.planetes[i].pos[c] - self.scenario.planetes[k].pos[c])*(1/d**3) -res*self.scenario.planetes[k].vit[c])
        return np.array(F)

    def update(self):
        n = len(self.scenario.planetes)
        for k in range(n):
            if self.scenario.planetes[k].fixe:
                continue
            self.scenario.planetes[k].vit += self.scenario.planetes[k].acc*self.scenario.dt
            self.scenario.planetes[k].pos += self.scenario.planetes[k].vit*self.scenario.dt

    def step(self):
        # sys.shape = (len(sys),2) : on stocke un tableau couples de deux coordonées, on pose ci-dessous n = nombre de planètes
        n = len(self.scenario.planetes)
        save = -2
        continuer = True

        # pour toutes les planètes du système, si elles ne sont pas fixes, on somme l'accélération liée aux autres planètes
        for k in range(n):
            sumk = np.zeros_like(self.scenario.planetes[0].pos)
            if self.scenario.planetes[k].fixe:
                continue
            for i in range(n):
                if k==i:
                    continue
                # d: distance entre la planète i et k
                d = self.scenario.planetes[i].dist(self.scenario.planetes[k])
                if d <= self.scenario.planetes[i].rayon + self.scenario.planetes[k].rayon:
                    if k == self.scenario.i_sat:
                        continuer=False
                        #si la position initiale a laquelle on lache le satelite est dans une des planètes, alors la couleur renvoyée est noire
                        if (self.scenario.t==0) and d <= self.scenario.planetes[i].rayon:
                            save = -1
                            return save,continuer
                        #si le satellite touche une planete, on renvoie la couleur associée à cette planète
                        else:
                            save = i
                            return save, continuer
                    else:
                        continue
                sumk += self.calcul_force(k, i,self.scenario.res,self.scenario.G,d)
            self.scenario.planetes[k].acc = sumk


        #actualisation de la position et de la vitesse
        self.update()
        return save, continuer

    
    def get_positions(self):
        return np.array([planete.pos for planete in self.scenario.planetes])