import numpy as np

#cette classe marche aussi en 3d

class Planete :
    def __init__(self,pos,vit,acc,m,r,dt,fixe):
        self.pos = pos
        self.vit = vit
        self.acc = acc
        self.acc_old = np.array([0 for _ in pos],dtype=float)
        self.m = m
        self.r = r
        self.dt = dt
        self.fixe = fixe
    
    # renvoie en texte la représentation d'une planète (utile pour le fichier csv)
    def __repr__(self):
        return f"pos={self.pos}, vit={self.vit}, m={self.m}, r={self.r}, dt={self.dt}, fixe={self.fixe}"

    def dist(self,p):
        return np.linalg.norm(self.pos - p.pos)



    def calcul_force(self,p,res,G):
        d = self.dist(p)
        if d <= self.r + p.r:
            return np.array([0 for _ in range(len(self.pos))])

        F = []
        for i in range(len(self.pos)):
            F.append(self.m *G *(self.pos[i] - p.pos[i])*(1/d**3) -res*self.vit[i])
        return np.array(F)
        



    def update_pos(self):
        self.pos += self.vit * self.dt + 0.5 * self.acc * self.dt**2
        self.acc_old = self.acc.copy()  # on sauvegarde a(t)

    def update_vit(self):
        # Étape 3 : v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
        self.vit += 0.5 * (self.acc_old + self.acc) * self.dt

