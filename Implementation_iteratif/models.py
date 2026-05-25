import numpy as np

class Planete :
    def __init__(self,pos,vit,acc,m,r,dt,fixe):
        self.pos = pos
        self.vit = vit
        self.acc = acc
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
            return [0,0]

        Fx = self.m *G *(self.pos[0] - p.pos[0])*(1/d**3) -res*p.vit[0]
        Fy = self.m *G *(self.pos[1] - p.pos[1])*(1/d**3) -res*p.vit[1]
        return np.array([Fx,Fy])
        

    def update(self):
        self.vit += self.acc*self.dt
        self.pos += self.vit*self.dt

