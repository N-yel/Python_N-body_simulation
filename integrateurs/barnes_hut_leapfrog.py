from integrateurs.base import BaseIntegrator
import integrateurs.barnes_hut_leapfrog
import numpy as np

# attention: mauvaise gestion des collisions avec cet intégrateur : lorsque des planètes se rapprochent, d -> 0 donc 1/d^3 -> +\infty. Les forces deviennent donc très grandes dans le cas d'une collision
class Quadtree:
    def __init__(self, centre, taille,dim):
        self.centre = np.array(centre, dtype=float)
        self.taille = taille          # demi-largeur du carré
        self.mtot = 0.0
        self.centre_masse = np.array([0.0 for _ in range(dim)])
        self.nb_planetes = 0
        self.dim = dim
        self.fils = [None for _ in range(2**dim)]
        self.planete = None                   # seulement si nœud feuille



    def get_quadrant(self, pos):
        """
        Retourne l'index du quadrant/octant et le centre du fils correspondant.
        
        En 2D : 4 quadrants (2^2)
        En 3D : 8 octants  (2^3)
        
        L'index est calculé en binaire :
        - bit 0 : x (0=gauche, 1=droite)
        - bit 1 : y (0=bas,    1=haut  )
        - bit 2 : z (0=avant,  1=arrière) [seulement en 3D]
        """
        quart = self.taille / 2
        idx = 0
        new_centre = np.copy(self.centre)

        for i in range(self.dim):
            if pos[i] >= self.centre[i]:
                idx += 2**i              # on met le bit i à 1
                new_centre[i] += quart   # on se déplace vers la droite/haut/arrière
            else:
                new_centre[i] -= quart   # on se déplace vers la gauche/bas/avant

        return idx, new_centre

    def est_feuille(self):
        return all(f is None for f in self.fils)

    def ajouter_planete(self, planete):
        #Mise à jour masse et barycentre
        #on calcule dans un premier temps la moyenne des position pondérée par la masse, puis on met à jour mtot pour enfin divisé la moyenne par mtot
        self.centre_masse = (self.centre_masse * self.mtot + planete.pos * planete.masse)
        self.mtot += planete.masse
        self.centre_masse /= self.mtot
        self.nb_planetes += 1

        #Deux cas de base: si il y a une feuille ou si il n'y en a aucune
        #Cas 1 : nœud vide -> on place la planète ici 
        if self.nb_planetes == 1:
            self.planete = planete
            return

        #Cas 2 : nœud feuille avec déjà une planète -> on subdivise
        if self.est_feuille():
            # On replace l'ancienne planète dans le bon fils
            ancienne = self.planete
            self.planete = None

            idx, new_centre = self.get_quadrant(ancienne.pos)
            self.fils[idx] = Quadtree(new_centre, self.taille / 2,self.dim)
            self.fils[idx].ajouter_planete(ancienne)

        #Cas 3 : nœud interne (ou après subdivision) -> on descend
        idx, new_centre = self.get_quadrant(planete.pos)
        if self.fils[idx] is None:
            self.fils[idx] = Quadtree(new_centre, self.taille / 2,self.dim)
        self.fils[idx].ajouter_planete(planete)
    
    def ajouter_planetes(self,planetes):
        for planete in planetes:
            self.ajouter_planete(planete)


    def calcul_force(self, planete, theta, G):
        # Nœud vide
        if self.nb_planetes == 0:
            return np.array([0.0 for _ in range(self.dim)])

        # On ne calcule pas la force d'une planète sur elle-même
        if self.nb_planetes == 1 and self.planete is planete:
            return np.array([0.0 for _ in range(self.dim)])

        d = np.linalg.norm(planete.pos - self.centre_masse)
        s = self.taille * 2

        # Nœud externe OU critère d'approximation satisfait
        if self.nb_planetes == 1 or (s / d) < theta:
            delta = self.centre_masse - planete.pos
            force = G * planete.masse * self.mtot / d**3 * delta
            return force

        # Sinon on descend dans les fils
        else:
            force = np.array([0.0 for _ in range(self.dim)])
            for fils in self.fils:
                if fils is not None:
                    force += fils.calcul_force(planete, theta, G)
            return force
        
class Barnes_hut_leapfrog_Integrator(BaseIntegrator):
    def __init__(self,scenario):
        self.scenario = scenario
        self.planetes = self.scenario.planetes
        self.nom = "Barnes_hut_leapfrog"
    
    def update_pos(self):
        for p in self.planetes:
            if p.fixe :
                continue
            p.pos += p.vit * self.scenario.dt + 0.5 * p.acc * self.scenario.dt**2
            p.acc_old = p.acc.copy()  # on sauvegarde a(t)

    def update_vit(self):
        for p in self.planetes:
            if p.fixe:
                continue
            p.vit += 0.5 * (p.acc_old + p.acc) * self.scenario.dt

    def step(self,theta=0.5):
        n = len(self.planetes)
        dim = self.planetes[0].pos.shape[0]
        save = -1
        continuer = True

        self.update_pos()
            
        arbre = Quadtree([0.0 for _ in range(dim)],self.scenario.long,dim)
        arbre.ajouter_planetes(self.planetes)
        

        for k in range(n):
            # Si k == i_sat: pour detecter les collisions on fait la méthode classique
            if k == self.scenario.i_sat:
                sumk = np.array([0.0 for _ in range(dim)],dtype=float)
                for i in range(n):
                    if k == i:
                        continue
                    d = self.planetes[i].dist(self.planetes[k])
                    if d <= self.planetes[i].rayon + self.planetes[k].rayon:
                        continuer=False
                        #si la position initiale a laquelle on lache le satelite est dans une des planètes, alors la couleur renvoyée est noire
                        if (self.scenario.t==0) and d <= self.planetes[i].rayon:
                            save = -1
                            return save,continuer
                        #si le satellite touche une planete, on renvoie la couleur associée à cette planète
                        else:
                            save = i
                            return save, continuer
                    F = []
                    for c in range(dim):
                        F.append(self.planetes[i].masse *self.scenario.G *(self.planetes[i].pos[c] - self.planetes[k].pos[c])*(1/d**3) -self.scenario.res*self.planetes[k].vit[c])
                    sumk += np.array(F)
                self.planetes[k].acc = sumk
            else:
                if not self.planetes[k].fixe:
                    self.planetes[k].acc = arbre.calcul_force(self.planetes[k],theta,self.scenario.G)/self.planetes[k].masse
            
        self.update_vit()
        return save, continuer
    
    def get_positions(self):
        return np.array([p.pos for p in self.planetes])

    
    
