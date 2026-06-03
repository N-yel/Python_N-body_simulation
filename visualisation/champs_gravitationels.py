import numpy as np
import csv
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from time import perf_counter
from copy import deepcopy
from PIL import Image


class Calcul_Champs():
    def __init__(self,integrator,enregistrer_temps=False):
        self.integrator=integrator
        self.enregistrer_temps=enregistrer_temps
        self._timestamp = datetime.now().strftime(r"%Y%m%d_%H%M%S")
        self._nom_csv   = f"temps_calcul_{self._timestamp}.csv"
        self._nom_image = f"champ_gravitationnel_{self._timestamp}.png"

    
    def crash(self):
        continuer = True
        save = -2
        while self.integrator.scenario.t < self.integrator.scenario.t_max:
            if not continuer:
                break
            #on ajoute la valeur dans la liste que si t est entier
            save, continuer = self.integrator.step()
            self.integrator.scenario.t+=self.integrator.scenario.dt
        return save
    

    def _calculer_iter(self):
        assert(self.integrator.scenario.i_sat != -1)
        assert(self.integrator.scenario.planetes[0].pos.shape[0] == 2)
        save = np.full((self.integrator.scenario.long,self.integrator.scenario.haut),-2,dtype = int)
        couleurs = self.integrator.scenario.couleurs()
        start0 = perf_counter()

        fichier_image = Path("Images") / Path(self._nom_image)
        fichier_image.parent.mkdir(parents=True, exist_ok=True) #crée dossier Image si il n'existe pas
        if self.enregistrer_temps:
            fichier_csv = Path("temps") /  Path(self._nom_csv)
            fichier_csv.parent.mkdir(parents=True, exist_ok=True)
            with (open(fichier_csv, "w", newline="",encoding="utf-8") if self.enregistrer_temps else nullcontext()) as f:
                if f is not None:
                    f.write(f"# Integrateur = {self.integrator.nom} \n")
                    f.write(f"# Scenario = {self.integrator.scenario.nom} \n")
                    f.write(f"# dt = {self.integrator.scenario.dt}, t_max = {self.integrator} \n")
                    f.write(f"# long ={self.integrator.scenario.long},{self.integrator.scenario.haut} \n")
                    temps = csv.writer(f)
                    temps.writerow(["ligne","temps"])

                for i in range(self.integrator.scenario.long):
                    start = perf_counter()
                    for j in range(self.integrator.scenario.haut):
                        #Calcul de la collision
                        scenario_act = deepcopy(self)
                        scenario_act.integrator.scenario.planetes[self.integrator.scenario.i_sat].pos = np.array([i-self.integrator.scenario.long/2,j-self.integrator.scenario.haut/2],dtype=float)
                        save[i][j] = scenario_act.crash()
                        del scenario_act #je ne suis pas sur de l'utilité de cette commande. Mais dans le doute
                    duree = perf_counter() - start
                    # print(duree)
                    if f is not None:
                        temps.writerow([f"{i}",f"{duree}"])
                        f.flush()

                    #sauvegarde progressive de l'image
                    arr = np.array(couleurs[save], dtype=np.uint8)
                    img = Image.fromarray(arr, mode="RGB")
                    img.save(fichier_image)
        print(f"fini en {round((perf_counter() - start0)//60)} min et {round((perf_counter() - start0)%60)} s")

    def _calculer_parall(self):
        nb_fini_prec = 0
        compteur_fini_act=0
        couleurs = self.integrator.scenario.couleurs()
        tab_save =  np.full((self.integrator.scenario.long,self.integrator.scenario.haut),-2)
        start = perf_counter()

        fichier_image = Path("Images") / Path(self._nom_image)
        fichier_image.parent.mkdir(parents=True, exist_ok=True) #crée dossier Image si il n'existe pas
        fichier_csv = Path("temps") /  Path(self._nom_csv)
        fichier_csv.parent.mkdir(parents=True, exist_ok=True)
        with (open(fichier_csv, "w", newline="",encoding="utf-8") if self.enregistrer_temps else nullcontext()) as f:
            if f is not None:
                f.write(f"# Integrateur = {self.integrator.nom} \n")
                f.write(f"# Scenario = {self.integrator.scenario.nom} \n")
                f.write(f"# dt = {self.integrator.scenario.dt}, t_max = {self.integrator} \n")
                f.write(f"# long ={self.integrator.scenario.long},{self.integrator.scenario.haut} \n")
                temps = csv.writer(f)
                temps.writerow(["nombre de points calculés","temps"])

            while self.integrator.scenario.t <= self.integrator.scenario.t_max:
                nb_fini = np.count_nonzero(~self.integrator.tab_continuer)
                compteur_fini_act+= nb_fini - nb_fini_prec
                nb_fini_prec = nb_fini
                #on arrete si on a trouvé 95% des trajectoires ont été calculés : il ne reste plus que les trajectoires périodiques
                if nb_fini >= self.integrator.scenario.long*self.integrator.scenario.haut*0.95:
                    break
                if compteur_fini_act >= self.integrator.scenario.long and f is not None:
                        temps.writerow([f"{compteur_fini_act}",f"{perf_counter() - start}"])
                        start = perf_counter()
                        f.flush()
                        compteur_fini_act=0
                


                tab_step = self.integrator.step()

                nouveaux_finis = (tab_step != -2) & (tab_save == -2)
                tab_save[nouveaux_finis] = tab_step[nouveaux_finis]
                self.integrator.scenario.t += self.integrator.scenario.dt


                #sauvegarde progressive de l'image
                arr = np.array(couleurs[tab_save], dtype=np.uint8)
                img = Image.fromarray(arr, mode="RGB")
                img.save(fichier_image)

        

    def calculer(self):
        if self.integrator.nom !=  "parallel_integrator":
            self._calculer_iter()
        else:
            self._calculer_parall()

        



