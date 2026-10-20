from integrateurs import Parallel_Integrator
import numpy as np

class Simulation:
    def _run_trajectoire_iter(self,integrator):
        """retourne un tableau de positions dans le temps"""
        continuer = True
        positions = []
        while integrator.scenario.t < integrator.scenario.t_max:
            if not continuer:
                break
            integrator.scenario.t+=integrator.scenario.dt
            #on ajoute la valeur dans la liste que si t est entier
            if round(integrator.scenario.t,5) % 1 == 0:
                #si on fait juste traj.append(var.sys), cela renvoie un pointeur tjr vers la même valeur
                positions.append(integrator.get_positions())
            _, continuer = integrator.step()
        return positions
    
    def _run_trajectoire_parall(self,integrator):
        assert(integrator.scenario.i_sat != -1 and len(integrator.scenario.planetes[0].pos)==2)
        pos = integrator.scenario.planetes[integrator.scenario.i_sat].pos
        integrator.scenario.long = 1
        integrator.scenario.haut = 1
        nv_integrator = Parallel_Integrator(integrator.scenario)
        nv_integrator.tab_pos = np.array([pos])
        positions = []
        while nv_integrator.scenario.t < nv_integrator.scenario.t_max:
            if not nv_integrator.tab_continuer[0]:
                break
            positions.append(nv_integrator.get_positions())
            nv_integrator.step()
            nv_integrator.scenario.t += nv_integrator.scenario.dt
        return positions



    def run_trajectoire(self, integrator):
        if integrator.nom !=  "parallel_integrator":
            return self._run_trajectoire_iter(integrator)
        else:
            return self._run_trajectoire_parall(integrator)


