import numpy as np
from scenarios import scenario_3corps as s
from integrateurs import Parallel_Integrator as Integrator
from simulation import Simulation
# from visualisation import animer
from visualisation import Calcul_Champs

integrateur = Integrator(scenario=s)
# simu = Simulation()
# trajectoire = simu.run_trajectoire(integrator=integrateur)
# trajectoire_np = np.array(trajectoire)

# animer(trajectoire=trajectoire_np,scenario=s)
c = Calcul_Champs(integrator=integrateur,enregistrer_temps=True)
c.calculer()

