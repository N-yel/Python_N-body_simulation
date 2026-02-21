"""
this file is used to generate images
"""
import numpy as np
from PIL import Image
import fun1 as fun
import var1 as var

nom = "graphe_"+"i_sat = "+ str(var.i_sat) + "sysv = "+ str(var.sysv) +"_sysa = "+ str(var.sysa) +"_res = "+ str(var.res) +"_dt = "+ str(var.dt) +"_m = "+ str(var.m)+"_r = "+ str(var.r) +".png"
if __name__ == "__main__":
    tab = fun.calculer(var.sys,var.sysv,var.long,var.haut,var.dt,var.t_max,var.res,var.i_sat,var.m,var.r,var.fixe)
    arr = np.array(tab, dtype=np.uint8)  # forme (h, w, 3)
    img = Image.fromarray(arr, mode="RGB")
    img.save(nom)