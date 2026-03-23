import var_poo as var
import fun_iter as fun
import pygame as pg 
pg.init()
surf = pg.display.set_mode((var.haut,var.haut))

continuer = True
save = -1

while continuer:
    save, continuer = fun.actual(var.planetes,var.t,var.res,var.i_sat)
    var.t+=var.dt
    #si la touche q est pressée, on quitte
    for event in pg.event.get():
        if event.type == pg.QUIT :
            continuer = False
    touches = pg.key.get_pressed()
    if touches[pg.K_q]:
        continuer = False
    surf.fill((0,0,0))
    #on affiche le temps
    text = pg.font.Font(None, 20).render(f"Temps: {var.t}", True, (255, 255, 255))
    surf.blit(text, (10, 10))
    

    for i in range(var.planetes.shape[0]):
            pg.draw.circle(surf,var.planetes[i].couleur,(var.planetes[i].pos[0],var.planetes[i].pos[1]),var.planetes[i].r)

    pg.display.update()
print(save)
pg.quit()
