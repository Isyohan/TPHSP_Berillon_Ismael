Hardware for Signal Prossesing TP

Théo BERILLON & Yohan ISMAEL
TP1
On réalise une multiplication de matrices initialisées aléatoirement (float entre -1 et 1) de dimension 500.
Durée moyenne de calcul avec GPU = 0.205s (sur 5 éxecutions)


n=2000

CPU : 1m40s=100s

GPU : 0.415s

accélération pratique -> 240

On réalise n*n calculs en parallèle.

TP2 :

Convolution sur matrices de petites tailles : 2*2, 3*3, 1*1 avec succès pour des tailles de noyaux différentes (1*1,2*2,3*3)


Conv + Maxpooling avec n = 20 000:

CPU : 2m53.011s= 173.011s
GPU : 8.227s

accélération pratique -> 21

On pourrait s'attendre à mieux, il y a plusirus améliorations possible dans notre code.
