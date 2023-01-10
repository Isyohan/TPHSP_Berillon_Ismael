# Hardware for Signal Prossesing TP Théo BERILLON & Yohan ISMAEL
## TP1 :
On réalise une multiplication de matrices initialisées aléatoirement (float entre -1 et 1) de dimension 500.
Durée moyenne de calcul avec GPU = 0.205s (sur 5 éxecutions)


n=2000

CPU : 1m40s=100s

GPU : 0.415s

accélération pratique -> 240

On réalise n*n calculs en parallèle.

## TP2 :

Convolution sur matrices de petites tailles : 2x2, 3x3, 1x1 avec succès pour des tailles de noyaux différentes (1x1,2x2,3x3)


## TP3 : Convolution et maxpooling
 
Conv + Maxpooling avec n = 20 000:

CPU : 2m53.011s= 173.011s

GPU : 8.227s

accélération pratique -> 21

On pourrait s'attendre à mieux, il y a plusieurs améliorations possible dans notre code.

## TP4 :

Tous les calculs d'une couche à l'autre sont parallélisables. En particulier les couches linéaires sont parallélisables facilement car c'est une mutliplication de matrice.

Réseau final : 32x32x1 --Conv--> 28x28x6 --MeanPooling--> 14x14x6 --Conv--> 10x10x16 --MeanPooling--> 5x5x16 --Flatten--> 400 --Dense(tanh)--> 120 --Dense(tanh)--> 84 --Dense(softmax)--> 10 (nombre de labels de MNIST)

