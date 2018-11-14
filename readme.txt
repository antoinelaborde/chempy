Chempy version 0.1
Date: 18/11/04

News: 
ALA dans util:
- update des fonctions select/delete (gestion par index)
- fonction grouping
- fonction divmin, divmax, mean, sum, check_duplicate


ALA dans classes:
- gestion des erreurs dans la classe div. Classe Div plus robuste.
- Attention à la méthode de création des Div (même pour les développeurs). Utilisez l'init de la classe SVP.
Faire comme ça: my_div = Div(d = my_array, i = my_i, v = my_v) 
Pas comme ça: my_div = Div(); my_div.d = my_array etc... 
Sinon on perd l'automaticité de la vérification des champs d,i,v


A faire:
-> PCA class: interaction avec fonction PCA ? apply pca etc. ? classe générique (PCA, PLS, comdim etc.)
-> import functions: renommer, restructurer ?

Penser à faire un git !