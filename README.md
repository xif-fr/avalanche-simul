# avalanche-simul
2D simulation of an avalanche as a density-variable fluid

# Liens utiles :

- Schéma plus avancé pour les fluides à densité variable : https://www.ljll.math.upmc.fr/~frey/papers/Navier-Stokes/Fraigneau%20Y.,%20Guermond%20J.L.,%20Quartapelle%20L.,%20Approximation%20of%20variable%20density%20incompressible%20flows%20by%20means%20of%20finite%20elements%20and%20finite%20volumes.pdf
- Poly de mécanique des fluides fondamentale : https://perso.limsi.fr/wietze/cours/MF/meca_flu_poly2020-2021.pdf (eq. 1.87 pour Navier-Stokes généralisé)


# To do:

1. Schéma "naïf" à viscosité constante : fractional time stepping avec semi-lag pour advecter la vitesse et la densité; domaine rectangulaire avec cond. lim. outflow
2. Analyser l'ordre, la stabilité, la convergence. Validation ?
3. Explorer les comportements, ébauche de diagramme. Taux de mixing suivant les paramètres. Diffusion de la densité.
4. Tester la diffusion numérique avec un simple mode de Fourier.

Au choix :

5. Implémenter le schéma plus avancé, toujours à viscosité constante, et vérifier numériquement la stabilité et l'ordre.

Ou :

5. Viscosité variable (voire loi non-newtonienne, c'est pas grand chose de plus) fonction du champ de densité. Penser à la grille duale pour ça pour simplifier/symétriser l'implémentation du ∇(μ∇u)
