# Solveur du Problème du Sac à Dos (0-1)

Un projet Python complet pour résoudre le problème du sac à dos avec différentes méthodes complètes et incomplètes.

---

## Description

Ce projet implémente plusieurs algorithmes pour résoudre le problème classique du sac à dos 0-1 :

### ✅ Méthodes complètes (garantissent l’optimalité)
- Programmation Dynamique  
- Branch and Bound  
- MIP (Programmation Linéaire en Nombres Entiers)

### ⚡ Méthodes incomplètes (heuristiques rapides)
- Algorithme Glouton (simple, K=3 et probabiiste alpha = 0.9)
- Recuit Simulé  
- Algorithmes Génétiques
- Recherche Tabou

L’objectif est de comparer les performances (temps d’exécution, qualité de la solution) des différentes approches sur des instances de difficulté variée.

---

## Structure du projet


knapsack_project_python/
│
├── knapsack_solver.py # Programme principal
├── knapsack_methods.py # Implémentations des méthodes
├── instance_loader.py # Chargement des instances
├── results_analyzer.py # Analyse des résultats
├── run_experiment.py # Script d'exécution
├── requirements.txt # Dépendances Python
│
├── notebooks/
│ └── results_analysis.ipynb # Notebook d'analyse
│
├── data/
│ └── kplib/ # Instances de benchmark
│
├── results/ # Résultats générés
│
└── presentation/
└── knapsack_presentation.pdf # Présentation du projet (PDF)


---

##  Installation et dépendances

Toutes les bibliothèques nécessaires se trouvent dans le fichier : requirements.txt


Pour installer les dépendances :

```bash
pip install -r requirements.txt

## Exécution du projet

Pour lancer les expériences :

python run_experiment.py


Pour analyser les résultats :

python results_analyzer.py --results results/results.csv

## Présentation du projet

La présentation du projet est disponible ici : knapsack_presentation.pdf
