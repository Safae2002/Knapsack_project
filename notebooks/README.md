# 🎒 Solveur du Problème du Sac à Dos (0-1)

Un projet Python complet pour résoudre le problème du sac à dos avec différentes méthodes complètes et incomplètes.

## 📋 Description

Ce projet implémente plusieurs algorithmes pour résoudre le problème classique du sac à dos 0-1 :
- **Méthodes complètes** : garantissent l'optimalité (Branch and Bound, Programmation Dynamique, MIP)
- **Méthodes incomplètes** : heuristiques rapides (Glouton, Recuit Simulé, Algorithmes Génétiques, etc.)

L'objectif est de comparer les performances (temps, qualité de solution) de différentes approches sur des instances de difficulté variée.

## 🏗️ Structure du projet

knapsack_project_python/
├── knapsack_solver.py # Programme principal
├── knapsack_methods.py # Implémentations des méthodes
├── instance_loader.py # Chargement des instances
├── results_analyzer.py # Analyse des résultats
├── run_experiment.py # Script d'exécution
├── requirements.txt # Dépendances Python
├── notebooks/
│ └── results_analysis.ipynb # Notebook d'analyse
├── data/
│ └── kplib/ # Instances de benchmark
└── results/ # Résultats générés