#!/usr/bin/env python3
"""
Script simplifié pour exécuter l'expérimentation
"""

import argparse
import sys
import os

# Ajouter le chemin courant pour les imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from knapsack_solver import main as run_experiment


def parse_arguments():
    """Parse les arguments en ligne de commande"""
    parser = argparse.ArgumentParser(
        description="Solveur du problème du sac à dos avec différentes méthodes"
    )
    
    parser.add_argument(
        "--time-limit",
        type=int,
        default=300,
        help="Limite de temps par instance en secondes (défaut: 300)"
    )
    
    parser.add_argument(
        "--kplib-path",
        type=str,
        default="data/kplib",
        help="Chemin vers le dossier kplib (défaut: data/kplib)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="results/results.csv",
        help="Fichier de sortie pour les résultats (défaut: results/results.csv)"
    )
    
    parser.add_argument(
        "--only-complete",
        action="store_true",
        help="Exécuter seulement les méthodes complètes"
    )
    
    parser.add_argument(
        "--only-incomplete",
        action="store_true",
        help="Exécuter seulement les méthodes incomplètes"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # Afficher la configuration
    print(f"⚙️  Configuration:")
    print(f"   Limite de temps: {args.time_limit} secondes")
    print(f"   Chemin kplib: {args.kplib_path}")
    print(f"   Fichier de sortie: {args.output}")
    
    # Vérifier si kplib existe
    if not os.path.exists(args.kplib_path):
        print(f"⚠️  Attention: le dossier kplib n'existe pas à {args.kplib_path}")
        print("ℹ️  Vous pouvez le cloner avec:")
        print("   git clone https://github.com/jorlou/kplib.git data/kplib")
        response = input("Voulez-vous continuer avec des instances générées? (o/n): ")
        if response.lower() != 'o':
            sys.exit(1)
    
    # Lancer l'expérimentation
    try:
        # Note: pour appliquer les arguments, vous devez modifier knapsack_solver.py
        # ou créer une fonction main avec paramètres
        # Import classes directly
        from knapsack_solver import ExperimentRunner
        
        # Initialize runner with args
        runner = ExperimentRunner(time_limit_ms=args.time_limit * 1000)
        
        # Load instances
        if not runner.load_instances(args.kplib_path):
             sys.exit(1)
             
        # Run experiments based on flags
        if not runner.instances:
            print("❌ Aucune instance à traiter!")
            sys.exit(1)
            
        print(f"\n🚀 Démarrage des expériences sur {len(runner.instances)} instances")
        print("=" * 80)
        
        from tqdm import tqdm
        import time
        
        for i, instance in enumerate(tqdm(runner.instances, desc="Instances")):
            print(f"\n{'='*60}")
            print(f"Instance {i+1}/{len(runner.instances)}: {instance.name}")
            print(f"Taille: {instance.n}, Capacité: {instance.capacity}, Difficulté: {instance.difficulty}")
            print(f"{'='*60}")
            
            # Méthodes complètes
            if not args.only_incomplete:
                print("\n📈 MÉTHODES COMPLÈTES:")
                complete_results = runner.run_complete_methods(instance)
                runner.results.extend(complete_results)
            
            # Méthodes incomplètes
            if not args.only_complete:
                print("\n📉 MÉTHODES INCOMPLÈTES:")
                incomplete_results = runner.run_incomplete_methods(instance)
                runner.results.extend(incomplete_results)
                
            time.sleep(0.1)
            
        print(f"\n✅ Expériences terminées!")
        
        # Save and report
        runner.save_results(args.output)
        runner.generate_summary_report()
    except KeyboardInterrupt:
        print("\n\n❌ Expérimentation interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur lors de l'exécution: {e}")
        sys.exit(1)