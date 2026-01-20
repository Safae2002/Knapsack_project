#!/usr/bin/env python3
"""
Solveur principal pour le problème du sac à dos
Implémente plusieurs méthodes complètes et incomplètes
"""

import time
import json
import pandas as pd
from typing import List, Dict
from tqdm import tqdm
import os
import sys

# Ajouter le chemin courant pour les imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from instance_loader import InstanceLoader, KnapsackInstance
from knapsack_methods import KnapsackSolver, KnapsackResult


class ExperimentRunner:
    """Exécute les expériences et collecte les résultats"""
    
    def __init__(self, time_limit_ms: int = 300000):  # 5 minutes par défaut
        self.solver = KnapsackSolver(time_limit_ms)
        self.results: List[KnapsackResult] = []
        self.instances: List[KnapsackInstance] = []
    
    def load_instances(self, kplib_path: str = "data/kplib"):
        """Charge les instances depuis kplib"""
        print("📂 Chargement des instances...")
        self.instances = InstanceLoader.load_benchmark_instances(kplib_path)
        
        if not self.instances:
            print("❌ Aucune instance chargée!")
            return False
        
        print(f"✅ {len(self.instances)} instances chargées")
        
        # Afficher un résumé
        by_difficulty = {}
        for inst in self.instances:
            by_difficulty[inst.difficulty] = by_difficulty.get(inst.difficulty, 0) + 1
        
        for diff, count in by_difficulty.items():
            print(f"  - {diff}: {count} instances")
        
        return True
    
    def run_complete_methods(self, instance: KnapsackInstance) -> List[KnapsackResult]:
        """Exécute toutes les méthodes complètes sur une instance"""
        results = []
        
        print(f"\n🔍 Résolution de '{instance.name}' (n={instance.n}, capacité={instance.capacity})")
        
        # 1. MIP avec OR-Tools
        print("  🧮 MIP OR-Tools...", end=" ", flush=True)
        result = self.solver.solve_complete_mip_ortools(instance)
        results.append(result)
        print(f"✓ Valeur: {result.best_value}, Temps: {result.time_ms:.0f}ms")
        
        # 2. MIP avec PuLP (seulement pour petites instances)
        if instance.n <= 100:
            print("  🧮 MIP PuLP...", end=" ", flush=True)
            result = self.solver.solve_complete_mip_pulp(instance)
            results.append(result)
            print(f"✓ Valeur: {result.best_value}, Temps: {result.time_ms:.0f}ms")
        
        # 3. Branch and Bound personnalisé (seulement pour petites instances)
        if instance.n <= 30:
            print("  🌳 Branch and Bound...", end=" ", flush=True)
            result = self.solver.solve_complete_branch_and_bound(instance)
            results.append(result)
            print(f"✓ Valeur: {result.best_value}, Temps: {result.time_ms:.0f}ms")
        
        # 4. Programmation dynamique (seulement si capacité raisonnable)
        if instance.capacity <= 10000 and instance.n <= 500:
            print("  📊 Programmation dynamique...", end=" ", flush=True)
            result = self.solver.solve_complete_dp(instance)
            results.append(result)
            print(f"✓ Valeur: {result.best_value}, Temps: {result.time_ms:.0f}ms")
        
        return results
    
    def run_incomplete_methods(self, instance: KnapsackInstance) -> List[KnapsackResult]:
        """Exécute toutes les méthodes incomplètes sur une instance"""
        results = []
        
        # 1. Glouton simple
        print("  🎯 Glouton simple...", end=" ", flush=True)
        result = self.solver.solve_incomplete_greedy_simple(instance)
        results.append(result)
        print(f"✓ Valeur: {result.best_value}, Temps: {result.time_ms:.0f}ms")
        
        # 2. Glouton aléatoire (k-meilleurs)
        print("  🎲 Glouton aléatoire (k=3)...", end=" ", flush=True)
        result = self.solver.solve_incomplete_greedy_random(instance, k=3)
        results.append(result)
        print(f"✓ Valeur: {result.best_value}, Temps: {result.time_ms:.0f}ms")
        
        # 3. Glouton probabiliste
        print("  🎰 Glouton probabiliste (alpha=0.9)...", end=" ", flush=True)
        result = self.solver.solve_incomplete_greedy_probabilistic(instance, alpha=0.9)
        results.append(result)
        print(f"✓ Valeur: {result.best_value}, Temps: {result.time_ms:.0f}ms")
        
        # 4. Recuit simulé (seulement pour instances moyennes/grandes)
        if instance.n >= 50:
            print("  🔥 Recuit simulé...", end=" ", flush=True)
            result = self.solver.solve_incomplete_simulated_annealing(instance)
            results.append(result)
            print(f"✓ Valeur: {result.best_value}, Temps: {result.time_ms:.0f}ms")
        
        # 5. Algorithme génétique
        if instance.n >= 100:
            print("  🧬 Algorithme génétique...", end=" ", flush=True)
            result = self.solver.solve_incomplete_genetic_algorithm(instance)
            results.append(result)
            print(f"✓ Valeur: {result.best_value}, Temps: {result.time_ms:.0f}ms")
        
        # 6. Recherche Tabu
        if instance.n >= 50:
            print("  📋 Recherche Tabu...", end=" ", flush=True)
            result = self.solver.solve_incomplete_tabu_search(instance)
            results.append(result)
            print(f"✓ Valeur: {result.best_value}, Temps: {result.time_ms:.0f}ms")
        
        return results
    
    def run_all_experiments(self):
        """Exécute toutes les expériences sur toutes les instances"""
        if not self.instances:
            print("❌ Aucune instance à traiter!")
            return
        
        print(f"\n🚀 Démarrage des expériences sur {len(self.instances)} instances")
        print("=" * 80)
        
        for i, instance in enumerate(tqdm(self.instances, desc="Instances")):
            print(f"\n{'='*60}")
            print(f"Instance {i+1}/{len(self.instances)}: {instance.name}")
            print(f"Taille: {instance.n}, Capacité: {instance.capacity}, Difficulté: {instance.difficulty}")
            print(f"{'='*60}")
            
            # Méthodes complètes
            print("\n📈 MÉTHODES COMPLÈTES:")
            complete_results = self.run_complete_methods(instance)
            self.results.extend(complete_results)
            
            # Méthodes incomplètes
            print("\n📉 MÉTHODES INCOMPLÈTES:")
            incomplete_results = self.run_incomplete_methods(instance)
            self.results.extend(incomplete_results)
            
            # Petite pause pour la lisibilité
            time.sleep(0.1)
        
        print(f"\n✅ Expériences terminées!")
        print(f"📊 {len(self.results)} résultats collectés")
    
    def save_results(self, output_file: str = "results/results.csv"):
        """Sauvegarde les résultats dans un fichier CSV"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Convertir en DataFrame
        data = []
        for result in self.results:
            row = result.to_dict()
            
            # Ajouter des informations de l'instance
            instance = next((inst for inst in self.instances 
                           if inst.name == result.instance_name), None)
            if instance:
                row.update({
                    'Difficulty': instance.difficulty,
                    'Class': 'Unknown',  # Default class
                    'n': instance.n,
                    'Capacity': instance.capacity,
                    'TotalWeight': sum(instance.weights),
                    'TotalProfit': sum(instance.profits)
                })
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Réorganiser les colonnes
        cols = ['Instance', 'Difficulty', 'Class', 'n', 'Capacity', 'TotalWeight', 
                'TotalProfit', 'Method', 'Value', 'Time(ms)', 'Nodes', 
                'Optimal', 'OptimalKnown', 'Gap(%)']
        
        # Garder seulement les colonnes présentes
        cols = [col for col in cols if col in df.columns]
        df = df[cols]
        
        # Sauvegarder
        df.to_csv(output_file, index=False)
        print(f"💾 Résultats sauvegardés dans {output_file}")
        
        # Sauvegarder aussi en JSON pour une analyse plus facile
        json_file = output_file.replace('.csv', '.json')
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"💾 Résultats JSON sauvegardés dans {json_file}")
        
        return df
    
    def generate_summary_report(self):
        """Génère un rapport de synthèse"""
        if not self.results:
            print("❌ Aucun résultat à analyser!")
            return
        
        print("\n" + "="*80)
        print("📋 RAPPORT DE SYNTHÈSE")
        print("="*80)
        
        # Convertir en DataFrame pour l'analyse
        df = pd.DataFrame([r.to_dict() for r in self.results])
        
        # Convertir optimal en booléen pour les calculs
        df['Optimal_Bool'] = df['Optimal'] == 'true'
        
        # 1. Identifier la meilleure valeur trouvée pour chaque instance
        best_values = df.groupby('Instance')['Value'].max().to_dict()
        
        # 2. Statistiques par méthode
        print("\n📊 PERFORMANCE PAR MÉTHODE:")
        print("-" * 80)
        
        summary_data = []
        for method in df['Method'].unique():
            method_df = df[df['Method'] == method]
            
            avg_value = method_df['Value'].mean()
            avg_time = method_df['Time(ms)'].mean()
            optimal_rate = method_df['Optimal_Bool'].mean() * 100
            
            # Calculer le gap moyen de manière robuste
            gaps = []
            for _, row in method_df.iterrows():
                # Référence = OptimalKnown si > 0, sinon Meilleure Valeur trouvée
                ref = row.get('OptimalKnown', -1)
                if ref <= 0:
                    ref = best_values.get(row['Instance'], 0)
                
                if ref > 0:
                    gap = (ref - row['Value']) / ref * 100
                    gaps.append(gap)
                else:
                    gaps.append(0.0)
            
            avg_gap = sum(gaps) / len(gaps) if gaps else 0.0
            
            summary_data.append({
                'Méthode': method,
                'Valeur moyenne': f"{avg_value:.0f}",
                'Temps moyen (ms)': f"{avg_time:.0f}",
                'Taux optimal (%)': f"{optimal_rate:.1f}%",
                'Gap moyen (%)': f"{avg_gap:.2f}%"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
            
        
        # Statistiques par difficulté
        if 'Difficulty' in df.columns:
            print("\n🎯 PERFORMANCE PAR DIFFICULTÉ:")
            print("-" * 80)
            
            for difficulty in df['Difficulty'].unique():
                diff_df = df[df['Difficulty'] == difficulty]
                if len(diff_df) > 0:
                    print(f"\n{str(difficulty).upper()}:")
                    for method in diff_df['Method'].unique():
                        method_df = diff_df[diff_df['Method'] == method]
                        avg_value = method_df['Value'].mean()
                        avg_time = method_df['Time(ms)'].mean()
                        print(f"  {method}: valeur={avg_value:.0f}, temps={avg_time:.0f}ms")
        
        # Recommandations
        print("\n💡 RECOMMANDATIONS:")
        print("-" * 80)
        
        # Meilleure méthode complète (équilibre temps/précision)
        complete_df = df[df['Method'].str.contains('Complete')]
        if len(complete_df) > 0:
            complete_df = complete_df.copy()
            complete_df['time_norm'] = (complete_df['Time(ms)'] - complete_df['Time(ms)'].min()) / \
                                      (complete_df['Time(ms)'].max() - complete_df['Time(ms)'].min())
            complete_df['score'] = 0.5 * (1 - complete_df['time_norm']) + \
                                  0.5 * complete_df['Optimal_Bool']
            
            best_complete = complete_df.groupby('Method')['score'].mean().idxmax()
            print(f"✅ Méthode complète recommandée: {best_complete}")
        
        # Meilleure méthode incomplète (rapidité)
        incomplete_df = df[df['Method'].str.contains('Incomplete')]
        if len(incomplete_df) > 0:
            incomplete_df = incomplete_df.copy()
            incomplete_df['time_norm'] = (incomplete_df['Time(ms)'] - incomplete_df['Time(ms)'].min()) / \
                                        (incomplete_df['Time(ms)'].max() - incomplete_df['Time(ms)'].min())
            incomplete_df['value_norm'] = (incomplete_df['Value'] - incomplete_df['Value'].min()) / \
                                         (incomplete_df['Value'].max() - incomplete_df['Value'].min())
            incomplete_df['score'] = 0.7 * (1 - incomplete_df['time_norm']) + \
                                   0.3 * incomplete_df['value_norm']
            
            best_incomplete = incomplete_df.groupby('Method')['score'].mean().idxmax()
            print(f"✅ Méthode incomplète recommandée: {best_incomplete}")
        
        print("\n" + "="*80)


def main():
    """Fonction principale"""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║         SOLVEUR DU PROBLÈME DU SAC À DOS (0-1)           ║
    ║        Méthodes complètes et incomplètes en Python       ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Configuration
    TIME_LIMIT_MS = 5 * 60 * 1000  # 5 minutes par instance
    KPLIB_PATH = "data/kplib"
    OUTPUT_FILE = "results/results.csv"
    
    # Création du runner
    runner = ExperimentRunner(time_limit_ms=TIME_LIMIT_MS)
    
    # Chargement des instances
    if not runner.load_instances(KPLIB_PATH):
        print("❌ Impossible de charger les instances. Vérifiez le chemin:", KPLIB_PATH)
        print("ℹ️  Vous pouvez cloner kplib avec: git clone https://github.com/jorlou/kplib.git data/kplib")
        return
    
    # Exécution des expériences
    runner.run_all_experiments()
    
    # Sauvegarde des résultats
    df = runner.save_results(OUTPUT_FILE)
    
    # Génération du rapport
    runner.generate_summary_report()
    
    print("\n✨ Expérimentation terminée avec succès!")
    print("📈 Vous pouvez analyser les résultats avec:")
    print(f"   - Le fichier CSV: {OUTPUT_FILE}")
    print("   - Le notebook Jupyter: notebooks/results_analysis.ipynb")


if __name__ == "__main__":
    main()