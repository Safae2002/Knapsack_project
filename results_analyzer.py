# results_analysis.py
"""
Analyse complète des résultats du projet Knapsack
Reproduit exactement les analyses du fichier fourni
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple

# Configuration des graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

class KnapsackResultsAnalyzer:
    """Analyseur complet des résultats du sac à dos"""
    
    def __init__(self, results_file: str = "results/results.csv"):
        self.results_file = results_file
        self.data = None
        self.ground_truth_method = "Complete_MIP_ORTools"
        self.complete_methods = ["Complete_MIP_ORTools", "Complete_MIP_PuLP"]
        self.incomplete_methods = [
            "Incomplete_Greedy_Simple", 
            "Incomplete_Greedy_Random_k3",
            "Incomplete_Greedy_Probabilistic_alpha0.9",
            "Incomplete_SimulatedAnnealing",
            "Incomplete_TabuSearch"
        ]
    
    def load_and_prepare_data(self) -> bool:
        """Charge et prépare les données"""
        try:
            if not os.path.exists(self.results_file):
                print(f"❌ Fichier introuvable: {self.results_file}")
                print(f"   Recherche dans: {os.path.abspath(self.results_file)}")
                return False
            
            # Chargement des données
            self.data = pd.read_csv(self.results_file)
            print(f"✅ Données chargées: {len(self.data)} lignes")
            
            # Standardisation des noms de colonnes si nécessaire
            self._standardize_column_names()
            
            # Conversion des types de données
            self._convert_data_types()
            
            # Afficher les informations de base
            self._display_basic_info()
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            return False
    
    def _standardize_column_names(self):
        """Standardise les noms de colonnes"""
        column_mapping = {
            'time_ms': 'Time(ms)',
            'value': 'Value',
            'method': 'Method',
            'instance': 'Instance',
            'n': 'n',
            'capacity': 'Capacity',
            'optimal': 'Optimal',
            'gap_percent': 'Gap(%)'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in self.data.columns and new_name not in self.data.columns:
                self.data[new_name] = self.data[old_name]
    
    def _convert_data_types(self):
        """Convertit les types de données"""
        if 'Time(ms)' in self.data.columns:
            self.data['Time(ms)'] = pd.to_numeric(self.data['Time(ms)'], errors='coerce')
        
        if 'Value' in self.data.columns:
            self.data['Value'] = pd.to_numeric(self.data['Value'], errors='coerce')
        
        if 'Gap(%)' in self.data.columns:
            self.data['Gap(%)'] = pd.to_numeric(self.data['Gap(%)'], errors='coerce')
        
        if 'Optimal' in self.data.columns:
            # Convertir en booléen
            self.data['Optimal'] = self.data['Optimal'].astype(str).str.lower().isin(['true', '1', 'yes', 'oui'])
    
    def _display_basic_info(self):
        """Affiche les informations de base"""
        print(f"\n📊 Informations sur les données:")
        print(f"   Colonnes disponibles: {list(self.data.columns)}")
        print(f"   Méthodes présentes: {self.data['Method'].unique().tolist()}")
        print(f"   Nombre d'instances: {self.data['Instance'].nunique()}")
        print(f"   Tailles d'instances: {sorted(self.data['n'].unique().tolist())}")
    
    def add_timeout_column(self, timeout_ms: int = 300000):
        """Ajoute une colonne TimeOut (5 minutes par défaut)"""
        print(f"\n⏱️  Ajout de la colonne TimeOut (timeout = {timeout_ms}ms)...")
        
        self.data["TimeOut"] = self.data["Time(ms)"].astype(int) >= timeout_ms
        
        # Marquer comme timeout si méthode complète et non optimale
        complete_methods = [m for m in self.data['Method'].unique() if 'Complete' in m]
        for method in complete_methods:
            mask = (self.data["Optimal"] == False) & (self.data["Method"] == method)
            self.data.loc[mask, "TimeOut"] = True
        
        print(f"   Nombre de timeouts détectés: {self.data['TimeOut'].sum()}")
        
        # Afficher les 10 dernières lignes
        print("\n📋 10 dernières lignes avec TimeOut:")
        print(self.data[['Instance', 'Method', 'Time(ms)', 'Optimal', 'TimeOut']].tail(10))
    
    def calculate_optimality_gaps(self, threshold: float = 0.01):
        """
        Calcule les gaps d'optimalité par rapport à la méthode de référence.
        Si la référence a échoué ou n'est pas optimale, on compare avec la meilleure solution trouvée par n'importe quelle méthode.
        """
        print(f"\n📐 Calcul des gaps d'optimalité (référence: {self.ground_truth_method})...")
        
        # 1. Identifier la meilleure valeur trouvée pour chaque instance (toutes méthodes confondues)
        best_overall = self.data.groupby(['Instance', 'n', 'Capacity'])['Value'].max().reset_index()
        best_overall = best_overall.rename(columns={'Value': 'Best_Value_Found'})
        
        # 2. Identifier la valeur de la méthode de référence
        ref_data = self.data[self.data['Method'] == self.ground_truth_method][['Instance', 'n', 'Capacity', 'Value']]
        ref_data = ref_data.rename(columns={'Value': 'Value_Ref'})
        
        # 3. Fusionner
        self.data = pd.merge(self.data, best_overall, on=['Instance', 'n', 'Capacity'], how='left')
        self.data = pd.merge(self.data, ref_data, on=['Instance', 'n', 'Capacity'], how='left')
        
        # 4. Déterminer la valeur de base pour le calcul du gap
        # On utilise l'optimal connu si > 0, sinon Best_Value_Found
        # Dans nos données, OptimalKnown est souvent -1, donc Best_Value_Found est plus fiable
        def get_gap(row):
            # Priorité: 1. OptimalKnown (si valide > 0) 2. Best_Value_Found
            base_value = row['Best_Value_Found']
            if 'OptimalKnown' in row and row['OptimalKnown'] > 0:
                base_value = row['OptimalKnown']
                
            if base_value > 0:
                return (base_value - row['Value']) / base_value * 100
            return 0.0

        self.data['Gap(%)'] = self.data.apply(get_gap, axis=1)
        self.data['Gap_numeric'] = self.data['Gap(%)']
        
        # 5. Mettre à jour la colonne Optimal
        # Pour les méthodes complètes, on garde l'optimalité originale du solveur (si elle est à False, c'est un timeout)
        # Mais on peut l'améliorer : si le gap est > 0, c'est forcément pas optimal
        def is_optimal(row):
            gap_is_small = row['Gap(%)'] <= (threshold * 100)
            
            if 'Complete' in row['Method']:
                # Si c'est une méthode complète, elle doit AUSSI avoir son flag Optimal à True
                # (ce qui signifie que le solveur a fini la recherche)
                original_optimal = str(row['Optimal']).lower() in ['true', '1', 'yes']
                return original_optimal and gap_is_small
            else:
                # Pour les heuristiques, on se base uniquement sur le gap
                return gap_is_small

        self.data['Optimal'] = self.data.apply(is_optimal, axis=1)
        
        # Supprimer les colonnes temporaires
        self.data = self.data.drop(columns=['Best_Value_Found', 'Value_Ref'])
        
        # Afficher un résumé
        print("\n📊 Résumé des gaps calculés (par rapport à la meilleure solution trouvée):")
        summary = self.data.groupby('Method')['Gap(%)'].mean().sort_values()
        for method, gap in summary.items():
            print(f"   {method:40}: gap moyen = {gap:7.4f}%")
        
        print("\n📋 10 premières lignes avec les nouveaux gaps:")
        print(self.data[['Instance', 'Method', 'Value', 'Optimal', 'Gap(%)']].head(10))
    
    def plot_time_vs_n(self):
        """Graphique du temps en fonction de n"""
        print("\n📈 Graphique: Temps vs Nombre d'items...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Graphique 1: Échelle normale
        ax1 = axes[0]
        sns.lineplot(data=self.data, x='n', y='Time(ms)', hue='Method', 
                     marker='o', alpha=0.7, ax=ax1)
        ax1.set_xlabel('Number of Items (n)')
        ax1.set_ylabel('Time (ms)')
        ax1.set_title('Time vs Number of Items for Different Algorithms')
        ax1.legend(title='Algorithm')
        
        # Graphique 2: Échelle logarithmique
        ax2 = axes[1]
        sns.lineplot(data=self.data, x='n', y='Time(ms)', hue='Method', 
                     marker='o', alpha=0.7, ax=ax2)
        ax2.set_yscale('log')
        ax2.set_xlabel('Number of Items (n)')
        ax2.set_ylabel('Time (log scale)')
        ax2.set_title('Time vs Number of Items for Different Algorithms')
        ax2.legend(title='Algorithm')
        
        plt.tight_layout()
        plt.savefig('results/plots/time_vs_n.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("   ✅ Graphique sauvegardé: results/plots/time_vs_n.png")
    
    def plot_nodes_vs_n(self):
        """Graphique du nombre de nœuds explorés (si disponible)"""
        if 'Nodes' not in self.data.columns:
            print("⚠️  Colonne 'Nodes' non disponible - saut du graphique")
            return
        
        print("\n📈 Graphique: Nœuds explorés vs Nombre d'items...")
        
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=self.data, x='n', y='Nodes', hue='Method', 
                     marker='o', alpha=0.7)
        plt.yscale('log')
        plt.xlabel('Number of Items (n)')
        plt.ylabel('Number of Nodes Explored (log scale)')
        plt.title('Number of Nodes Explored vs Number of Items for Different Algorithms')
        plt.legend(title='Algorithm')
        
        plt.tight_layout()
        plt.savefig('results/plots/nodes_vs_n.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("   ✅ Graphique sauvegardé: results/plots/nodes_vs_n.png")
    
    def plot_time_vs_ratio(self):
        """Graphique du temps en fonction du ratio n/capacité"""
        print("\n📈 Graphique: Temps vs Ratio n/Capacité...")
        
        # Calculer les ratios
        self.data['ratio_n_capacity'] = self.data['n'] / self.data['Capacity']
        self.data['ratio_capacity_n'] = self.data['Capacity'] / self.data['n']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Filtrer les méthodes complètes
        complete_methods = [m for m in self.data['Method'].unique() if 'Complete' in m]
        complete_data = self.data[self.data['Method'].isin(complete_methods)]
        
        # Graphique 1: Ratio n/Capacité
        ax1 = axes[0]
        sns.lineplot(data=complete_data, x='ratio_n_capacity', y='Time(ms)', 
                     hue='Method', marker='o', alpha=0.7, ax=ax1)
        ax1.set_yscale('log')
        ax1.set_xlabel('Ratio Number of Items/Capacity')
        ax1.set_ylabel('Time (log scale)')
        ax1.set_title('Time vs Ratio Number of Items/Capacity for Different Algorithms')
        ax1.legend(title='Algorithm')
        
        # Graphique 2: Ratio Capacité/n
        ax2 = axes[1]
        sns.lineplot(data=complete_data, x='ratio_capacity_n', y='Time(ms)', 
                     hue='Method', marker='o', alpha=0.7, ax=ax2)
        ax2.set_yscale('log')
        ax2.set_xlabel('Ratio Capacity/Number of Items')
        ax2.set_ylabel('Time (log scale)')
        ax2.set_title('Time vs Ratio Capacity/Number of Items for Different Algorithms')
        ax2.legend(title='Algorithm')
        
        plt.tight_layout()
        plt.savefig('results/plots/time_vs_ratio.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("   ✅ Graphique sauvegardé: results/plots/time_vs_ratio.png")
    
    def plot_optimality_vs_n(self):
        """Graphique de l'optimalité en fonction de n"""
        print("\n📈 Graphique: Optimalité vs Nombre d'items...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Graphique 1: Optimalité binaire
        ax1 = axes[0]
        self.data['Optimal_numeric'] = self.data['Optimal'].astype(int)
        sns.lineplot(data=self.data, x='n', y='Optimal_numeric', 
                     hue='Method', marker='o', alpha=0.7, ax=ax1)
        ax1.set_ylim(-0.1, 1.1)
        ax1.set_xlabel('Number of Items (n)')
        ax1.set_ylabel('Optimality (1 = Optimal, 0 = Not Optimal)')
        ax1.set_title('Optimality vs Number of Items for All Algorithms')
        ax1.legend(title='Algorithm', fancybox=True, framealpha=0.5)
        
        # Graphique 2: Gap d'optimalité
        ax2 = axes[1]
        # Convertir Gap(%) en numérique pour le tracé
        self.data['Gap_numeric'] = pd.to_numeric(self.data['Gap(%)'], errors='coerce')
        sns.lineplot(data=self.data, x='n', y='Gap_numeric', 
                     hue='Method', marker='o', alpha=0.7, ax=ax2)
        
        # Définir les limites de l'axe Y
        if not self.data['Gap_numeric'].isna().all():
            y_min = max(self.data['Gap_numeric'].min() - 1, 0)
            y_max = min(self.data['Gap_numeric'].max() + 1, 100)
            ax2.set_ylim(y_min, y_max)
        
        ax2.set_xlabel('Number of Items (n)')
        ax2.set_ylabel('Optimality Gap (%)')
        ax2.set_title('Optimality Gap vs Number of Items for All Algorithms')
        ax2.legend(title='Algorithm', fancybox=True, framealpha=0.5)
        
        plt.tight_layout()
        plt.savefig('results/plots/optimality_vs_n.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("   ✅ Graphique sauvegardé: results/plots/optimality_vs_n.png")
    
    def plot_optimality_vs_ratio(self):
        """Graphique de l'optimalité en fonction du ratio n/capacité"""
        print("\n📈 Graphique: Optimalité vs Ratio n/Capacité...")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Graphique 1: Optimalité binaire
        ax1 = axes[0]
        sns.lineplot(data=self.data, x='ratio_n_capacity', y='Optimal_numeric', 
                     hue='Method', marker='o', alpha=0.7, ax=ax1)
        ax1.set_ylim(-0.1, 1.1)
        ax1.set_xlabel('Ratio Number of Items/Capacity')
        ax1.set_ylabel('Optimality (1 = Optimal, 0 = Not Optimal)')
        ax1.set_title('Optimality vs Ratio Number of Items/Capacity for All Algorithms')
        ax1.legend(title='Algorithm', fancybox=True, framealpha=0.5)
        
        # Graphique 2: Gap d'optimalité
        ax2 = axes[1]
        sns.lineplot(data=self.data, x='ratio_n_capacity', y='Gap_numeric', 
                     hue='Method', marker='o', alpha=0.7, ax=ax2)
        ax2.set_xlabel('Ratio Number of Items/Capacity')
        ax2.set_ylabel('Optimality Gap (%)')
        ax2.set_title('Optimality Gap vs Ratio Number of Items/Capacity for All Algorithms')
        ax2.legend(title='Algorithm', fancybox=True, framealpha=0.5)
        
        plt.tight_layout()
        plt.savefig('results/plots/optimality_vs_ratio.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("   ✅ Graphique sauvegardé: results/plots/optimality_vs_ratio.png")
    
    def generate_comparative_table(self):
        """Génère un tableau comparatif détaillé"""
        print("\n📊 Tableau comparatif des méthodes:")
        print("=" * 120)
        
        comparison_data = []
        methods = sorted(self.data['Method'].unique())
        
        for method in methods:
            method_data = self.data[self.data['Method'] == method]
            
            avg_value = method_data['Value'].mean()
            std_value = method_data['Value'].std()
            avg_time = method_data['Time(ms)'].mean()
            std_time = method_data['Time(ms)'].std()
            optimal_rate = method_data['Optimal'].mean() * 100
            
            # Calculer le gap moyen (en numérique)
            if 'Gap_numeric' in method_data.columns:
                avg_gap = method_data['Gap_numeric'].mean()
            else:
                avg_gap = np.nan
            
            method_type = 'Complète' if 'Complete' in method else 'Incomplète'
            
            comparison_data.append({
                'Méthode': method,
                'Type': method_type,
                'Valeur moyenne': f"{avg_value:.0f} ± {std_value:.0f}",
                'Temps moyen (ms)': f"{avg_time:.0f} ± {std_time:.0f}",
                'Taux optimal (%)': f"{optimal_rate:.1f}%",
                'Gap moyen (%)': f"{avg_gap:.2f}%" if not np.isnan(avg_gap) else "N/A"
            })
        
        # Créer et afficher le DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        print("=" * 120)
        
        # Sauvegarder en CSV
        comparison_df.to_csv('results/comparison_table.csv', index=False)
        print("💾 Tableau comparatif sauvegardé: results/comparison_table.csv")
        
        return comparison_df
    
    def generate_statistical_summary(self):
        """Génère un résumé statistique"""
        print("\n📈 Résumé statistique:")
        print("-" * 80)
        
        # Statistiques par méthode
        for method in sorted(self.data['Method'].unique()):
            method_data = self.data[self.data['Method'] == method]
            
            print(f"\n🔍 {method}:")
            print(f"   Nombre d'exécutions: {len(method_data)}")
            print(f"   Valeur moyenne: {method_data['Value'].mean():.0f}")
            print(f"   Valeur médiane: {method_data['Value'].median():.0f}")
            print(f"   Écart-type: {method_data['Value'].std():.0f}")
            print(f"   Temps moyen: {method_data['Time(ms)'].mean():.0f} ms")
            print(f"   Temps médian: {method_data['Time(ms)'].median():.0f} ms")
            print(f"   Taux d'optimalité: {method_data['Optimal'].mean()*100:.1f}%")
            
            if 'Gap_numeric' in method_data.columns:
                valid_gaps = method_data['Gap_numeric'].dropna()
                if len(valid_gaps) > 0:
                    print(f"   Gap moyen: {valid_gaps.mean():.2f}%")
                    print(f"   Gap médian: {valid_gaps.median():.2f}%")
        
        # Statistiques globales
        print(f"\n🌐 Statistiques globales:")
        print(f"   Nombre total d'exécutions: {len(self.data)}")
        print(f"   Nombre d'instances uniques: {self.data['Instance'].nunique()}")
        print(f"   Nombre de méthodes testées: {self.data['Method'].nunique()}")
        print(f"   Taille moyenne des instances: {self.data['n'].mean():.1f}")
        print(f"   Capacité moyenne: {self.data['Capacity'].mean():.0f}")
        
        print("-" * 80)
    
    def save_processed_data(self, output_file: str = "results/processed_results.csv"):
        """Sauvegarde les données traitées"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        self.data.to_csv(output_file, index=False)
        print(f"💾 Données traitées sauvegardées: {output_file}")
    
    def run_full_analysis(self):
        """Exécute l'analyse complète"""
        print("""
        ╔══════════════════════════════════════════════════════════╗
        ║          ANALYSE COMPLÈTE DES RÉSULTATS KNAPSACK        ║
        ╚══════════════════════════════════════════════════════════╝
        """)
        
        # Créer le dossier plots s'il n'existe pas
        os.makedirs('results/plots', exist_ok=True)
        
        # 1. Charger les données
        if not self.load_and_prepare_data():
            return
        
        # 2. Ajouter la colonne TimeOut
        self.add_timeout_column(timeout_ms=300000)  # 5 minutes
        
        # 3. Calculer les gaps d'optimalité
        self.calculate_optimality_gaps(threshold=0.01)  # 1%
        
        # 4. Générer les graphiques
        self.plot_time_vs_n()
        self.plot_nodes_vs_n()
        self.plot_time_vs_ratio()
        self.plot_optimality_vs_n()
        self.plot_optimality_vs_ratio()
        
        # 5. Générer les tableaux et résumés
        self.generate_comparative_table()
        self.generate_statistical_summary()
        
        # 6. Sauvegarder les données traitées
        self.save_processed_data()
        
        print("\n✨ Analyse terminée avec succès!")
        print("📁 Résultats disponibles dans le dossier 'results/'")
        print("📊 Visualisations disponibles dans 'results/plots/'")
        print("📄 Tableaux disponibles dans 'results/comparison_table.csv'")


def main():
    """Fonction principale"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyse complète des résultats du solveur de sac à dos"
    )
    
    parser.add_argument(
        "--results",
        default="results/results.csv",
        help="Fichier de résultats CSV (défaut: results/results.csv)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Dossier de sortie (défaut: results)"
    )
    
    args = parser.parse_args()
    
    # Créer le dossier de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Exécuter l'analyse
    analyzer = KnapsackResultsAnalyzer(results_file=args.results)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()