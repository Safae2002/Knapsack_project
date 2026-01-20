import os
import numpy as np
from typing import List, Tuple, Dict, Optional
import random

class KnapsackInstance:
    """Représente une instance du problème du sac à dos"""
    
    def __init__(self, name: str, n: int, capacity: int, 
                 weights: List[int], profits: List[int], 
                 difficulty: str = "unknown", optimal_value: int = -1):
        self.name = name
        self.n = n
        self.capacity = capacity
        self.weights = np.array(weights, dtype=int)
        self.profits = np.array(profits, dtype=int)
        self.difficulty = difficulty
        self.optimal_value = optimal_value
        
    def __repr__(self):
        return f"KnapsackInstance(name={self.name}, n={self.n}, capacity={self.capacity}, difficulty={self.difficulty})"


class InstanceLoader:
    """Charge les instances de benchmark depuis kplib"""
    
    @staticmethod
    def read_instance(filepath: str, difficulty: str = "unknown") -> Optional[KnapsackInstance]:
        """
        Lit une instance depuis un fichier au format kplib
        
        Format attendu:
        ligne 1: n (nombre d'items)
        ligne 2: capacité
        lignes 3 à n+2: profit poids
        """
        try:
            with open(filepath, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
            
            if len(lines) < 2:
                print(f"Fichier trop court: {filepath}")
                return None
            
            n = int(lines[0])
            capacity = int(lines[1])
            
            if len(lines) < 2 + n:
                print(f"Pas assez d'items dans {filepath}")
                return None
            
            profits = []
            weights = []
            
            for i in range(n):
                parts = lines[2 + i].split()
                if len(parts) < 2:
                    print(f"Ligne invalide: {lines[2 + i]}")
                    return None
                profits.append(int(parts[0]))
                weights.append(int(parts[1]))
            
            # Extraire le nom complet depuis le chemin (ex: 00Uncorrelated_n00050_R01000_s000)
            path_parts = filepath.replace('\\', '/').split('/')
            
            # Essayer de trouver la structure standard kplib
            # data/kplib/00Uncorrelated/n00050/R01000/s000.kp
            if len(path_parts) >= 4:
                file_name = path_parts[-1].replace('.kp', '')
                range_dir = path_parts[-2]
                size_dir = path_parts[-3]
                category_dir = path_parts[-4]
                
                name = f"{category_dir}_{size_dir}_{range_dir}_{file_name}"
            else:
                # Fallback sur le nom du fichier
                name = path_parts[-1].replace('.kp', '')
            
            return KnapsackInstance(name, n, capacity, weights, profits, difficulty)
            
        except Exception as e:
            print(f"Erreur lecture {filepath}: {e}")
            return None
    
    @staticmethod
    def generate_fallback_instances() -> List[KnapsackInstance]:
        """Génère des instances de secours si kplib n'est pas disponible"""
        instances = []
        random.seed(42)
        
        sizes_config = [
            {"min_n": 10, "max_n": 20, "max_weight": 50, "difficulty": "facile"},
            {"min_n": 40, "max_n": 60, "max_weight": 100, "difficulty": "moyen"},
            {"min_n": 100, "max_n": 150, "max_weight": 200, "difficulty": "difficile"}
        ]
        
        for config in sizes_config:
            for i in range(10):  # 10 instances par difficulté
                n = random.randint(config["min_n"], config["max_n"])
                weights = [random.randint(1, config["max_weight"]) for _ in range(n)]
                profits = [random.randint(1, config["max_weight"] * 2) for _ in range(n)]
                total_weight = sum(weights)
                capacity = total_weight // 2
                
                name = f"{config['difficulty']}_gen_{i}"
                instance = KnapsackInstance(
                    name=name,
                    n=n,
                    capacity=capacity,
                    weights=weights,
                    profits=profits,
                    difficulty=config["difficulty"]
                )
                instances.append(instance)
        
        return instances
    
    @staticmethod
    def load_benchmark_instances(kplib_path: str = "data/kplib") -> List[KnapsackInstance]:
        """Charge les instances depuis kplib"""
        instances = []
        
        # Configuration des instances par difficulté
        instances_config = {
            "facile": [
                # n=50
                "00Uncorrelated/n00050/R01000/s000.kp",
                "00Uncorrelated/n00050/R01000/s001.kp",
                "01WeaklyCorrelated/n00050/R01000/s000.kp",
                "02StronglyCorrelated/n00050/R01000/s000.kp",
                "03InverseStronglyCorrelated/n00050/R01000/s000.kp",
                "04AlmostStronglyCorrelated/n00050/R01000/s000.kp",
                "05SubsetSum/n00050/R01000/s000.kp",
                "06UncorrelatedWithSimilarWeights/n00050/R01000/s000.kp",
                "07SpannerUncorrelated/n00050/R01000/s000.kp",
                "08SpannerWeaklyCorrelated/n00050/R01000/s000.kp"
            ],
            "moyen": [
                # n=100
                "00Uncorrelated/n00100/R01000/s000.kp",
                "00Uncorrelated/n00100/R01000/s001.kp",
                "01WeaklyCorrelated/n00100/R01000/s000.kp",
                "01WeaklyCorrelated/n00100/R01000/s001.kp",
                "02StronglyCorrelated/n00100/R01000/s000.kp",
                "03InverseStronglyCorrelated/n00100/R01000/s000.kp",
                "04AlmostStronglyCorrelated/n00100/R01000/s000.kp",
                "05SubsetSum/n00100/R01000/s000.kp",
                "06UncorrelatedWithSimilarWeights/n00100/R01000/s000.kp",
                "07SpannerUncorrelated/n00100/R01000/s000.kp"
            ],
            "difficile": [
                # n=1000
                "00Uncorrelated/n01000/R01000/s000.kp",
                "00Uncorrelated/n01000/R01000/s001.kp",
                "01WeaklyCorrelated/n01000/R01000/s000.kp",
                "02StronglyCorrelated/n01000/R01000/s000.kp",
                "03InverseStronglyCorrelated/n01000/R01000/s000.kp",
                "04AlmostStronglyCorrelated/n01000/R01000/s000.kp",
                "05SubsetSum/n01000/R01000/s000.kp",
                "06UncorrelatedWithSimilarWeights/n01000/R01000/s000.kp",
                "07SpannerUncorrelated/n01000/R01000/s000.kp",
                "08SpannerWeaklyCorrelated/n01000/R01000/s000.kp"
            ]
        }
        
        loaded_count = 0
        for difficulty, file_list in instances_config.items():
            for file_rel_path in file_list:
                file_path = os.path.join(kplib_path, file_rel_path)
                
                if os.path.exists(file_path):
                    instance = InstanceLoader.read_instance(file_path, difficulty)
                    if instance:
                        instances.append(instance)
                        loaded_count += 1
                else:
                    print(f"Fichier introuvable: {file_path}")
        
        if loaded_count > 0:
            print(f"✓ {loaded_count} instances chargées depuis kplib")
        else:
            print("✗ Aucune instance trouvée dans kplib. Génération d'instances de secours...")
            instances = InstanceLoader.generate_fallback_instances()
        
        return instances