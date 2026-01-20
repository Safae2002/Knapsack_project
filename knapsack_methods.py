import time
import random
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from ortools.linear_solver import pywraplp
import math
from scipy.optimize import linprog
import pulp

class KnapsackResult:
    """Stocke les résultats d'une exécution"""
    
    def __init__(self, instance_name: str, method: str, best_value: int,
                 time_ms: float, nodes: int = -1, optimal: bool = False,
                 optimal_known: int = -1):
        self.instance_name = instance_name
        self.method = method
        self.best_value = best_value
        self.time_ms = time_ms
        self.nodes = nodes
        self.optimal = optimal
        self.optimal_known = optimal_known
        self.gap_percent = self._calculate_gap()
    
    def _calculate_gap(self) -> float:
        if self.optimal_known is not None and self.optimal_known > 0:
            return ((self.optimal_known - self.best_value) / self.optimal_known) * 100
        return -1.0
    
    def to_dict(self) -> Dict:
        return {
            'Instance': self.instance_name,
            'Method': self.method,
            'Value': self.best_value,
            'Time(ms)': self.time_ms,
            'Nodes': self.nodes,
            'Optimal': 'true' if self.optimal else 'false',
            'OptimalKnown': self.optimal_known,
            'Gap(%)': self.gap_percent
        }


class KnapsackSolver:
    """Implémente toutes les méthodes de résolution"""
    
    def __init__(self, time_limit_ms: int = 300000):  # 5 minutes par défaut
        self.time_limit_ms = time_limit_ms
    
    @staticmethod
    def _order_by_ratio(instance) -> List[int]:
        """Trie les indices par ratio profit/poids décroissant"""
        ratios = [(i, instance.profits[i] / instance.weights[i]) 
                 for i in range(instance.n)]
        ratios.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in ratios]
    
    # ===================== MÉTHODES COMPLÈTES =====================
    
    def solve_complete_mip_pulp(self, instance) -> KnapsackResult:
        """Résolution par programmation linéaire en nombres entiers avec PuLP"""
        start_time = time.time()
        
        # Création du modèle
        prob = pulp.LpProblem("Knapsack", pulp.LpMaximize)
        
        # Variables de décision
        x = pulp.LpVariable.dicts('x', range(instance.n), lowBound=0, upBound=1, cat='Binary')
        
        # Fonction objectif
        prob += pulp.lpSum([instance.profits[i] * x[i] for i in range(instance.n)])
        
        # Contrainte de capacité
        prob += pulp.lpSum([instance.weights[i] * x[i] for i in range(instance.n)]) <= instance.capacity
        
        # Configuration du solveur
        prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=self.time_limit_ms/1000))
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Récupération de la solution
        if pulp.LpStatus[prob.status] == 'Optimal':
            best_value = int(pulp.value(prob.objective))
            optimal = True
        else:
            # Essayer de récupérer une solution réalisable
            best_value = 0
            optimal = False
            for i in range(instance.n):
                if x[i].value() and x[i].value() > 0.5:
                    best_value += instance.profits[i]
        
        return KnapsackResult(
            instance_name=instance.name,
            method="Complete_MIP_PuLP",
            best_value=best_value,
            time_ms=elapsed_ms,
            nodes=-1,  # PuLP ne donne pas directement le nombre de nœuds
            optimal=optimal,
            optimal_known=instance.optimal_value
        )
    
    def solve_complete_mip_ortools(self, instance) -> KnapsackResult:
        """Résolution avec OR-Tools (CBC)"""
        start_time = time.time()
        
        # Création du solveur
        solver = pywraplp.Solver.CreateSolver('CBC')
        if not solver:
            return KnapsackResult(
                instance_name=instance.name,
                method="Complete_MIP_ORTools",
                best_value=0,
                time_ms=0,
                optimal=False,
                optimal_known=instance.optimal_value
            )
        
        # Variables
        x = {}
        for i in range(instance.n):
            x[i] = solver.IntVar(0, 1, f'x_{i}')
        
        # Contrainte de capacité
        constraint = solver.Constraint(0, float(instance.capacity))
        for i in range(instance.n):
            constraint.SetCoefficient(x[i], float(instance.weights[i]))
        
        # Fonction objectif
        objective = solver.Objective()
        for i in range(instance.n):
            objective.SetCoefficient(x[i], float(instance.profits[i]))
        objective.SetMaximization()
        
        # Configuration
        solver.SetTimeLimit(int(self.time_limit_ms))
        
        # Résolution
        status = solver.Solve()
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Interprétation des résultats
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            best_value = int(objective.Value() + 0.5)
            optimal = (status == pywraplp.Solver.OPTIMAL)
        else:
            best_value = 0
            optimal = False
        
        return KnapsackResult(
            instance_name=instance.name,
            method="Complete_MIP_ORTools",
            best_value=best_value,
            time_ms=elapsed_ms,
            nodes=solver.nodes(),
            optimal=optimal,
            optimal_known=instance.optimal_value
        )
    
    def solve_complete_branch_and_bound(self, instance) -> KnapsackResult:
        """Implémentation d'un Branch and Bound personnalisé"""
        start_time = time.time()
        
        # Tri des items par ratio décroissant
        items = list(range(instance.n))
        items.sort(key=lambda i: instance.profits[i] / instance.weights[i], reverse=True)
        
        # Fonction d'évaluation linéaire (relaxation continue)
        def linear_relaxation(current_weight, current_value, index):
            """Estimation optimiste par relaxation continue"""
            if index >= len(items):
                return current_value
            
            remaining_capacity = instance.capacity - current_weight
            estimated_value = current_value
            
            i = index
            while i < len(items) and remaining_capacity > 0:
                item_idx = items[i]
                if instance.weights[item_idx] <= remaining_capacity:
                    estimated_value += instance.profits[item_idx]
                    remaining_capacity -= instance.weights[item_idx]
                else:
                    # Prendre une fraction de l'item
                    fraction = remaining_capacity / instance.weights[item_idx]
                    estimated_value += instance.profits[item_idx] * fraction
                    break
                i += 1
            
            return estimated_value
        
        best_value = [0]
        nodes_explored = [0]
        
        def branch_and_bound(current_weight, current_value, index):
            nodes_explored[0] += 1
            
            # Condition d'arrêt
            if time.time() - start_time > self.time_limit_ms / 1000:
                return
            
            if index >= len(items):
                if current_value > best_value[0]:
                    best_value[0] = current_value
                return
            
            item_idx = items[index]
            
            # Estimation optimiste
            optimistic_bound = linear_relaxation(current_weight, current_value, index)
            
            # Coupe si la borne supérieure est inférieure à la meilleure solution
            if optimistic_bound <= best_value[0]:
                return
            
            # Branche "prendre l'item"
            if current_weight + instance.weights[item_idx] <= instance.capacity:
                branch_and_bound(
                    current_weight + instance.weights[item_idx],
                    current_value + instance.profits[item_idx],
                    index + 1
                )
            
            # Branche "ne pas prendre l'item"
            branch_and_bound(current_weight, current_value, index + 1)
        
        # Lancement de la recherche
        branch_and_bound(0, 0, 0)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Vérification si la solution est optimale (si on a exploré tout l'arbre)
        total_nodes = 2 ** instance.n
        optimal = (nodes_explored[0] >= total_nodes) or (best_value[0] == sum(instance.profits))
        
        return KnapsackResult(
            instance_name=instance.name,
            method="Complete_BranchAndBound",
            best_value=best_value[0],
            time_ms=elapsed_ms,
            nodes=nodes_explored[0],
            optimal=optimal,
            optimal_known=instance.optimal_value
        )
    
    def solve_complete_dp(self, instance) -> KnapsackResult:
        """Programmation dynamique classique (pseudo-polynomial)"""
        start_time = time.time()
        
        n = instance.n
        capacity = instance.capacity
        
        # Matrice DP
        dp = [[0] * (capacity + 1) for _ in range(n + 1)]
        
        # Remplissage de la matrice
        for i in range(1, n + 1):
            for w in range(capacity + 1):
                if instance.weights[i-1] <= w:
                    dp[i][w] = max(
                        dp[i-1][w],
                        dp[i-1][w - instance.weights[i-1]] + instance.profits[i-1]
                    )
                else:
                    dp[i][w] = dp[i-1][w]
        
        # Reconstruction de la solution
        best_value = dp[n][capacity]
        elapsed_ms = (time.time() - start_time) * 1000
        
        return KnapsackResult(
            instance_name=instance.name,
            method="Complete_DP",
            best_value=best_value,
            time_ms=elapsed_ms,
            nodes=n * capacity,  # Nombre d'états explorés
            optimal=True,  # DP donne toujours la solution optimale
            optimal_known=instance.optimal_value
        )
    
    # ===================== MÉTHODES INCOMPLÈTES =====================
    
    def solve_incomplete_greedy_simple(self, instance) -> KnapsackResult:
        """Algorithme glouton simple (ratio profit/poids)"""
        start_time = time.time()
        
        # Tri par ratio décroissant
        items = list(range(instance.n))
        items.sort(key=lambda i: instance.profits[i] / instance.weights[i], reverse=True)
        
        current_weight = 0
        current_value = 0
        
        for item_idx in items:
            if current_weight + instance.weights[item_idx] <= instance.capacity:
                current_weight += instance.weights[item_idx]
                current_value += instance.profits[item_idx]
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return KnapsackResult(
            instance_name=instance.name,
            method="Incomplete_Greedy_Simple",
            best_value=current_value,
            time_ms=elapsed_ms,
            nodes=0,
            optimal=False,
            optimal_known=instance.optimal_value
        )
    
    def solve_incomplete_greedy_random(self, instance, k: int = 3) -> KnapsackResult:
        """Glouton aléatoire: choisit parmi les k meilleurs à chaque étape"""
        start_time = time.time()
        random.seed(42)  # Pour reproductibilité
        
        items = list(range(instance.n))
        current_weight = 0
        current_value = 0
        
        while items:
            # Filtrer les items qui rentrent encore
            feasible_items = [i for i in items 
                            if current_weight + instance.weights[i] <= instance.capacity]
            
            if not feasible_items:
                break
            
            # Trier par ratio
            feasible_items.sort(key=lambda i: instance.profits[i] / instance.weights[i], reverse=True)
            
            # Prendre les k meilleurs
            top_k = feasible_items[:min(k, len(feasible_items))]
            
            # Choisir aléatoirement parmi les k meilleurs
            chosen = random.choice(top_k)
            
            current_weight += instance.weights[chosen]
            current_value += instance.profits[chosen]
            items.remove(chosen)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return KnapsackResult(
            instance_name=instance.name,
            method=f"Incomplete_Greedy_Random_k{k}",
            best_value=current_value,
            time_ms=elapsed_ms,
            nodes=0,
            optimal=False,
            optimal_known=instance.optimal_value
        )
    
    def solve_incomplete_greedy_probabilistic(self, instance, alpha: float = 0.9) -> KnapsackResult:
        """Glouton probabiliste: sélection selon une distribution de probabilité"""
        start_time = time.time()
        random.seed(42)
        
        items = list(range(instance.n))
        current_weight = 0
        current_value = 0
        
        while items:
            # Items réalisables
            feasible_items = [i for i in items 
                            if current_weight + instance.weights[i] <= instance.capacity]
            
            if not feasible_items:
                break
            
            # Calcul des scores (ratio^alpha)
            scores = []
            for i in feasible_items:
                ratio = instance.profits[i] / instance.weights[i]
                scores.append(ratio ** alpha)
            
            total_score = sum(scores)
            
            # Sélection probabiliste
            if total_score > 0:
                r = random.random()
                cumulative = 0
                chosen = feasible_items[0]  # Valeur par défaut
                
                for i, score in zip(feasible_items, scores):
                    cumulative += score / total_score
                    if r <= cumulative:
                        chosen = i
                        break
                
                current_weight += instance.weights[chosen]
                current_value += instance.profits[chosen]
                items.remove(chosen)
            else:
                break
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return KnapsackResult(
            instance_name=instance.name,
            method=f"Incomplete_Greedy_Probabilistic_alpha{alpha}",
            best_value=current_value,
            time_ms=elapsed_ms,
            nodes=0,
            optimal=False,
            optimal_known=instance.optimal_value
        )
    
    def solve_incomplete_simulated_annealing(self, instance, 
                                           initial_temp: float = 1000,
                                           cooling_rate: float = 0.95,
                                           iterations_per_temp: int = 100) -> KnapsackResult:
        """Recuit simulé pour le sac à dos"""
        start_time = time.time()
        random.seed(42)
        
        # Solution initiale (gloutonne)
        items = list(range(instance.n))
        items.sort(key=lambda i: instance.profits[i] / instance.weights[i], reverse=True)
        
        current_solution = [0] * instance.n
        current_weight = 0
        current_value = 0
        
        for item_idx in items:
            if current_weight + instance.weights[item_idx] <= instance.capacity:
                current_solution[item_idx] = 1
                current_weight += instance.weights[item_idx]
                current_value += instance.profits[item_idx]
        
        best_solution = current_solution[:]
        best_value = current_value
        
        temperature = initial_temp
        iteration = 0
        
        while temperature > 0.1 and (time.time() - start_time) * 1000 < self.time_limit_ms:
            for _ in range(iterations_per_temp):
                # Générer un voisin
                neighbor = current_solution[:]
                neighbor_weight = current_weight
                neighbor_value = current_value
                
                # Opération: ajouter, retirer, ou échanger
                operation = random.choice(['add', 'remove', 'swap'])
                
                if operation == 'add':
                    # Trouver un item non pris qui peut être ajouté
                    available = [i for i in range(instance.n) 
                                if neighbor[i] == 0 and 
                                current_weight + instance.weights[i] <= instance.capacity]
                    if available:
                        idx = random.choice(available)
                        neighbor[idx] = 1
                        neighbor_weight = current_weight + instance.weights[idx]
                        neighbor_value = current_value + instance.profits[idx]
                    else:
                        continue
                
                elif operation == 'remove':
                    # Trouver un item pris
                    taken = [i for i in range(instance.n) if neighbor[i] == 1]
                    if taken:
                        idx = random.choice(taken)
                        neighbor[idx] = 0
                        neighbor_weight = current_weight - instance.weights[idx]
                        neighbor_value = current_value - instance.profits[idx]
                    else:
                        continue
                
                else:  # swap
                    taken = [i for i in range(instance.n) if current_solution[i] == 1]
                    available = [i for i in range(instance.n) if current_solution[i] == 0]
                    
                    if taken and available:
                        remove_idx = random.choice(taken)
                        # Correctly check if adding add_idx after removing remove_idx fits
                        potential_available = [i for i in available 
                                             if current_weight - instance.weights[remove_idx] + instance.weights[i] <= instance.capacity]
                        
                        if potential_available:
                            add_idx = random.choice(potential_available)
                            
                            neighbor[remove_idx] = 0
                            neighbor[add_idx] = 1
                            
                            neighbor_weight = (current_weight - instance.weights[remove_idx] + 
                                             instance.weights[add_idx])
                            neighbor_value = (current_value - instance.profits[remove_idx] + 
                                            instance.profits[add_idx])
                        else:
                            continue
                    else:
                        continue
                
                # Accepter ou rejeter la nouvelle solution
                delta = neighbor_value - current_value
                
                if delta > 0 or random.random() < math.exp(delta / temperature):
                    current_solution = neighbor
                    current_value = neighbor_value
                    current_weight = neighbor_weight
                    
                    if current_value > best_value:
                        best_value = current_value
                        best_solution = current_solution[:]
                
                iteration += 1
            
            # Refroidissement
            temperature *= cooling_rate
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return KnapsackResult(
            instance_name=instance.name,
            method=f"Incomplete_SimulatedAnnealing",
            best_value=best_value,
            time_ms=elapsed_ms,
            nodes=iteration,
            optimal=False,
            optimal_known=instance.optimal_value
        )
    
    def solve_incomplete_genetic_algorithm(self, instance,
                                         population_size: int = 50,
                                         generations: int = 100,
                                         crossover_rate: float = 0.8,
                                         mutation_rate: float = 0.1) -> KnapsackResult:
        """Algorithme génétique pour le sac à dos"""
        start_time = time.time()
        random.seed(42)
        
        # Fonction d'évaluation d'un chromosome
        def evaluate(chromosome):
            total_weight = sum(instance.weights[i] for i, gene in enumerate(chromosome) if gene == 1)
            total_value = sum(instance.profits[i] for i, gene in enumerate(chromosome) if gene == 1)
            
            # Pénalité pour les solutions invalides
            if total_weight > instance.capacity:
                # Pénalité proportionnelle au dépassement
                penalty = (total_weight - instance.capacity) * 10
                return total_value - penalty
            return total_value
        
        # Génération initiale
        population = []
        for _ in range(population_size):
            chromosome = [random.randint(0, 1) for _ in range(instance.n)]
            # Réparation: si trop lourd, retirer des items aléatoirement
            while sum(instance.weights[i] for i, gene in enumerate(chromosome) if gene == 1) > instance.capacity:
                ones = [i for i, gene in enumerate(chromosome) if gene == 1]
                if ones:
                    chromosome[random.choice(ones)] = 0
                else:
                    break
            population.append(chromosome)
        
        best_solution = None
        best_value = 0
        
        generation = 0
        while generation < generations and (time.time() - start_time) * 1000 < self.time_limit_ms:
            # Évaluation
            fitness = [evaluate(chromosome) for chromosome in population]
            
            # Meilleure solution de cette génération
            current_best = max(fitness)
            if current_best > best_value:
                best_value = current_best
                best_solution = population[fitness.index(current_best)][:]
            
            # Sélection (tournoi)
            new_population = []
            for _ in range(population_size):
                # Tournoi de taille 2
                idx1, idx2 = random.sample(range(population_size), 2)
                if fitness[idx1] > fitness[idx2]:
                    new_population.append(population[idx1][:])
                else:
                    new_population.append(population[idx2][:])
            
            # Croisement
            for i in range(0, population_size - 1, 2):
                if random.random() < crossover_rate:
                    point = random.randint(1, instance.n - 1)
                    parent1 = new_population[i]
                    parent2 = new_population[i + 1]
                    
                    child1 = parent1[:point] + parent2[point:]
                    child2 = parent2[:point] + parent1[point:]
                    
                    new_population[i] = child1
                    new_population[i + 1] = child2
            
            # Mutation
            for i in range(population_size):
                if random.random() < mutation_rate:
                    mutation_point = random.randint(0, instance.n - 1)
                    new_population[i][mutation_point] = 1 - new_population[i][mutation_point]
            
            # Réparation des solutions invalides
            for chromosome in new_population:
                while sum(instance.weights[i] for i, gene in enumerate(chromosome) if gene == 1) > instance.capacity:
                    ones = [i for i, gene in enumerate(chromosome) if gene == 1]
                    if ones:
                        chromosome[random.choice(ones)] = 0
                    else:
                        break
            
            population = new_population
            generation += 1
        
        # Calcul final de la meilleure solution (sans pénalité)
        final_score = 0
        if best_solution:
            final_value = sum(instance.profits[i] for i, gene in enumerate(best_solution) if gene == 1)
            final_weight = sum(instance.weights[i] for i, gene in enumerate(best_solution) if gene == 1)
            
            if final_weight > instance.capacity:
                # Si invalide, prendre la meilleure solution valide dans la population finale
                valid_solutions = [chrom for chrom in population 
                                 if sum(instance.weights[i] for i, gene in enumerate(chrom) if gene == 1) <= instance.capacity]
                if valid_solutions:
                    final_score = max(sum(instance.profits[i] for i, gene in enumerate(chrom) if gene == 1) 
                                     for chrom in valid_solutions)
                else:
                    final_score = 0
            else:
                final_score = final_value
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return KnapsackResult(
            instance_name=instance.name,
            method="Incomplete_GeneticAlgorithm",
            best_value=final_score,
            time_ms=elapsed_ms,
            nodes=generation * population_size,
            optimal=False,
            optimal_known=instance.optimal_value
        )
    
    def solve_incomplete_tabu_search(self, instance, tabu_size: int = 10,
                                   max_iterations: int = 1000) -> KnapsackResult:
        """Recherche Tabu pour le sac à dos"""
        start_time = time.time()
        random.seed(42)
        
        # Solution initiale (gloutonne)
        current_solution = [0] * instance.n
        current_weight = 0
        current_value = 0
        
        items = list(range(instance.n))
        items.sort(key=lambda i: instance.profits[i] / instance.weights[i], reverse=True)
        
        for item_idx in items:
            if current_weight + instance.weights[item_idx] <= instance.capacity:
                current_solution[item_idx] = 1
                current_weight += instance.weights[item_idx]
                current_value += instance.profits[item_idx]
        
        best_solution = current_solution[:]
        best_value = current_value
        
        tabu_list = []
        iteration = 0
        
        while iteration < max_iterations and (time.time() - start_time) * 1000 < self.time_limit_ms:
            # Générer les voisins
            neighbors = []
            neighbor_values = []
            neighbor_moves = []
            
            # Voisins par flip (changer un bit)
            for i in range(instance.n):
                neighbor = current_solution[:]
                neighbor[i] = 1 - neighbor[i]  # Flip
                
                # Vérifier si le mouvement est tabou
                move = ('flip', i)
                if move in tabu_list:
                    continue
                
                # Calculer la valeur
                neighbor_weight = sum(instance.weights[j] for j, gene in enumerate(neighbor) if gene == 1)
                neighbor_value = sum(instance.profits[j] for j, gene in enumerate(neighbor) if gene == 1)
                
                # Accepter seulement si réalisable
                if neighbor_weight <= instance.capacity:
                    neighbors.append(neighbor)
                    neighbor_values.append(neighbor_value)
                    neighbor_moves.append(move)
            
            if not neighbors:
                break
            
            # Choisir le meilleur voisin
            best_neighbor_idx = max(range(len(neighbors)), key=lambda i: neighbor_values[i])
            
            # Mettre à jour la solution
            current_solution = neighbors[best_neighbor_idx]
            current_value = neighbor_values[best_neighbor_idx]
            
            # Mettre à jour la meilleure solution globale
            if current_value > best_value:
                best_value = current_value
                best_solution = current_solution[:]
            
            # Mettre à jour la liste Tabu
            tabu_list.append(neighbor_moves[best_neighbor_idx])
            if len(tabu_list) > tabu_size:
                tabu_list.pop(0)
            
            iteration += 1
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return KnapsackResult(
            instance_name=instance.name,
            method=f"Incomplete_TabuSearch",
            best_value=best_value,
            time_ms=elapsed_ms,
            nodes=iteration,
            optimal=False,
            optimal_known=instance.optimal_value
        )
    