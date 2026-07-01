import random
import math
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
from matplotlib.patches import Rectangle
from collections import Counter

class CuttingStockSolver:
    def __init__(self, stock_length, piece_lengths, num_agents, method):
        self.stock_length = stock_length
        self.piece_lengths = sorted(piece_lengths, reverse=True)
        self.num_agents = num_agents
        self.method = method
        self.best_solution = None
        self.best_waste = float('inf')
        self.best_cuts_count = float('inf')
        self.all_agent_solutions = []
        self.execution_times = {}
        self.color_map = {}

    def generate_random_solution(self):
        """Generate a valid random cutting pattern with no negative waste"""
        remaining = self.stock_length
        pattern = []

        while remaining >= min(self.piece_lengths):
            possible_pieces = [p for p in self.piece_lengths if p <= remaining]
            if not possible_pieces:
                break
            piece = random.choice(possible_pieces)
            pattern.append(piece)
            remaining -= piece

        waste = max(0, remaining)
        return pattern, waste

    def evaluate_solution(self, pattern):
        """Calculate waste and number of cuts for a pattern"""
        total_length = sum(pattern)
        waste = max(0, self.stock_length - total_length)
        cuts_count = max(0, len(pattern) - 1)
        return waste, cuts_count

    def hill_climbing(self, max_iterations=1000):
        """Hill Climbing optimization with valid solutions"""
        start_time = time.time()
        current_pattern, current_waste = self.generate_random_solution()
        current_cuts = len(current_pattern) - 1

        for _ in range(max_iterations):
            neighbor = current_pattern.copy()
            operation = random.random()

            if operation < 0.3 and len(neighbor) > 1:
                idx = random.randint(0, len(neighbor)-1)
                neighbor.pop(idx)
            elif operation < 0.6:
                remaining = self.stock_length - sum(neighbor)
                possible_pieces = [p for p in self.piece_lengths if p <= remaining]
                if possible_pieces:
                    neighbor.append(random.choice(possible_pieces))
            else:
                if len(neighbor) > 0:
                    idx = random.randint(0, len(neighbor)-1)
                    old_piece = neighbor[idx]
                    remaining = self.stock_length - (sum(neighbor) - old_piece)
                    possible_pieces = [p for p in self.piece_lengths
                                     if p <= remaining and p != old_piece]
                    if possible_pieces:
                        neighbor[idx] = random.choice(possible_pieces)

            neighbor_waste, neighbor_cuts = self.evaluate_solution(neighbor)

            if (neighbor_waste < current_waste) or \
               (neighbor_waste == current_waste and neighbor_cuts < current_cuts):
                current_pattern, current_waste, current_cuts = neighbor, neighbor_waste, neighbor_cuts

        execution_time = time.time() - start_time
        return current_pattern, current_waste, current_cuts, execution_time

    def simulated_annealing(self, max_iterations=1000, initial_temp=100, cooling_rate=0.99):
        """Simulated Annealing optimization with valid solutions"""
        start_time = time.time()
        current_pattern, current_waste = self.generate_random_solution()
        current_cuts = len(current_pattern) - 1
        temp = initial_temp

        for i in range(max_iterations):
            neighbor = current_pattern.copy()
            operation = random.random()

            if operation < 0.3 and len(neighbor) > 1:
                idx = random.randint(0, len(neighbor)-1)
                neighbor.pop(idx)
            elif operation < 0.6:
                remaining = self.stock_length - sum(neighbor)
                possible_pieces = [p for p in self.piece_lengths if p <= remaining]
                if possible_pieces:
                    neighbor.append(random.choice(possible_pieces))
            else:
                if len(neighbor) > 0:
                    idx = random.randint(0, len(neighbor)-1)
                    old_piece = neighbor[idx]
                    remaining = self.stock_length - (sum(neighbor) - old_piece)
                    possible_pieces = [p for p in self.piece_lengths
                                     if p <= remaining and p != old_piece]
                    if possible_pieces:
                        neighbor[idx] = random.choice(possible_pieces)

            neighbor_waste, neighbor_cuts = self.evaluate_solution(neighbor)

            if neighbor_waste < current_waste:
                accept_prob = 1.0
            else:
                energy_diff = (neighbor_waste - current_waste) + 0.1 * (neighbor_cuts - current_cuts)
                accept_prob = math.exp(-energy_diff / temp)

            if random.random() < accept_prob:
                current_pattern, current_waste, current_cuts = neighbor, neighbor_waste, neighbor_cuts

            temp *= cooling_rate

        execution_time = time.time() - start_time
        return current_pattern, current_waste, current_cuts, execution_time

    def genetic_algorithm(self, population_size=50, generations=100, mutation_rate=0.1):
        """Genetic Algorithm optimization with valid solutions"""
        start_time = time.time()

        def create_individual():
            pattern, waste = self.generate_random_solution()
            return pattern

        def fitness(pattern):
            total = sum(pattern)
            waste = max(0, self.stock_length - total)
            cuts = max(0, len(pattern) - 1)
            return -waste - 0.01 * cuts

        population = [create_individual() for _ in range(population_size)]

        for _ in range(generations):
            fitness_scores = [fitness(ind) for ind in population]

            new_population = []
            for _ in range(population_size):
                a, b = random.sample(range(population_size), 2)
                winner = population[a] if fitness_scores[a] > fitness_scores[b] else population[b]
                new_population.append(winner.copy())

            for i in range(0, population_size-1, 2):
                parent1, parent2 = new_population[i], new_population[i+1]
                if random.random() < 0.7:
                    min_len = min(len(parent1), len(parent2))
                    if min_len > 1:
                        crossover_point = random.randint(1, min_len-1)
                        parent1[crossover_point:], parent2[crossover_point:] = \
                            parent2[crossover_point:], parent1[crossover_point:]

            for i in range(population_size):
                if random.random() < mutation_rate:
                    if random.random() < 0.33 and len(new_population[i]) > 1:
                        idx = random.randint(0, len(new_population[i])-1)
                        new_population[i].pop(idx)
                    elif random.random() < 0.66:
                        remaining = self.stock_length - sum(new_population[i])
                        possible_pieces = [p for p in self.piece_lengths if p <= remaining]
                        if possible_pieces:
                            new_population[i].append(random.choice(possible_pieces))
                    else:
                        if len(new_population[i]) > 0:
                            idx = random.randint(0, len(new_population[i])-1)
                            old_piece = new_population[i][idx]
                            remaining = self.stock_length - (sum(new_population[i]) - old_piece)
                            possible_pieces = [p for p in self.piece_lengths
                                             if p <= remaining and p != old_piece]
                            if possible_pieces:
                                new_population[i][idx] = random.choice(possible_pieces)

            population = new_population

        best_idx = np.argmax([fitness(ind) for ind in population])
        best_pattern = population[best_idx]
        best_waste = max(0, self.stock_length - sum(best_pattern))
        best_cuts = max(0, len(best_pattern) - 1)
        execution_time = time.time() - start_time
        return best_pattern, best_waste, best_cuts, execution_time

    def collaborative_optimization(self, algorithm, max_iterations=1000):
        """Collaborative Agent-Based Optimization with same algorithm"""
        agents_solutions = []
        total_time = 0

        for agent_id in range(self.num_agents):
            if algorithm == "HC":
                pattern, waste, cuts, exec_time = self.hill_climbing(max_iterations)
            elif algorithm == "SA":
                pattern, waste, cuts, exec_time = self.simulated_annealing(max_iterations)
            elif algorithm == "GA":
                pattern, waste, cuts, exec_time = self.genetic_algorithm()

            agents_solutions.append({
                "agent": agent_id+1,
                "algorithm": algorithm,
                "pattern": pattern,
                "waste": waste,
                "cuts": cuts,
                "time": exec_time
            })
            total_time += exec_time

        best_solution = min(agents_solutions, key=lambda x: (x["waste"], x["cuts"]))
        self.all_agent_solutions = agents_solutions
        self.execution_times["Collaborative"] = total_time / self.num_agents
        return best_solution["pattern"], best_solution["waste"], best_solution["cuts"]

    def hyper_metaheuristic_approach(self, max_iterations=1000):
        """Hyper Meta-Heuristic Approach with different algorithms per agent"""
        agents_solutions = []
        algorithms = ["HC", "SA", "GA"]
        total_time = 0

        for agent_id in range(self.num_agents):
            algo = algorithms[agent_id % len(algorithms)]
            if algo == "HC":
                pattern, waste, cuts, exec_time = self.hill_climbing(max_iterations)
            elif algo == "SA":
                pattern, waste, cuts, exec_time = self.simulated_annealing(max_iterations)
            elif algo == "GA":
                pattern, waste, cuts, exec_time = self.genetic_algorithm()

            agents_solutions.append({
                "agent": agent_id+1,
                "algorithm": algo,
                "pattern": pattern,
                "waste": waste,
                "cuts": cuts,
                "time": exec_time
            })
            total_time += exec_time

        best_solution = min(agents_solutions, key=lambda x: (x["waste"], x["cuts"]))
        self.all_agent_solutions = agents_solutions
        self.execution_times["Hyper"] = total_time / self.num_agents
        return best_solution["pattern"], best_solution["waste"], best_solution["cuts"]

    def solve(self):
        """Main solving method"""
        if self.method == "single":
            print("Please select a specific algorithm (HC, SA, or GA) for single method.")
            return

        start_time = time.time()

        if self.method in ["HC", "SA", "GA"]:
            if self.num_agents > 1:
                best_pattern, best_waste, best_cuts = self.collaborative_optimization(self.method)
            else:
                if self.method == "HC":
                    best_pattern, best_waste, best_cuts, exec_time = self.hill_climbing()
                    self.execution_times["Hill Climbing"] = exec_time
                elif self.method == "SA":
                    best_pattern, best_waste, best_cuts, exec_time = self.simulated_annealing()
                    self.execution_times["Simulated Annealing"] = exec_time
                elif self.method == "GA":
                    best_pattern, best_waste, best_cuts, exec_time = self.genetic_algorithm()
                    self.execution_times["Genetic Algorithm"] = exec_time
        elif self.method == "hyper":
            best_pattern, best_waste, best_cuts = self.hyper_metaheuristic_approach()

        total_execution_time = time.time() - start_time
        self.execution_times["Total"] = total_execution_time

        best_waste = max(0, best_waste)
        best_cuts = max(0, best_cuts)

        self.best_solution = best_pattern
        self.best_waste = best_waste
        self.best_cuts_count = best_cuts
        return best_pattern, best_waste, best_cuts

    def visualize_solution(self):
        """Create a visualization of the cutting pattern"""
        if not self.best_solution:
            print("No solution to visualize. Run solve() first.")
            return

        fig, (ax, legend_ax) = plt.subplots(2, 1, figsize=(12, 6),
                                          gridspec_kw={'height_ratios': [2, 1]})

        ax.add_patch(Rectangle((0, 0), self.stock_length, 1,
                    facecolor='lightgray', edgecolor='black', alpha=0.5))

        x_pos = 0
        unique_pieces = sorted(list(set(self.best_solution)), reverse=True)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_pieces)))  # pylint: disable=no-member
        self.color_map = {piece: color for piece, color in zip(unique_pieces, colors)}

        for piece in self.best_solution:
            color = self.color_map[piece]
            ax.add_patch(Rectangle((x_pos, 0), piece, 1,
                                facecolor=color, edgecolor='black', linewidth=1))
            x_pos += piece

        if self.best_waste > 0:
            ax.add_patch(Rectangle((x_pos, 0), self.best_waste, 1,
                                 facecolor='white', edgecolor='black',
                                 hatch='//', alpha=0.3))

        legend_patches = []
        for piece, color in self.color_map.items():
            legend_patches.append(Rectangle((0, 0), 1, 1, facecolor=color, label=f"{piece:.2f}m"))

        if self.best_waste > 0:
            legend_patches.append(Rectangle((0, 0), 1, 1,
                                 facecolor='white', edgecolor='black',
                                 hatch='//', alpha=0.3, label=f"Waste ({self.best_waste:.2f}m)"))

        legend_ax.legend(handles=legend_patches, loc='center', ncol=len(legend_patches),
                        frameon=False)
        legend_ax.axis('off')

        ax.set_xlim(0, self.stock_length)
        ax.set_ylim(0, 1.2)
        ax.set_yticks([])
        ax.set_title(f"Optimal Cutting Pattern\nTotal Waste: {self.best_waste:.2f}m | Cuts: {self.best_cuts_count}",
                    fontsize=12, pad=20)
        ax.set_xlabel("Length (meters)", fontsize=10)

        plt.tight_layout()
        plt.show()

    def plot_method_comparison(self):
        methods = []
        wastes = []

        if self.num_agents > 1:
            if self.method == "hyper":
                hc_waste = [s['waste'] for s in self.all_agent_solutions if s['algorithm'] == "HC"]
                sa_waste = [s['waste'] for s in self.all_agent_solutions if s['algorithm'] == "SA"]
                ga_waste = [s['waste'] for s in self.all_agent_solutions if s['algorithm'] == "GA"]

                methods = ['HC (Hyper)', 'SA (Hyper)', 'GA (Hyper)']
                wastes = [min(hc_waste), min(sa_waste), min(ga_waste)]
            else:
                methods = [f'{self.method} (Agent {s["agent"]})' for s in self.all_agent_solutions]
                wastes = [s['waste'] for s in self.all_agent_solutions]
        else:
            methods = [self.method]
            wastes = [self.best_waste]

        plt.figure(figsize=(10, 5))
        bars = plt.bar(methods, wastes, color=['#4C72B0', '#55A868', '#C44E52'])

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}m',
                    ha='center', va='bottom')

        plt.title('Method Waste Comparison')
        plt.ylabel('Waste (meters)')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def plot_efficiency_scatter(self):
        if not self.all_agent_solutions:
            return

        plt.figure(figsize=(8, 6))
        color_map = {'HC': '#4C72B0', 'SA': '#55A868', 'GA': '#C44E52'}

        for solution in self.all_agent_solutions:
            plt.scatter(solution['time'], solution['waste'],
                       c=color_map[solution['algorithm']],
                       s=100, alpha=0.7,
                       label=f"{solution['algorithm']} (Agent {solution['agent']})")

        plt.title('Time-Efficiency Comparison')
        plt.xlabel('Execution Time (seconds)')
        plt.ylabel('Waste (meters)')
        plt.grid(True, linestyle='--', alpha=0.5)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        plt.tight_layout()
        plt.show()

    def plot_algorithm_distribution(self):
        if not self.all_agent_solutions or len(self.all_agent_solutions) < 3:
            return

        data = pd.DataFrame(self.all_agent_solutions)

        plt.figure(figsize=(10, 6))
        boxprops = dict(linestyle='-', linewidth=2, color='darkgoldenrod')
        medianprops = dict(linestyle='-', linewidth=2, color='firebrick')

        data.boxplot(column='waste', by='algorithm', grid=False,
                    boxprops=boxprops, medianprops=medianprops,
                    patch_artist=True,
                    flierprops=dict(marker='o', markersize=5))

        plt.title('Algorithm Waste Distribution')
        plt.suptitle('')
        plt.xlabel('Algorithm')
        plt.ylabel('Waste (meters)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        means = data.groupby('algorithm')['waste'].mean()
        for i, mean in enumerate(means):
            plt.text(i+1, mean+0.05, f'Avg: {mean:.2f}m',
                    ha='center', color='blue', weight='bold')

        plt.tight_layout()
        plt.show()

def get_user_input():
    """Get input parameters from user"""
    print("=== Cutting Stock Problem Solver ===")

    while True:
        try:
            stock_length = float(input("Enter the total length of the stock roll (meters) Example: 123.45: "))
            if stock_length <= 0:
                print("Please enter a positive number.")
            else:
                break
        except ValueError:
            print("Please enter a valid number.")

    piece_lengths = []
    print("\nEnter the lengths of pieces to be cut Example: 12.34 (enter 0 when done):")
    while True:
        try:
            piece = float(input(f"Piece {len(piece_lengths)+1} length (meters): "))
            if piece == 0:
                if len(piece_lengths) == 0:
                    print("Please enter at least one piece length.")
                else:
                    break
            elif piece <= 0:
                print("Please enter a positive number.")
            elif piece > stock_length:
                print("Piece length cannot exceed stock length.")
            else:
                piece_lengths.append(piece)
        except ValueError:
            print("Please enter a valid number.")

    while True:
        try:
            num_agents = int(input("\nEnter the number of agents to use (1-10): "))
            if 1 <= num_agents <= 10:
                break
            else:
                print("Please enter a number between 1 and 10.")
        except ValueError:
            print("Please enter a valid integer.")

    print("\nSelect optimization method:")
    print("1. Single Algorithm (choose one)")
    print("2. Collaborative Agent-Based Optimization (same algorithm)")
    print("3. Hyper Meta-Heuristic Approach (different algorithms)")

    while True:
        choice = input("Enter your choice (1-3): ")
        if choice == '1':
            print("\nSelect algorithm:")
            print("HC - Hill Climbing")
            print("SA - Simulated Annealing")
            print("GA - Genetic Algorithm")
            method = input("Enter algorithm code (HC/SA/GA): ").upper()
            if method in ["HC", "SA", "GA"]:
                break
            else:
                print("Invalid algorithm choice.")
        elif choice == '2':
            print("\nSelect algorithm for collaborative agents:")
            print("HC - Hill Climbing")
            print("SA - Simulated Annealing")
            print("GA - Genetic Algorithm")
            method = input("Enter algorithm code (HC/SA/GA): ").upper()
            if method in ["HC", "SA", "GA"]:
                break
            else:
                print("Invalid algorithm choice.")
        elif choice == '3':
            method = "hyper"
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

    return stock_length, piece_lengths, num_agents, method

def main():
    # Kullanıcı girişlerini al
    stock_length, piece_lengths, num_agents, method = get_user_input()

    # Problemi çöz
    print("\nSolving the Cutting Stock Problem...")
    solver = CuttingStockSolver(stock_length, piece_lengths, num_agents, method)
    best_pattern, best_waste, best_cuts = solver.solve()

    # Sonuçları göster
    print("\n=== Optimal Solution ===")
    print(f"Cutting Pattern: {[round(p, 2) for p in best_pattern]}")
    print(f"Total Waste: {best_waste:.2f} meters")
    print(f"Number of Cuts: {best_cuts}")
    print(f"Stock Utilization: {(sum(best_pattern)/stock_length)*100:.2f}%")

    # Tüm ajan sonuçlarını göster (çoklu ajan durumunda)
    if num_agents > 1:
        print("\n=== All Agents' Solutions ===")
        best_agent = None
        best_waste = float('inf')
        best_cuts = float('inf')

        for solution in solver.all_agent_solutions:
            print(f"\nAgent {solution['agent']} ({solution['algorithm']}):")
            print(f"Pattern: {[round(p, 2) for p in solution['pattern']]}")
            print(f"Waste: {solution['waste']:.2f}m | Cuts: {solution['cuts']} | Time: {solution['time']:.4f}s")

            if (solution['waste'] < best_waste) or \
               (solution['waste'] == best_waste and solution['cuts'] < best_cuts):
                best_agent = solution['agent']
                best_waste = solution['waste']
                best_cuts = solution['cuts']

        print(f"\nBest performing agent: Agent {best_agent}")

    # Performans metrikleri
    print("\n=== Performance Metrics ===")
    if method in ["HC", "SA", "GA"] and num_agents == 1:
        print(f"{method} Execution Time: {solver.execution_times.get(method, 0):.4f} seconds")
    elif method in ["HC", "SA", "GA"] and num_agents > 1:
        print(f"Average {method} Time per Agent: {solver.execution_times.get('Collaborative', 0):.4f} seconds")
    elif method == "hyper":
        print(f"Average Time per Agent: {solver.execution_times.get('Hyper', 0):.4f} seconds")
    print(f"Total Execution Time: {solver.execution_times.get('Total', 0):.4f} seconds")

    # Görselleri oluştur
    solver.visualize_solution()

    # Çoklu ajan durumunda ek görseller
    if num_agents > 1:
        solver.plot_method_comparison()
        solver.plot_efficiency_scatter()
        solver.plot_algorithm_distribution()

if __name__ == "__main__":
    main()