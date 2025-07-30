import random
import time
import os
from datetime import datetime
import math
import numpy as np
from tqdm import tqdm, trange
from scipy.stats.qmc import Sobol
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager

# ──────────────────────────────────────────────────────────────────────────────
# 1. Load dependency files
# ──────────────────────────────────────────────────────────────────────────────
from evaluate_HD_22 import evaluate
from constraint_utils_numpy_22 import load_s_table, adjust_chromosome

# ──────────────────────────────────────────────────────────────────────────────
# 2. Final configuration definition
# ──────────────────────────────────────────────────────────────────────────────
SEED = 20010205
TIME_LIMIT_SECONDS = 7200

# Problem dimension definition (3x3x10)
N_BINARY = 3 * 3 * 10
N_CUSTOMER = 100
CHROM_LENGTH = N_BINARY + N_CUSTOMER + N_CUSTOMER
POP_SIZE = 200
N_GENERATIONS = 300
TOURNAMENT_SIZE = 4
EARLY_STOP_GAP = 10

FIXED_TRANSPORT = 2

s_table = load_s_table()

# Existing SA parameters
SA_INITIAL_TEMP = 1500000.0
SA_COOLING_RATE = 0.995

# ──────────────────────────────────────────────────────────────────────────────
# 3. Initialize evaluation cache and global variables
# ──────────────────────────────────────────────────────────────────────────────
worker_fitness_cache = None


def init_worker(seed, f_cache):
    """Initializes worker processes, sets a unique seed for each, and loads necessary global objects."""
    global worker_fitness_cache, s_table
    worker_fitness_cache = f_cache

    if s_table is None:
        s_table = load_s_table()

    process_seed = seed + os.getpid()
    random.seed(process_seed)
    np.random.seed(process_seed)


class FitnessCache:
    def __init__(self, shared_dict, shared_hits, shared_misses):
        self.cache, self.hits, self.misses = shared_dict, shared_hits, shared_misses

    def get_fitness(self, chrom):
        key = tuple(chrom)
        if key in self.cache:
            self.hits.value += 1
            return self.cache[key]
        self.misses.value += 1
        f = evaluate(chrom)
        self.cache[key] = f
        return f

    def get_stats(self):
        return f"Cache Hits: {self.hits.value}, Misses: {self.misses.value}"


# ──────────────────────────────────────────────────────────────────────────────
# 4. AMA operation functions
# ──────────────────────────────────────────────────────────────────────────────
def set_seeds(seed_value):
    """Sets the global seed to ensure reproducibility of results."""
    random.seed(seed_value)
    np.random.seed(seed_value)


def create_heuristic_individual():
    """Creates a single individual with a heuristic for the 'b' vector."""
    b = np.zeros(N_BINARY, dtype=int)
    num_products, num_plants, num_dcs = 3, 3, 10
    for k in range(num_products):
        num_plants_to_open = np.random.randint(1, 3)
        open_plant_indices = np.random.choice(range(num_plants), size=num_plants_to_open, replace=False)
        num_dcs_to_select = random.randint(2, 5)
        dc_indices = np.random.choice(range(num_dcs), size=num_dcs_to_select, replace=False)
        for i in open_plant_indices:
            for j in dc_indices:
                idx = k * (num_plants * num_dcs) + i * num_dcs + j
                b[idx] = 1
    d = np.random.randint(1, 11, size=N_CUSTOMER)

    # [MODIFIED] Create the 't' vector based on the FIXED_TRANSPORT setting
    if FIXED_TRANSPORT is not None:
        t = np.full(N_CUSTOMER, FIXED_TRANSPORT, dtype=int)
    else:
        t = np.random.randint(1, 4, size=N_CUSTOMER)

    return {'chrom': np.concatenate([b, d, t]), 'fitness': None}


def create_initial_population(n_total_samples):
    """Creates the initial population using the heuristic method."""
    print(f"Creating {n_total_samples} initial samples using Heuristic strategy...")
    individuals = []

    n_heuristic = n_total_samples

    if n_heuristic > 0:
        print(f" - Generating {n_heuristic} samples using the mixed K-I-J heuristic...")
        for _ in trange(n_heuristic, desc="   - Heuristic samples"):
            individuals.append(create_heuristic_individual())

    return individuals


def tournament_selection(pop):
    """Performs tournament selection to choose parents for the next generation."""
    sel = []
    for _ in range(POP_SIZE):
        aspirants = random.sample(pop, TOURNAMENT_SIZE)
        winner = max(aspirants, key=lambda x: x.get('fitness') or -math.inf)
        sel.append(winner.copy())
    return sel


def template_crossover(p1, p2, crossover_rate):
    """Performs template (mask)-based crossover for all b, d, t vectors."""
    if random.random() >= crossover_rate:
        return [p1.copy(), p2.copy()]  # Do not perform crossover

    p1_chrom, p2_chrom = p1['chrom'], p2['chrom']
    mask = np.random.randint(0, 2, size=CHROM_LENGTH, dtype=bool)
    c1_chrom = np.where(mask, p1_chrom, p2_chrom)
    c2_chrom = np.where(~mask, p1_chrom, p2_chrom)

    return [
        {'chrom': c1_chrom.astype(int), 'fitness': None},
        {'chrom': c2_chrom.astype(int), 'fitness': None}
    ]


def mutate(ind, mutation_rate):
    """Performs mutation on an individual's chromosome."""
    chrom = ind['chrom'].copy()
    mutation_rate_open = mutation_rate / 2.0
    t_rate_multiplier = 3.0
    t_mutation_rate = min(1.0, mutation_rate * t_rate_multiplier)

    for i in range(CHROM_LENGTH):
        # Mutate 'b' vector (binary part)
        if i < N_BINARY:
            if chrom[i] == 1 and random.random() < mutation_rate:
                chrom[i] = 0
            elif chrom[i] == 0 and random.random() < mutation_rate_open:
                chrom[i] = 1
        # [MODIFIED] Mutate 't' vector only if it is not fixed
        elif FIXED_TRANSPORT is None and i >= N_BINARY + N_CUSTOMER and random.random() < t_mutation_rate:
            cur = chrom[i]
            population = [1, 2, 3]
            weights = [1, 1, 1]
            if cur in population:
                idx_to_remove = population.index(cur)
                population.pop(idx_to_remove)
                weights.pop(idx_to_remove)
            new_val = random.choices(population, weights=weights, k=1)[0]
            chrom[i] = new_val
        # Mutate 'd' vector
        elif N_BINARY <= i < N_BINARY + N_CUSTOMER and random.random() < mutation_rate:
            cur = chrom[i]
            choices = list(range(1, 11))
            choices.remove(cur)
            chrom[i] = random.choice(choices)
    return {'chrom': chrom, 'fitness': None}


def traditional_local_search(ind, s_table):
    """Performs a local search using Simulated Annealing principles."""
    global worker_fitness_cache
    original_chrom = ind['chrom']
    original_fitness = worker_fitness_cache.get_fitness(original_chrom)
    current_chrom, current_fitness = original_chrom.copy(), original_fitness
    best_chrom, best_fitness = original_chrom.copy(), original_fitness
    temp = SA_INITIAL_TEMP

    neighbors = []

    # [MODIFIED] Define available neighborhood operators based on whether 't' is fixed
    available_operators = ['d_swap', 'b_close_one', 'b_close_two']
    if FIXED_TRANSPORT is None:
        available_operators.append('t_move')

    for _ in range(3):
        neigh = current_chrom.copy()
        chosen_op = random.choice(available_operators)

        if chosen_op == 't_move':
            t_start_index = N_BINARY + N_CUSTOMER
            idx = random.randrange(t_start_index, CHROM_LENGTH)
            neigh[idx] += random.choice([-1, 1])
            neigh[idx] = np.clip(neigh[idx], 1, 3)
        elif chosen_op == 'd_swap':
            d_start_index, d_end_index = N_BINARY, N_BINARY + N_CUSTOMER
            idx1, idx2 = random.sample(range(d_start_index, d_end_index), 2)
            neigh[idx1], neigh[idx2] = neigh[idx2], neigh[idx1]
        elif chosen_op == 'b_close_one':
            one_indices = np.where(neigh[:N_BINARY] == 1)[0]
            if len(one_indices) >= 1:
                idx_to_flip = random.choice(one_indices)
                neigh[idx_to_flip] = 0
        else:  # chosen_op == 'b_close_two'
            one_indices = np.where(neigh[:N_BINARY] == 1)[0]
            if len(one_indices) >= 2:
                indices_to_flip = random.sample(list(one_indices), 2)
                neigh[indices_to_flip[0]] = 0
                neigh[indices_to_flip[1]] = 0
        neighbors.append(neigh)

    adjusted_neighbors = [adjust_chromosome(n, s_table) for n in neighbors]

    for chrom_to_eval in adjusted_neighbors:
        f_eval = worker_fitness_cache.get_fitness(chrom_to_eval)
        if f_eval > current_fitness:
            current_fitness, current_chrom = f_eval, chrom_to_eval.copy()
        else:
            if temp > 1e-9:
                acceptance_prob = np.exp((f_eval - current_fitness) / temp)
                if random.random() < acceptance_prob:
                    current_fitness, current_chrom = f_eval, chrom_to_eval.copy()
        if current_fitness > best_fitness:
            best_fitness, best_chrom = current_fitness, current_chrom.copy()

    temp *= SA_COOLING_RATE
    final_ind = {'chrom': best_chrom, 'fitness': best_fitness}
    return final_ind


def adjust_and_evaluate_cached(ind):
    """Adjusts a chromosome based on constraints and evaluates its fitness using a cache."""
    global s_table, worker_fitness_cache
    ind['chrom'] = adjust_chromosome(ind['chrom'], s_table)
    original_fitness = worker_fitness_cache.get_fitness(ind['chrom'])
    ind['fitness'] = original_fitness
    return ind


# ──────────────────────────────────────────────────────────────────────────────
# 5. Main AMA
# ──────────────────────────────────────────────────────────────────────────────
def memetic_algorithm(fitness_cache):
    """The main memetic algorithm loop."""
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=os.cpu_count(), initializer=init_worker,
                             initargs=(SEED, fitness_cache)) as pool:

        print(f"1) Creating and evaluating initial population of {POP_SIZE}...")
        init_inds = create_initial_population(POP_SIZE)
        futs = [pool.submit(adjust_and_evaluate_cached, ind) for ind in init_inds]

        population = []
        for fut in tqdm(as_completed(futs), total=len(futs), desc="   - Initial Evaluation"):
            population.append(fut.result())

        population.sort(key=lambda x: x['fitness'], reverse=True)
        best_overall = population[0].copy()
        print("Initial population evaluated.")

        stagnant, global_stagnant, generation_logs, time_log, fitness_log = 0, 0, [], [], []
        gen = 0
        high_mutation_generations_left = 0

        last_d_vector_shake_up_time = 0.0

        while time.time() - start_time < TIME_LIMIT_SECONDS:
            gen += 1
            elapsed_time = time.time() - start_time

            num_elites = int(POP_SIZE * 0.02)
            population.sort(key=lambda x: x.get('fitness') or -math.inf, reverse=True)
            elites = [ind.copy() for ind in population[:num_elites]]

            time_trigger = (elapsed_time - last_d_vector_shake_up_time) >= 1200
            stagnation_trigger = global_stagnant >= 10

            if time_trigger or stagnation_trigger:
                if time_trigger:
                    print(
                        f"\n[TIME TRIGGER] 20-minute interval reached. Triggering D-Vector Shake-up!")
                else:
                    print(
                        f"\n[GLOBAL STAGNATION] Gen={gen}, GlobalStagnant={global_stagnant}. Triggering D-Vector Shake-up!")

                non_elites_indices = list(range(num_elites, POP_SIZE))
                num_to_shake = int(len(non_elites_indices) * 0.25)
                indices_to_shake = random.sample(non_elites_indices, num_to_shake)

                d_map = {i: random.randint(1, 10) for i in range(1, 11)}

                shake_up_futs = []
                for idx in indices_to_shake:
                    ind = population[idx]
                    d_start = N_BINARY
                    d_end = N_BINARY + N_CUSTOMER
                    ind['chrom'][d_start:d_end] = [d_map.get(val, val) for val in ind['chrom'][d_start:d_end]]
                    ind['fitness'] = None
                    shake_up_futs.append(pool.submit(adjust_and_evaluate_cached, ind))

                if shake_up_futs:
                    for fut in tqdm(as_completed(shake_up_futs), total=len(shake_up_futs),
                                    desc="   - D-Vector Shake-up Eval", leave=False):
                        fut.result()

                global_stagnant = 0
                last_d_vector_shake_up_time = elapsed_time

                population.sort(key=lambda x: x.get('fitness') or -math.inf, reverse=True)
                elites = [ind.copy() for ind in population[:num_elites]]

            stagnation_threshold = 10 if elapsed_time >= 5400 else 5
            if stagnant >= stagnation_threshold and high_mutation_generations_left == 0:
                print(
                    f"\n[STAGNATION ALERT] Gen={gen}, Stagnant={stagnant}. Activating high mutation rate for 5 generations!")
                high_mutation_generations_left = 5
                stagnant = 0

            e0 = 2 * random.random() - 1
            e1 = 1 - stagnant / (EARLY_STOP_GAP or 1)
            e = e0 * (e1 * (1 - gen / N_GENERATIONS))
            w_exploit, w_explore = max(0, min(1, 1 - abs(e))), 1 - max(0, min(1, 1 - abs(e)))
            gamma = 4.0
            LOCAL_SEARCH_RATE = 0.05 + 0.25 * (w_exploit ** gamma)
            base_mutation_rate = (2 / CHROM_LENGTH) * (0.50 + 2.50 * (w_explore ** gamma))

            if high_mutation_generations_left > 0:
                mutation_rate = base_mutation_rate * 5.0
                high_mutation_generations_left -= 1
            else:
                mutation_rate = base_mutation_rate
            crossover_rate = 0.90 + 0.10 * (w_explore ** gamma)

            mating = tournament_selection(population)
            offspring = []
            for i in range(0, POP_SIZE, 2):
                p1, p2 = mating[i], mating[i + 1]
                c1, c2 = template_crossover(p1, p2, crossover_rate)
                offspring.extend([mutate(c1, mutation_rate), mutate(c2, mutation_rate)])

            evaluated_offspring = []
            eval_futs = [pool.submit(adjust_and_evaluate_cached, ind) for ind in offspring]
            desc = f"   - Gen {gen} Full Eval"
            for fut in tqdm(as_completed(eval_futs), total=len(eval_futs), desc=desc, leave=False):
                evaluated_offspring.append(fut.result())

            n_ls = int(LOCAL_SEARCH_RATE * len(evaluated_offspring))
            if n_ls > 0:
                ls_indices_to_improve = np.random.choice(range(len(evaluated_offspring)), size=n_ls, replace=False)
                ls_futs_map = {pool.submit(traditional_local_search, evaluated_offspring[i].copy(), s_table): i for i in
                               ls_indices_to_improve}
                ls_desc = f"   - Gen {gen} Local Search"
                for fut in tqdm(as_completed(ls_futs_map), total=len(ls_futs_map), desc=ls_desc, leave=False):
                    improved_ind = fut.result()
                    original_index = ls_futs_map[fut]
                    if improved_ind['fitness'] > evaluated_offspring[original_index]['fitness']:
                        evaluated_offspring[original_index] = improved_ind

            evaluated_offspring.sort(key=lambda x: x.get('fitness') or -math.inf, reverse=True)
            population = elites + evaluated_offspring[:POP_SIZE - num_elites]

            current_best = population[0]
            if current_best['fitness'] > best_overall['fitness']:
                best_overall = current_best.copy()
                stagnant = 0
                global_stagnant = 0
            else:
                stagnant += 1
                global_stagnant += 1

            remaining_time = TIME_LIMIT_SECONDS - elapsed_time
            postfix_str = (
                f"Gen={gen}, Best={best_overall['fitness']:.2f}, Stagnant(S/G)={stagnant}/{global_stagnant}, "
                f"TimeLeft={remaining_time / 60:.1f}m")
            print(postfix_str)
            generation_logs.append(
                f"{postfix_str}, Cache(H/M): {fitness_cache.hits.value}/{fitness_cache.misses.value}")
            time_log.append(elapsed_time)
            fitness_log.append(best_overall['fitness'])

            if gen >= N_GENERATIONS:
                print("\n[Max Generations Reached]")
                break
    return best_overall, generation_logs, time_log, fitness_log


# ──────────────────────────────────────────────────────────────────────────────
# 6. Convergence graph creation function
# ──────────────────────────────────────────────────────────────────────────────
def plot_convergence(time_data, fitness_data, filepath):
    """Plots the convergence curve and saves it to a file."""
    if not time_data or not fitness_data:
        print("Warning: No data to plot for convergence graph.")
        return
    plt.figure(figsize=(12, 7))
    plt.plot(np.array(time_data) / 60, fitness_data, marker='.', linestyle='-', markersize=4)
    plt.title("Convergence Curve: Best Fitness over Time")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Best Fitness Score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# 7. Script execution
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd

    set_seeds(SEED)
    start = time.time()
    print(f"Algorithm started: {datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Running for a maximum of {TIME_LIMIT_SECONDS / 60:.0f} minutes.")
    if FIXED_TRANSPORT is not None:
        print(f"Transport vector ('t') is FIXED to: {FIXED_TRANSPORT}")
    else:
        print("Transport vector ('t') is VARIABLE.")

    best, logs, time_log_data, fitness_log_data = None, None, None, None

    with Manager() as manager:
        shared_dict = manager.dict()
        shared_hits = manager.Value('i', 0)
        shared_misses = manager.Value('i', 0)
        cache = FitnessCache(shared_dict, shared_hits, shared_misses)
        best, logs, time_log_data, fitness_log_data = memetic_algorithm(cache)
        total_evaluations = cache.misses.value

    end = time.time()
    print(f"\nTotal execution time: {end - start:.2f} seconds")
    if best:
        print("Final Best Fitness:", best['fitness'])
    print(f"Total Actual Evaluations (Cache Misses): {total_evaluations}")

    os.makedirs("result", exist_ok=True)
    try:
        script_basename = os.path.basename(__file__).replace('.py', '')
    except NameError:
        script_basename = "interactive_execution"

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_filename = f"result_{script_basename}_{timestamp}.txt"
    path = os.path.join("result", report_filename)

    with open(path, "w", encoding="utf-8") as f:
        if best:
            f.write("=== Best Solution Found ===\n")
            f.write(f"Best Fitness: {best['fitness']}\n")
            chrom_str = np.array2string(best['chrom'], separator=', ')
            f.write(f"Best Chromosome: {chrom_str}\n\n")
        else:
            f.write("=== No solution found ===\n\n")
    print(f"Report saved to: {path}")

    if time_log_data and fitness_log_data:
        df_log = pd.DataFrame({
            'time_seconds': time_log_data,
            'fitness': fitness_log_data
        })
        csv_filename = f"convergence_data_{script_basename}_{timestamp}.csv"
        csv_path = os.path.join("result", csv_filename)
        df_log.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"Convergence data for plotting saved to: {csv_path}")

        plot_filepath = os.path.join("result", f"convergence_plot_{script_basename}_{timestamp}.png")
        plot_convergence(time_log_data, fitness_log_data, plot_filepath)
        print(f"Individual convergence graph saved to: {plot_filepath}")