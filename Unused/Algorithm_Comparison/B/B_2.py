import random
import time
import os
from datetime import datetime
import math

import numpy as np
from tqdm import tqdm, trange
from scipy.stats.qmc import Sobol

import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────

from evaluate_HD_2 import evaluate
from constraint_utils_numpy_2 import load_s_table, adjust_chromosome

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager

SEED = 20010205
TIME_LIMIT_SECONDS = 7200

N_BINARY = 3 * 3 * 10
N_CUSTOMER = 100
CHROM_LENGTH = N_BINARY + N_CUSTOMER + N_CUSTOMER
POP_SIZE = 200
N_GENERATIONS = 300
TOURNAMENT_SIZE = 4
EARLY_STOP_GAP = 10

s_table = load_s_table()

worker_fitness_cache = None


def init_worker(seed, f_cache):
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


def set_seeds(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)


def create_heuristic_individual():
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
    t = np.random.randint(1, 4, size=N_CUSTOMER)
    return {'chrom': np.concatenate([b, d, t]), 'fitness': None}


def create_initial_population(n_total_samples):
    print(f"Creating {n_total_samples} initial samples using Sobol + Heuristic strategy...")
    individuals = []

    n_sobol = 0
    n_heuristic = n_total_samples - n_sobol

    if n_sobol > 0:
        print(f" - Generating {n_sobol} samples using Scrambled Sobol sequence...")
        sobol_sampler = Sobol(d=CHROM_LENGTH, scramble=True, seed=SEED)
        samples_normalized = sobol_sampler.random(n=n_sobol)
        for s_norm in tqdm(samples_normalized, desc="   - Sobol samples"):
            b = np.round(s_norm[:N_BINARY]).astype(int)
            d_start, t_start = N_BINARY, N_BINARY + N_CUSTOMER
            d = (np.round(np.array(s_norm[d_start:t_start]) * 9) + 1).astype(int)
            t = (np.round(np.array(s_norm[t_start:]) * 2) + 1).astype(int)
            individuals.append({'chrom': np.concatenate([b, d, t]), 'fitness': None})

    if n_heuristic > 0:
        print(f" - Generating {n_heuristic} samples using the mixed K-I-J heuristic...")
        for _ in trange(n_heuristic, desc="   - Heuristic samples"):
            individuals.append(create_heuristic_individual())

    return individuals


def tournament_selection(pop):
    sel = []
    for _ in range(POP_SIZE):
        aspirants = random.sample(pop, TOURNAMENT_SIZE)
        winner = max(aspirants, key=lambda x: x.get('fitness') or -math.inf)
        sel.append(winner.copy())
    return sel


def template_crossover(p1, p2, crossover_rate):
    if random.random() >= crossover_rate:
        return [p1.copy(), p2.copy()]

    p1_chrom, p2_chrom = p1['chrom'], p2['chrom']
    mask = np.random.randint(0, 2, size=CHROM_LENGTH, dtype=bool)
    c1_chrom = np.where(mask, p1_chrom, p2_chrom)
    c2_chrom = np.where(~mask, p1_chrom, p2_chrom)

    return [
        {'chrom': c1_chrom.astype(int), 'fitness': None},
        {'chrom': c2_chrom.astype(int), 'fitness': None}
    ]


def mutate(ind, mutation_rate):
    chrom = ind['chrom'].copy()
    # [VERSION B - MODIFIED] Simpler mutation logic as adaptive rates are gone
    mutation_rate_open = mutation_rate
    t_mutation_rate = mutation_rate

    for i in range(CHROM_LENGTH):
        if i < N_BINARY:
            if random.random() < mutation_rate:
                chrom[i] = 1 - chrom[i]  # Simple flip
        elif i >= N_BINARY + N_CUSTOMER and random.random() < t_mutation_rate:
            cur = chrom[i]
            population = [1, 2, 3]
            if cur in population:
                population.remove(cur)
            new_val = random.choice(population)
            chrom[i] = new_val
        elif N_BINARY <= i < N_BINARY + N_CUSTOMER and random.random() < mutation_rate:
            cur = chrom[i]
            choices = list(range(1, 11))
            choices.remove(cur)
            chrom[i] = random.choice(choices)
    return {'chrom': chrom, 'fitness': None}




def adjust_and_evaluate_cached(ind):
    global s_table, worker_fitness_cache
    ind['chrom'] = adjust_chromosome(ind['chrom'], s_table)
    original_fitness = worker_fitness_cache.get_fitness(ind['chrom'])
    ind['fitness'] = original_fitness
    return ind


def memetic_algorithm(fitness_cache):
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


        while time.time() - start_time < TIME_LIMIT_SECONDS:
            gen += 1
            elapsed_time = time.time() - start_time

            num_elites = int(POP_SIZE * 0.02)
            population.sort(key=lambda x: x.get('fitness') or -math.inf, reverse=True)
            elites = [ind.copy() for ind in population[:num_elites]]

            mutation_rate = 0.02
            crossover_rate = 0.9

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


def plot_convergence(time_data, fitness_data, filepath):
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



if __name__ == "__main__":
    import pandas as pd

    set_seeds(SEED)
    start = time.time()
    print(f"Algorithm started: {datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Running for a maximum of {TIME_LIMIT_SECONDS / 60:.0f} minutes.")

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
        f.write("=== Memetic Algorithm Report ===\n")
        f.write(f"Execution Started: {datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Execution Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Time: {end - start:.2f} seconds\n\n")
        f.write("=== Evaluation Efficiency ===\n")
        f.write(f"Total actual calls to 'evaluate' function (Cache Misses): {total_evaluations}\n\n")
        if best:
            f.write("=== Best Solution Found ===\n")
            f.write(f"Best Fitness: {best['fitness']}\n")
            chrom_str = np.array2string(best['chrom'], separator=', ')
            f.write(f"Best Chromosome: {chrom_str}\n\n")
        else:
            f.write("=== No solution found ===\n\n")
        if logs:
            f.write("=== Generation Logs ===\n")
            for line in logs:
                f.write(line + "\n")
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
