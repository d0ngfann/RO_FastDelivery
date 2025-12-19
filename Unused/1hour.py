import random
import time
import os
from datetime import datetime
import math

import numpy as np
from tqdm import tqdm, trange
from scipy.stats.qmc import Sobol
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor

# ──────────────────────────────────────────────────────────────────────────────
# 1. 의존성 파일 불러오기
# ──────────────────────────────────────────────────────────────────────────────
# 아래 파일들이 코드 실행 위치에 존재해야 합니다.
try:
    from evaluate_HD_1 import evaluate
    from constraint_utils_numpy_1 import load_s_table, adjust_chromosome
except ImportError:
    print("Warning: 'evaluate_HD_1' or 'constraint_utils_numpy_1' not found. Using dummy functions.")


    # --- 의존성 파일이 없는 환경을 위한 Dummy 함수 ---
    def evaluate(chrom):
        b_score = -np.sum(chrom[:(3 * 5 * 20)]) * 10
        d_score = np.sum(chrom[(3 * 5 * 20):(3 * 5 * 20) + 100])
        t_score = np.sum(chrom[(3 * 5 * 20) + 100:])
        return b_score + d_score + t_score + random.uniform(-100, 100)


    def load_s_table():
        # 더미 s_table이 len()을 지원하도록 수정
        return np.random.rand(100, 5)


    def adjust_chromosome(chrom, s_table):
        return chrom
    # --- Dummy 함수 끝 ---

# ──────────────────────────────────────────────────────────────────────────────

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager

# ──────────────────────────────────────────────────────────────────────────────
# 2. 최종 구성 정의
# ──────────────────────────────────────────────────────────────────────────────
SEED = 20010205
TIME_LIMIT_SECONDS = 3600  # 1시간 실행 제한

N_BINARY = 3 * 5 * 20
N_CUSTOMER = 100
CHROM_LENGTH = N_BINARY + N_CUSTOMER + N_CUSTOMER
POP_SIZE = 300
N_GENERATIONS = 20000  # 시간 제한의 보조 역할
TOURNAMENT_SIZE = 4
EARLY_STOP_GAP = 30  # 동적 파라미터 계산에 사용

# 초기 샘플 수 축소
SURROGATE_INITIAL_SAMPLES = 6000

# [오류 수정] 메인 프로세스에서도 s_table이 필요하므로 여기서 로드합니다.
# 워커 프로세스는 init_worker에서 각자 알아서 로드하므로 서로 영향을 주지 않습니다.
s_table = load_s_table()

# 기존 SA 파라미터
SA_INITIAL_TEMP = 1500000.0
SA_COOLING_RATE = 0.995

# 앙상블 모델 수
N_SURROGATE_MODELS = 10  # 앙상블에 사용할 RandomForest 모델 수

# ──────────────────────────────────────────────────────────────────────────────
# 3. 평가 캐시 및 전역 변수 초기화
# ──────────────────────────────────────────────────────────────────────────────
worker_fitness_cache = None

def init_worker(seed, f_cache):
    """워커 프로세스를 초기화하고, 각자 고유한 시드를 설정하며, 필요한 전역 객체를 로드합니다."""
    global worker_fitness_cache, s_table
    worker_fitness_cache = f_cache

    # s_table은 각 워커 프로세스에서 한 번만 로드합니다.
    if s_table is None:
        s_table = load_s_table()

    # 메인 시드와 프로세스 ID를 조합하여 고유하고 결정적인 시드 생성
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
# 4. GA 연산 함수들 (전략에 맞게 수정)
# ──────────────────────────────────────────────────────────────────────────────
def set_seeds(seed_value):
    """전역 시드를 설정하여 결과의 재현성을 보장합니다."""
    random.seed(seed_value)
    np.random.seed(seed_value)


class SurrogateEnsemble:
    """여러 개의 RandomForest 모델을 관리하는 앙상블 클래스."""

    def __init__(self, n_models=10, seed=None):
        self.n_models = n_models
        self.seed = seed
        self.models = self._create_models()

    def _create_models(self):
        """과적합 방지 하이퍼파라미터가 적용된 RandomForest 모델 10개를 생성합니다."""
        models = []
        print(f"Creating an ensemble of {self.n_models} RandomForest models with anti-overfitting settings.")
        for i in range(self.n_models):
            model_seed = self.seed + i if self.seed is not None else None
            models.append(RandomForestRegressor(
                n_estimators=100,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=3,
                max_features='sqrt',
                random_state=model_seed,
                n_jobs=-1
            ))
        return models

    def fit(self, X, y):
        print(f"  - Training surrogate ensemble with {self.n_models} models...")
        n_samples = X.shape[0]
        for model in self.models:
            # Use iloc for data selection as requested in user preferences.
            # However, this code uses numpy arrays, not pandas DataFrames.
            # np.random.choice is the numpy equivalent for this operation.
            indices = np.random.choice(n_samples, size=int(n_samples * 0.8), replace=True)
            X_sample, y_sample = X[indices], y[indices]
            model.fit(X_sample, y_sample)

    def predict(self, X):
        if X.shape[0] == 0:
            return np.array([]), np.array([])
        predictions = np.array([model.predict(X) for model in self.models])
        mu = np.mean(predictions, axis=0)
        sigma = np.std(predictions, axis=0)
        return mu, sigma


def calculate_expected_improvement(mu, sigma, f_best, xi=0.01):
    if mu.shape[0] == 0:
        return np.array([])
    sigma_safe = np.maximum(sigma, 1e-9)
    improvement = mu - f_best - xi
    Z = improvement / sigma_safe
    ei = improvement * norm.cdf(Z) + sigma_safe * norm.pdf(Z)
    return ei


def calculate_d_diversity_penalty(chrom, penalty_coefficient=1000000.0):
    d_part = chrom[N_BINARY: N_BINARY + N_CUSTOMER]
    num_unique_d = len(np.unique(d_part))
    if 2 <= num_unique_d <= 7:
        return 0.0
    if num_unique_d < 2:
        deviation = 2 - num_unique_d
    else:  # num_unique_d > 7
        deviation = num_unique_d - 7
    penalty = penalty_coefficient * (deviation ** 2)
    return penalty


def create_heuristic_individual():
    b = np.zeros(N_BINARY, dtype=int)
    num_products, num_plants, num_dcs = 3, 5, 20
    for k in range(num_products):
        num_plants_to_open = np.random.randint(1, 3)
        open_plant_indices = np.random.choice(range(num_plants), size=num_plants_to_open, replace=False)
        num_dcs_to_select = random.randint(2, 7)
        dc_indices = np.random.choice(range(num_dcs), size=num_dcs_to_select, replace=False)
        for i in open_plant_indices:
            for j in dc_indices:
                idx = k * (num_plants * num_dcs) + i * num_dcs + j
                b[idx] = 1
    d = np.random.randint(1, 21, size=N_CUSTOMER)
    t = np.random.randint(1, 4, size=N_CUSTOMER)
    return {'chrom': np.concatenate([b, d, t]), 'fitness': None}


def create_initial_population(n_total_samples):
    print(f"Creating {n_total_samples} initial samples using Sobol + Heuristic strategy...")
    individuals = []
    n_sobol = 4096
    if n_total_samples < n_sobol:
        n_sobol, n_heuristic = n_total_samples, 0
    else:
        n_heuristic = n_total_samples - n_sobol

    if n_sobol > 0:
        print(f" - Generating {n_sobol} samples using Scrambled Sobol sequence...")
        sobol_sampler = Sobol(d=CHROM_LENGTH, scramble=True, seed=SEED)
        samples_normalized = sobol_sampler.random(n=n_sobol)
        for s_norm in tqdm(samples_normalized, desc="   - Sobol samples"):
            b = np.round(s_norm[:N_BINARY]).astype(int)
            d_start, t_start = N_BINARY, N_BINARY + N_CUSTOMER
            d = (np.round(np.array(s_norm[d_start:t_start]) * 19) + 1).astype(int)
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


def hybrid_blx_crossover(p1, p2, crossover_rate):
    if random.random() >= crossover_rate:
        return [p1.copy(), p2.copy()]
    p1_chrom, p2_chrom = p1['chrom'], p2['chrom']
    c1_chrom, c2_chrom = np.zeros_like(p1_chrom), np.zeros_like(p2_chrom)
    mask = np.random.randint(0, 2, size=N_BINARY, dtype=bool)
    c1_chrom[:N_BINARY] = np.where(mask, p1_chrom[:N_BINARY], p2_chrom[:N_BINARY])
    c2_chrom[:N_BINARY] = np.where(mask, p2_chrom[:N_BINARY], p1_chrom[:N_BINARY])
    alpha = 0.5
    for i in range(N_BINARY, CHROM_LENGTH):
        gene1, gene2 = p1_chrom[i], p2_chrom[i]
        diff = abs(gene1 - gene2)
        min_val, max_val = min(gene1, gene2) - alpha * diff, max(gene1, gene2) + alpha * diff
        c1_gene, c2_gene = random.uniform(min_val, max_val), random.uniform(min_val, max_val)
        if i < N_BINARY + N_CUSTOMER:
            c1_chrom[i], c2_chrom[i] = np.clip(np.round(c1_gene), 1, 20), np.clip(np.round(c2_gene), 1, 20)
        else:
            c1_chrom[i], c2_chrom[i] = np.clip(np.round(c1_gene), 1, 3), np.clip(np.round(c2_gene), 1, 3)
    return [{'chrom': c1_chrom.astype(int), 'fitness': None}, {'chrom': c2_chrom.astype(int), 'fitness': None}]


def mutate(ind, mutation_rate):
    chrom = ind['chrom'].copy()
    mutation_rate_open = mutation_rate / 5.0
    for i in range(CHROM_LENGTH):
        if i < N_BINARY:
            if chrom[i] == 1 and random.random() < mutation_rate:
                chrom[i] = 0
            elif chrom[i] == 0 and random.random() < mutation_rate_open:
                chrom[i] = 1
        elif random.random() < mutation_rate:
            cur = chrom[i]
            choices = list(range(1, 21 if i < N_BINARY + N_CUSTOMER else 4))
            choices.remove(cur)
            chrom[i] = random.choice(choices)
    return {'chrom': chrom, 'fitness': None}


def strong_mutate(ind):
    """
    정체 상태 탈출을 위해 요청된 규칙에 따라 공격적인 변이를 수행합니다.
    """
    chrom = ind['chrom'].copy()

    # 1. b 부분: 1인 것들 중 10개를 선정하여 1 -> 0으로 바꾸기
    one_indices = np.where(chrom[:N_BINARY] == 1)[0]
    num_to_flip = min(10, len(one_indices))
    if num_to_flip > 0:
        indices_to_flip = np.random.choice(one_indices, num_to_flip, replace=False)
        chrom[indices_to_flip] = 0

    # 2. d 부분: distinct DC 개수가 7개 초과 시, 7개로 강제 조정
    d_start_idx = N_BINARY
    d_end_idx = N_BINARY + N_CUSTOMER
    d_part = chrom[d_start_idx:d_end_idx]

    unique_dcs, counts = np.unique(d_part, return_counts=True)

    if len(unique_dcs) > 7:
        sorted_dcs_by_freq = sorted(zip(unique_dcs, counts), key=lambda item: item[1], reverse=True)
        top_7_dcs = [dc for dc, count in sorted_dcs_by_freq[:7]]
        loser_dcs = [dc for dc, count in sorted_dcs_by_freq[7:]]
        for loser_dc in loser_dcs:
            indices_to_change = np.where(d_part == loser_dc)[0]
            for i in indices_to_change:
                d_part[i] = random.choice(top_7_dcs)
        chrom[d_start_idx:d_end_idx] = d_part

    # 3. t 부분: 10명 랜덤 선택하여 수송 방법 변경
    t_start_idx = N_BINARY + N_CUSTOMER
    t_indices_to_change = np.random.choice(range(t_start_idx, CHROM_LENGTH), 10, replace=False)
    for idx in t_indices_to_change:
        cur = chrom[idx]
        choices = [1, 2, 3]
        choices.remove(cur)
        chrom[idx] = random.choice(choices)

    return {'chrom': chrom, 'fitness': None}


def remap_d_composition(ind):
    """
    염색체의 d-part가 사용하는 DC 구성을 완전히 새로운 DC 집합으로 변경합니다.
    """
    chrom = ind['chrom'].copy()
    d_start_idx = N_BINARY
    d_end_idx = N_BINARY + N_CUSTOMER
    d_part = chrom[d_start_idx:d_end_idx]

    old_dcs = np.unique(d_part)
    num_distinct_dcs = len(old_dcs)
    all_possible_dcs = list(range(1, 21))
    new_dc_candidates = [dc for dc in all_possible_dcs if dc not in old_dcs]

    if len(new_dc_candidates) < num_distinct_dcs:
        new_dcs = np.random.choice(all_possible_dcs, num_distinct_dcs, replace=False)
    else:
        new_dcs = np.random.choice(new_dc_candidates, num_distinct_dcs, replace=False)

    mapping = {old_dc: new_dc for old_dc, new_dc in zip(old_dcs, new_dcs)}
    new_d_part = np.array([mapping[val] for val in d_part])
    chrom[d_start_idx:d_end_idx] = new_d_part
    return {'chrom': chrom, 'fitness': None}


def surrogate_assisted_local_search(ind, surrogate_ensemble, s_table):
    global worker_fitness_cache
    original_chrom = ind['chrom']
    original_fitness = worker_fitness_cache.get_fitness(original_chrom)

    current_chrom = original_chrom.copy()
    current_fitness = original_fitness
    best_chrom = original_chrom.copy()
    best_fitness = original_fitness
    temp = SA_INITIAL_TEMP

    neighbors = []
    for _ in range(10):
        neigh = current_chrom.copy()
        rand_choice = random.random()
        if rand_choice < 0.20:
            t_start_index = N_BINARY + N_CUSTOMER
            idx = random.randrange(t_start_index, CHROM_LENGTH)
            neigh[idx] += random.choice([-1, 1])
            neigh[idx] = np.clip(neigh[idx], 1, 3)
        elif rand_choice < 0.35:
            d_start_index = N_BINARY
            d_end_index = N_BINARY + N_CUSTOMER
            idx1, idx2 = random.sample(range(d_start_index, d_end_index), 2)
            neigh[idx1], neigh[idx2] = neigh[idx2], neigh[idx1]
        elif rand_choice < 0.70:
            b_part = neigh[:N_BINARY]
            one_indices = np.where(b_part == 1)[0]
            if len(one_indices) >= 2:
                indices_to_flip = random.sample(list(one_indices), 2)
                neigh[indices_to_flip[0]] = 0
                neigh[indices_to_flip[1]] = 0
        else:
            b_part = neigh[:N_BINARY]
            one_indices = np.where(b_part == 1)[0]
            if len(one_indices) >= 4:
                indices_to_flip = random.sample(list(one_indices), 4)
                neigh[indices_to_flip[0]] = 0
                neigh[indices_to_flip[1]] = 0
                neigh[indices_to_flip[2]] = 0
                neigh[indices_to_flip[3]] = 0
        neighbors.append(neigh)

    adjusted_neighbors = [adjust_chromosome(n, s_table) for n in neighbors]
    X_neighbors = [feature_transform(n) for n in adjusted_neighbors]
    if not X_neighbors: return {'chrom': best_chrom, 'fitness': best_fitness}, []

    pred_fitness, _ = surrogate_ensemble.predict(np.vstack(X_neighbors))
    top_3_indices = np.argsort(pred_fitness)[-3:]

    newly_evaluated_data = []
    for i in top_3_indices:
        chrom_to_eval = adjusted_neighbors[i]
        f_eval = worker_fitness_cache.get_fitness(chrom_to_eval)
        newly_evaluated_data.append((feature_transform(chrom_to_eval), f_eval))

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
    return final_ind, newly_evaluated_data


def feature_transform(chrom):
    return chrom.astype(float)


def adjust_and_evaluate_cached(ind):
    global s_table, worker_fitness_cache
    ind['chrom'] = adjust_chromosome(ind['chrom'], s_table)
    original_fitness = worker_fitness_cache.get_fitness(ind['chrom'])
    penalty = calculate_d_diversity_penalty(ind['chrom'])
    ind['fitness'] = original_fitness - penalty
    return ind


# ──────────────────────────────────────────────────────────────────────────────
# 5. 메인 메메틱 알고리즘 (1시간 최적화 전략 적용)
# ──────────────────────────────────────────────────────────────────────────────
def memetic_algorithm(fitness_cache):
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=os.cpu_count(),
                             initializer=init_worker,
                             initargs=(SEED, fitness_cache)) as pool:

        init_inds = create_initial_population(SURROGATE_INITIAL_SAMPLES)
        print("1) Evaluating initial samples...")
        futs = [pool.submit(adjust_and_evaluate_cached, ind) for ind in init_inds]
        training_X_list, training_y_list, processed_inds = [], [], []
        for fut in tqdm(as_completed(futs), total=len(futs), desc="   - Surrogate Init Eval"):
            ind = fut.result()
            processed_inds.append(ind)
            training_X_list.append(feature_transform(ind['chrom']))
            training_y_list.append(ind['fitness'])

        training_X = np.array(training_X_list)
        training_y = np.array(training_y_list)

        surrogate_ensemble = SurrogateEnsemble(n_models=N_SURROGATE_MODELS, seed=SEED)
        surrogate_ensemble.fit(training_X, training_y)
        print("Surrogate ensemble model has been trained.")

        processed_inds.sort(key=lambda x: x['fitness'], reverse=True)
        population = processed_inds[:POP_SIZE]
        best_overall = max(population, key=lambda x: x['fitness']).copy()

        stagnant, generation_logs = 0, []
        last_retrain_gen = 0
        gen = 0

        while time.time() - start_time < TIME_LIMIT_SECONDS:
            gen += 1
            elapsed_time = time.time() - start_time
            best_updated_this_gen = False

            num_elites = int(POP_SIZE * 0.02)
            population.sort(key=lambda x: x.get('fitness') or -math.inf, reverse=True)
            elites = [ind.copy() for ind in population[:num_elites]]

            if stagnant >= 5:
                print(
                    f"\n[STAGNATION ALERT] Gen={gen}, Stagnant={stagnant}. Initiating Population Rejuvenation with new rules!")
                population_to_rejuvenate = population[num_elites:]
                print(f"  - Applying strong mutation to {len(population_to_rejuvenate)} individuals.")
                rejuvenated_individuals = [strong_mutate(ind) for ind in population_to_rejuvenate]
                num_to_remap = len(rejuvenated_individuals) // 3
                indices_to_remap = np.random.choice(range(len(rejuvenated_individuals)), size=num_to_remap,
                                                    replace=False)
                if num_to_remap > 0:
                    print(f"  - Remapping DC composition for {num_to_remap} individuals.")
                    for i in indices_to_remap:
                        rejuvenated_individuals[i] = remap_d_composition(rejuvenated_individuals[i])
                eval_futs = [pool.submit(adjust_and_evaluate_cached, ind) for ind in rejuvenated_individuals]
                newly_evaluated_inds = []
                for fut in tqdm(as_completed(eval_futs), total=len(eval_futs), desc="  - Evaluating Rejuvenated Pop",
                                leave=False):
                    newly_evaluated_inds.append(fut.result())
                population = elites + newly_evaluated_inds
                if newly_evaluated_inds:
                    new_X_rejuvenated = np.array([feature_transform(ind['chrom']) for ind in newly_evaluated_inds])
                    new_y_rejuvenated = np.array([ind['fitness'] for ind in newly_evaluated_inds])
                    training_X = np.vstack([training_X, new_X_rejuvenated])
                    training_y = np.concatenate([training_y, new_y_rejuvenated])
                stagnant = 0
                print("  - Population Rejuvenation complete. Resuming normal evolution.")
                continue

            if elapsed_time < 1800:
                current_ei_xi = 2.0
            elif elapsed_time < 3000:
                current_ei_xi = 2.0 if stagnant >= 5 else 1.0
            else:
                current_ei_xi = 0.1

            E0 = 2 * random.random() - 1
            E1 = 1 - stagnant / (EARLY_STOP_GAP or 1)
            E = E0 * (E1 * (1 - gen / N_GENERATIONS))
            w_exploit = max(0, min(1, 1 - abs(E)))
            w_explore = 1 - w_exploit
            gamma = 4.0
            LOCAL_SEARCH_RATE = 0.05 + 0.25 * (w_exploit ** gamma)
            mutation_rate = (1 / CHROM_LENGTH) * (0.50 + 2.50 * (w_explore ** gamma))
            crossover_rate = 0.90 + 0.10 * (w_explore ** gamma)

            mating = tournament_selection(population)
            offspring = []
            for i in range(0, POP_SIZE, 2):
                p1, p2 = mating[i], mating[i + 1]
                c1, c2 = hybrid_blx_crossover(p1, p2, crossover_rate)
                offspring.extend([mutate(c1, mutation_rate), mutate(c2, mutation_rate)])

            # 이 부분은 메인 프로세스에서 실행되므로, 전역 s_table이 로드되어 있어야 합니다.
            adjusted_offspring_chroms = [adjust_chromosome(ind['chrom'], s_table) for ind in offspring]
            for i, adj_chrom in enumerate(adjusted_offspring_chroms): offspring[i]['chrom'] = adj_chrom

            X_off = np.array([feature_transform(ind['chrom']) for ind in offspring])
            mu, sigma = surrogate_ensemble.predict(X_off)
            for ind, pred_mu in zip(offspring, mu): ind['fitness'] = pred_mu

            ei_scores = calculate_expected_improvement(mu, sigma, best_overall['fitness'], xi=current_ei_xi)
            sorted_offspring_indices = np.argsort(ei_scores)[::-1]

            generation_plus_one = gen + 1
            if generation_plus_one <= 20: current_top_rate = 0.30
            elif generation_plus_one <= 40: current_top_rate = 0.25
            elif generation_plus_one <= 60: current_top_rate = 0.20
            elif generation_plus_one <= 80: current_top_rate = 0.15
            elif generation_plus_one <= 100: current_top_rate = 0.10
            else: current_top_rate = 0.05

            num_elites_to_eval = int(POP_SIZE * current_top_rate)
            num_random_to_eval = int(POP_SIZE * 0.10)
            elite_indices = sorted_offspring_indices[:num_elites_to_eval]
            remaining_indices = sorted_offspring_indices[num_elites_to_eval:]
            if len(remaining_indices) > num_random_to_eval:
                random_indices = np.random.choice(remaining_indices, size=num_random_to_eval, replace=False)
            else:
                random_indices = remaining_indices
            final_eval_indices = np.unique(np.concatenate([elite_indices, random_indices]))

            real_futs = {pool.submit(adjust_and_evaluate_cached, offspring[i]): i for i in final_eval_indices}
            new_training_data_list = []
            evaluated_offspring_indices = []
            for fut in tqdm(as_completed(real_futs), total=len(real_futs), desc=f"  - Gen {gen} Real Eval", leave=False):
                ind = fut.result()
                i = real_futs[fut]
                offspring[i] = ind
                new_training_data_list.append((feature_transform(ind['chrom']), ind['fitness']))
                evaluated_offspring_indices.append(i)

            n_ls = int(LOCAL_SEARCH_RATE * len(evaluated_offspring_indices))
            if n_ls > 0:
                ls_idxs = random.sample(evaluated_offspring_indices, n_ls)
                ls_futs = {
                    pool.submit(surrogate_assisted_local_search, offspring[i].copy(), surrogate_ensemble, s_table): i
                    for i in ls_idxs}
                for fut in tqdm(as_completed(ls_futs), total=len(ls_futs), desc=f"  - Gen {gen} Local Search", leave=False):
                    improved_ind, ls_new_data = fut.result()
                    i = ls_futs[fut]
                    offspring[i] = improved_ind
                    if ls_new_data: new_training_data_list.extend(ls_new_data)

            if new_training_data_list:
                new_X, new_y = zip(*new_training_data_list)
                training_X = np.vstack([training_X, np.array(new_X)])
                training_y = np.concatenate([training_y, np.array(new_y)])
                retrain_triggered = False
                if elapsed_time < 1800:
                    if gen - last_retrain_gen >= 10: retrain_triggered = True
                else:
                    if best_updated_this_gen or (gen - last_retrain_gen >= 20): retrain_triggered = True
                if retrain_triggered:
                    surrogate_ensemble.fit(training_X, training_y)
                    last_retrain_gen = gen

            offspring.sort(key=lambda x: x.get('fitness') or -math.inf, reverse=True)
            population = elites + offspring[:POP_SIZE - num_elites]

            real_evaluated_inds = [ind for ind in population if ind.get('fitness') is not None]
            if real_evaluated_inds:
                current_best = max(real_evaluated_inds, key=lambda x: x['fitness'])
                if current_best['fitness'] > best_overall['fitness']:
                    best_overall = current_best.copy()
                    stagnant = 0
                    best_updated_this_gen = True
                else:
                    stagnant += 1
            else:
                stagnant += 1

            remaining_time = TIME_LIMIT_SECONDS - elapsed_time
            postfix_str = (f"Gen={gen}, Best={best_overall['fitness']:.2f}, Stagnant={stagnant}, "
                           f"EI_xi={current_ei_xi:.2f}, TimeLeft={remaining_time / 60:.1f}m")
            print(postfix_str)
            generation_logs.append(
                f"{postfix_str}, Cache(H/M): {fitness_cache.hits.value}/{fitness_cache.misses.value}")

            if gen >= N_GENERATIONS:
                print("\n[Max Generations Reached]")
                break
    return best_overall, generation_logs


# ──────────────────────────────────────────────────────────────────────────────
# 6. 스크립트 실행
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    set_seeds(SEED)
    start = time.time()
    print(f"Algorithm started: {datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Running for a maximum of {TIME_LIMIT_SECONDS / 60:.0f} minutes.")

    best = None
    logs = None

    with Manager() as manager:
        shared_dict = manager.dict()
        shared_hits = manager.Value('i', 0)
        shared_misses = manager.Value('i', 0)
        cache = FitnessCache(shared_dict, shared_hits, shared_misses)

        best, logs = memetic_algorithm(cache)

    end = time.time()
    print(f"\nTotal execution time: {end - start:.2f} seconds")
    if best:
        print("Final Best Fitness:", best['fitness'])

    os.makedirs("result", exist_ok=True)
    try:
        script_basename = os.path.basename(__file__).replace('.py', '')
    except NameError:
        script_basename = "interactive_execution"

    report_filename = f"result_{script_basename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    path = os.path.join("result", report_filename)

    with open(path, "w", encoding="utf-8") as f:
        f.write("=== Memetic Algorithm Report ===\n")
        f.write(f"Execution Started: {datetime.fromtimestamp(start).strftime('%Y-%m-%d %H%M%S')}\n")
        f.write(f"Execution Finished: {datetime.fromtimestamp(end).strftime('%Y-%m-%d %H%M%S')}\n")
        f.write(f"Total Time: {end - start:.2f} seconds\n\n")

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