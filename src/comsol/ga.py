from functools import partial
import warnings

import numpy as np
import pygad
import torch

from comsol.model import MLP
from comsol.utils import Config
from comsol.datasets import BDDataset, FieldDataset

warnings.filterwarnings("ignore")
from comsol.console import console


def fitness_warper(fitness_metric, net):
    def fitness_func(ga_instance, solution, solution_idx):
        fitness = fitness_metric(solution, net)
        return fitness

    return fitness_func


def max_min_distance_four(solution, net):
    # line0_0, line1_0, line0_1, line1_1
    # 1 . 3
    # . . .
    # 0 . 2
    Bs: np.ndarray = net(torch.tensor([solution]).float()).detach().numpy().flatten()
    return abs(Bs[3] - Bs[2])


def max_min_distance_six(solution, net):
    # line0_0, line1_0, line0_1, line1_1
    # 1 . 3 . 5
    # . . . . .
    # 0 . 2 . 4
    if any(solution < 0) or any(solution > 1):
        return -1000
    Bs: np.ndarray = net(torch.tensor([solution]).float()).detach().numpy().flatten()
    return min(abs(Bs[3] - Bs[2]), abs(Bs[5] - Bs[4]))


def min_mse_and_central_field(solution, net):
    mse, grater_mean = net(torch.tensor([solution]).float()).detach().numpy().flatten()
    return -mse, grater_mean

def central_field(solution, net):
    finall_mean = net(torch.tensor([solution]).float()).detach().numpy().flatten()
    return finall_mean

def bd_wide(solution, net, freq_max, object_freq=3000):
    bd_points = net(torch.tensor([solution]).float()).detach().numpy().flatten()
    gaps = []
    base_points = []
    for i in range(0, len(bd_points) - 2, 2):
        if bd_points[i] > (object_freq / freq_max) or bd_points[i + 1] > (object_freq / freq_max):
            gaps.append(0)
            base_points.append(0)
            continue
        gaps.append(min(bd_points[i + 2], bd_points[i + 3]) - max(bd_points[i], bd_points[i + 1]))
        base_points.append(max(bd_points[i], bd_points[i + 1]))
    max_gap = max(gaps)
    max_gap_index = gaps.index(max_gap)
    ponit_in_gap_max = base_points[max_gap_index]
    print(f"max_gap: {max_gap * freq_max}, point_in_gap_max: {ponit_in_gap_max * freq_max}")
    return max(gaps) - 10 * abs(ponit_in_gap_max - object_freq / freq_max)


def fit(ckpt, pkl_path, cfg: Config):
    net = MLP(cfg)
    net.load_state_dict(torch.load(ckpt))
    net.eval()
    
    if cfg["dataset"]["type"] == "field":
        dataset = FieldDataset(pkl_path, cfg)
    elif cfg["dataset"]["type"] == "bd":
        dataset = BDDataset(pkl_path)

    if cfg["dataset"]["sampler"] == "four_points":
        fitness_func = fitness_warper(max_min_distance_four, net)
    elif cfg["dataset"]["sampler"] == "six_points":
        fitness_func = fitness_warper(max_min_distance_six, net)
    elif cfg["dataset"]["sampler"] == "field":
        fitness_func = fitness_warper(min_mse_and_central_field, net)
    elif cfg["dataset"]["sampler"] == "field_single":
        fitness_func = fitness_warper(central_field, net)
    elif cfg["dataset"]["sampler"] == "10_points":
        fitness_func = fitness_warper(partial(bd_wide, freq_max=dataset.freq_max, object_freq=cfg["ga"]["object_freq"]), net) # type: ignore
    else:
        raise ValueError(f"Unknown sampler: {cfg['dataset']['sampler']}")

    num_generations = 100
    num_parents_mating = 4

    sol_per_pop = 30
    num_genes = len(cfg["cell"].values())
    init_range_low = 0
    init_range_high = 1

    parent_selection_type = "sss"
    keep_parents = 1

    crossover_type = "single_point"

    mutation_type = "random"
    mutation_probability = 0.1
    fitness_per_generation = []
    def on_generation(ga_instance):
        best_fitness = ga_instance.best_solution()[1]  # 获取当前代的最佳适应度
        print(f"Generation {ga_instance.generations_completed}: Best Fitness = {best_fitness}")
        fitness_per_generation.append(best_fitness)
    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_func,
        sol_per_pop=sol_per_pop,
        num_genes=num_genes,
        init_range_low=init_range_low,
        init_range_high=init_range_high,
        parent_selection_type=parent_selection_type,
        keep_parents=keep_parents,
        crossover_type=crossover_type,
        mutation_type=mutation_type,
        mutation_probability=mutation_probability,
        gene_space=dict(low=0, high=1),
        on_generation=on_generation
    )
    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    prediction = net(torch.tensor([solution]).float()).detach().numpy().flatten()
    if isinstance(dataset, FieldDataset):
        mse, grater_mean = prediction
        params_dict = dict(zip(cfg["cell"].keys(), solution))
        solution, mse = dataset.denorm_params(params_dict), dataset.denorm_mse(mse)
        # console.log(f"Parameters of the best solution : {solution}")
        console.log(f"BEST Parameters (not denorm) : {params_dict}")
        console.log(f"BEST Parameters : {solution}")
        console.log(f"Predicted Band Outputs : {mse=}, {grater_mean=}")
        console.log(f"Fitness: {solution_fitness}")
    elif isinstance(dataset, BDDataset):
        denormed_solution = dataset.denorm_params(np.expand_dims(solution, axis=0))
        denormed_bd = dataset.denorm_bd(prediction)
        params_dict = {name: value for name, value in zip(dataset.params_name, denormed_solution[0])}
        console.log(f"BEST Parameters : {params_dict}")
        console.log(f"Predicted Band Outputs : {denormed_bd}")
        console.log(f"Fitness: {solution_fitness}")
    return fitness_per_generation
