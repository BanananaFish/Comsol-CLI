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

def bd_wide(solution, net):
    bd_points = net(torch.tensor([solution]).float()).detach().numpy().flatten()
    gaps = []
    for i in range(0, len(bd_points) - 2, 2):
        gaps.append(min(bd_points[i + 2], bd_points[i + 3]) - max(bd_points[i], bd_points[i + 1]))
    return max(gaps)


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
        fitness_func = fitness_warper(bd_wide, net)
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
        
        
    



# TODO: auto evaluate
