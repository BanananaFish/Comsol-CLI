import warnings

import numpy
import pygad
import torch

from comsol.model import MLP
from comsol.utils import Config
from comsol.datasets import FieldDataset

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
    Bs: numpy.ndarray = net(torch.tensor([solution]).float()).detach().numpy().flatten()
    return abs(Bs[3] - Bs[2])


def max_min_distance_six(solution, net):
    # line0_0, line1_0, line0_1, line1_1
    # 1 . 3 . 5
    # . . . . .
    # 0 . 2 . 4
    if any(solution < 0) or any(solution > 1.5):
        return -1000
    Bs: numpy.ndarray = net(torch.tensor([solution]).float()).detach().numpy().flatten()
    return min(abs(Bs[3] - Bs[2]), abs(Bs[5] - Bs[4]))


def min_mse_and_central_field(solution, net):
    if any(solution < 0):
        return -1000
    mse, grater_mean = net(torch.tensor([solution]).float()).detach().numpy().flatten()
    return mse + grater_mean


def fit(ckpt, pkl_path, cfg: Config):
    net = MLP(cfg)
    net.load_state_dict(torch.load(ckpt))
    net.eval()
    dataset = FieldDataset(pkl_path, cfg)

    if cfg["dataset"]["sampler"] == "four_points":
        fitness_func = fitness_warper(max_min_distance_four, net)
    elif cfg["dataset"]["sampler"] == "six_points":
        fitness_func = fitness_warper(max_min_distance_six, net)
    elif cfg["dataset"]["sampler"] == "field":
        fitness_func = fitness_warper(min_mse_and_central_field, net)
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
    )
    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    prediction = net(torch.tensor([solution]).float()).detach().numpy().flatten()
    mse, grater_mean = prediction
    params_dict = dict(zip(cfg["cell"].keys(), solution))
    solution, mse = dataset.denorm_params(params_dict), dataset.denorm_mse(mse)
    # console.log(f"Parameters of the best solution : {solution}")
    console.log(f"BEST Parameters : {solution}")
    console.log(f"Predicted Band Outputs : {mse=}, {grater_mean=}")
    console.log(f"Fitness: {solution_fitness}")


# TODO: auto evaluate
