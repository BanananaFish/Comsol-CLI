import numpy
import pygad
import torch
from loguru import logger

from comsol.model import MLP
from comsol.utils import BandDataset, Config


def fitness_warper(fitness_metric):
    def fitness_func(ga_instance, solution, solution_idx):
        fitness = fitness_metric(solution)
        return fitness

    return fitness_func


def fit(ckpt, pkl_path, cfg: Config):
    net = MLP(cfg)
    net.load_state_dict(torch.load(ckpt))
    net.eval()
    dataset = BandDataset(pkl_path, cfg)

    def fitness_func(ga_instance, solution, solution_idx):
        if any(solution < 0) or any(solution > 1):
            return -1000
        Bs: numpy.ndarray = (
            net(torch.tensor([solution]).float()).detach().numpy().flatten()
        )
        return Bs[2] - Bs[0]

    fitness_function = fitness_func

    num_generations = 100
    num_parents_mating = 4

    sol_per_pop = 30
    num_genes = 3
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
        fitness_func=fitness_function,
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
    solution, prediction = dataset.denormalization(solution, prediction)
    solution[0] = solution[0] * 360
    logger.info(f"Parameters of the best solution : {solution}")
    logger.info(f"Predicted output based on the best solution : {prediction}")
    logger.info(f"Fitness value of the best solution : {solution_fitness}")
    logger.info(
        f"Fitness value of the best solution denormalization= {solution_fitness * (dataset.res_max - dataset.res_min) + dataset.res_min:.5e}"
    )
