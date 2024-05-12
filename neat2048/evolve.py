import multiprocessing
import os
import pickle
import random

import neat
from fitness import calculate_fitness
from fitness_hyper import calculate_fitness as calculate_fitness_hyper
from fitness_optimized import calculate_fitness as calculate_fitness_optimized

ENABLE_HYPER = True
ENABLE_FITNESS_OPTIMIZED = False


class CustomParallelEvaluator(neat.ParallelEvaluator):
    def evaluate(self, genomes, config):
        global_seed = random.randint(0, 1000000)
        jobs = []
        for ignored_genome_id, genome in genomes:
            jobs.append(
                self.pool.apply_async(self.eval_function, (genome, config, global_seed))
            )

        # assign the fitness back to each genome
        for job, (ignored_genome_id, genome) in zip(jobs, genomes):
            genome.fitness = job.get(timeout=self.timeout)


def run(config_file):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    if ENABLE_FITNESS_OPTIMIZED:
        pe = CustomParallelEvaluator(
            multiprocessing.cpu_count(), calculate_fitness_optimized
        )
    if ENABLE_HYPER:
        pe = CustomParallelEvaluator(
            multiprocessing.cpu_count(), calculate_fitness_hyper
        )
    else:
        pe = CustomParallelEvaluator(multiprocessing.cpu_count(), calculate_fitness)
    winner = p.run(pe.evaluate, 200)

    print("\nBest genome:\n{!s}".format(winner))

    print("\nOutput:")
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    with open("output_network.pkl", "wb") as f:
        pickle.dump(winner_net, f)

    print("\nOutput network saved to output_network.pkl")

    stats.save()


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config")
    if ENABLE_HYPER:
        config_path += "-hyper"
    print(config_path)
    run(config_path)
