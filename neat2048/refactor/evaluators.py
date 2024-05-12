import random

import neat  # type: ignore


class ParallelEvaluatorWithRandomSeed(neat.ParallelEvaluator):
    def __init__(
        self, num_workers, eval_function, timeout=None, min_seed=0, max_seed=1000000
    ):
        super().__init__(num_workers, eval_function, timeout)
        self.min_seed = min_seed
        self.max_seed = max_seed

    def evaluate(self, genomes, config):
        global_seed = random.randint(
            self.min_seed, self.max_seed
        )  # random seed for each generation
        jobs = [
            self.pool.apply_async(self.eval_function, (genome, config, global_seed))
            for ignored_genome_id, genome in genomes
        ]
        # assign the fitness back to each genome
        for job, (ignored_genome_id, genome) in zip(jobs, genomes):
            genome.fitness = job.get(timeout=self.timeout)
