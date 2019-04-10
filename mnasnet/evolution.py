import copy
import numpy as np

'''
• ConvOP [0,1,2,3] conv/0, sep-conv/1, mobile-ib-conv-3/2, mobile-ib-conv-6/3.
• KernelSize [0,1] 3x3/0, 5x5/1.
• SkipOp [0,1,2,3] max/0, avg/1, id/2, no/3.
• Filters  Fi ...  omit currently.
• Layers [0,1,2,3] 1/0, 2/1, 3/2, 4/3.
• Quantz [0,1,2]   4/0, 8/1, 16/2.

[ConvOP, KernelSize, SkipOp, Layers, Quantz]
'''


class evolution(object):
    def __init__(self, args):
        self.population           = []
        self.population_size      = args.population_size
        self.sample_size          = args.sample_size
        self.num_blocks           = args.num_blocks

    def initialize(self):       # initialize a population P
        for i in range(self.population_size):
            model_code = []
            for k in range(self.num_blocks):
                block_code = [np.random.randint(4), np.random.randint(2), np.random.randint(4), 0, np.random.randint(3)] #
                model_code.append(block_code)
            score      = 0
            self.population.append([model_code, score])

    def sample(self):                 # sample S models from population
        sample_index = np.random.choice(self.population_size, size=self.sample_size, replace=False).tolist()
        sample       = [self.population[i] for i in sample_index]
        return sample

    def mutate(self, individual):           # mutate the best module to produce a child
        mutate_code   = copy.deepcopy(individual[0])
        for i in range(self.num_blocks):
            mutate_type  = np.random.randint(4)
            if mutate_type == 1:
                mutate_value = np.random.randint(2)
            elif mutate_type == 3:
                mutate_value = np.random.randint(3)
                mutate_type += 1
            else:
                mutate_value = np.random.randint(4)
            mutate_code[i][mutate_type] = mutate_value
        mutate_model = [mutate_code,0]

        return mutate_model

    def evolve(self, mutate_one, worst = None):
        print('mutate_one',mutate_one)
        print('worst',worst)
        if mutate_one[1]>0.2 and worst is not None: # mutate_one[1]>0.2 and
            self.population.remove(worst)
            self.population.append(mutate_one)
