import copy
import numpy as np

from utils.genotypes import PRIMITIVES

'''
[ConvOP, Concat]
'''


class evolution(object):
    def __init__(self, args):
        self.population           = []
        self.population_size      = args.population_size
        self.sample_size          = args.sample_size
        self.num_blocks           = args.num_blocks
        self.num_nasnet_blocks    = args.num_nasnet_blocks  # small blocks in cells
        self.op_types             = len(PRIMITIVES)
        self.num_bits             = len(args.Bits.split(','))

    def initialize(self):       # initialize a population P
        for i in range(self.population_size):
            model_code = []
            # normal
            for k in range(self.num_nasnet_blocks):
                block_code = [np.random.randint(self.op_types), np.random.randint(int(2+k/2))] #
                model_code.append(block_code)
            # reduce
            for k in range(self.num_nasnet_blocks):
                block_code = [np.random.randint(self.op_types), np.random.randint(int(2+k/2))] #
                model_code.append(block_code)
            quantz_code = np.random.randint(self.num_bits,size=self.num_blocks).tolist()
            model_code.append(quantz_code)
            score      = 0
            self.population.append([model_code, score])

    def sample(self):                 # sample S models from population
        sample_index = np.random.choice(self.population_size, size=self.sample_size, replace=False).tolist()
        sample       = [self.population[i] for i in sample_index]
        return sample

    def mutate(self, individual):           # mutate the best module to produce a child
        mutate_code   = copy.deepcopy(individual[0])
        # normal
        for i in range(self.num_nasnet_blocks):
            mutate_type  = np.random.randint(2)
            if mutate_type == 0:
                mutate_value = np.random.randint(self.op_types)
            else:
                mutate_value = np.random.randint(int(2+i/2))
            mutate_code[i][mutate_type] = mutate_value
        # reduce
        for i in range(self.num_nasnet_blocks):
            mutate_type  = np.random.randint(2)
            if mutate_type == 0:
                mutate_value = np.random.randint(self.op_types)
            else:
                mutate_value = np.random.randint(int(2+i/2))
            mutate_code[i+self.num_nasnet_blocks][mutate_type] = mutate_value
        mutate_code[-1][np.random.randint(self.num_blocks)] = np.random.randint(self.num_bits)
        mutate_model = [mutate_code,0]
        return mutate_model

    def evolve(self, mutate_one, worst = None):
        print('mutate_one',mutate_one)
        print('worst',worst)
        if mutate_one[1]>0.2 and worst is not None: # mutate_one[1]>0.2 and
            self.population.remove(worst)
            self.population.append(mutate_one)
