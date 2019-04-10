import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from controllers.environment import envs
# from model_search import Network
from nasnet import model_search
from nasnet.evolution import evolution
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser('mnasnet')
parser.add_argument('--data',               type=str,   default='../data',      help='location of the data corpus')
parser.add_argument('--batch_size',         type=int,   default=128,            help='batch size')
parser.add_argument('--learning_rate',      type=float, default=0.025,          help='init learning rate')
parser.add_argument('--learning_rate_min',  type=float, default=0.001,          help='min learning rate')
parser.add_argument('--momentum',           type=float, default=0.9,            help='momentum')
parser.add_argument('--weight_decay',       type=float, default=3e-4,           help='weight decay')
parser.add_argument('--report_freq',        type=float, default=50,             help='report frequency')
parser.add_argument('--gpu',                type=str,   default='0',            help='gpu device id')
parser.add_argument('--epochs_step',        type=int,   default=1,              help='num of epochs each model')
parser.add_argument('--epochs',             type=int,   default=1000,          help='num of training epochs')
parser.add_argument('--init_channels',      type=int,   default=36,              help='num of init channels')
parser.add_argument('--model_path',         type=str,   default='saved_models', help='path to save the model')
parser.add_argument('--cutout',             type=bool,  default=False,          help='use cutout')
parser.add_argument('--cutout_length',      type=int,   default=16,             help='cutout length')
parser.add_argument('--drop_path_prob',     type=float, default=0.3,            help='drop path probability')
parser.add_argument('--save',               type=str,   default='EXP',          help='experiment name')
parser.add_argument('--seed',               type=int,   default=2,              help='random seed')
parser.add_argument('--grad_clip',          type=float, default=5,              help='gradient clipping')
parser.add_argument('--population_size',    type=int,   default=16,              help='evolution population size')
parser.add_argument('--sample_size',        type=int,   default=16,              help='evolution sample size')
parser.add_argument('--num_nasnet_blocks',  type=int,   default=8,              help='num of block per cell') # mnasnet 7
parser.add_argument('--num_blocks',         type=int,   default=20,              help='num of block per model') # mnasnet 7
parser.add_argument('--dataset',            type=str,   default='cifar10',      help='cifar10 or imagenet')
parser.add_argument('--Bits',               type=str,   default='4,8,16',       help='Bits in quantization')
parser.add_argument('--bucket_size',        type=int,   default=256,            help='bucket size in quantization')
parser.add_argument('--target_Bit',         type=int,   default=8,              help='build a target model at the begin of reinforcement learning')
parser.add_argument('--test_dataset',       type=str,   default='cifar10',      help='cifar10 or Imagenet')
parser.add_argument('--num_layer_type',     type=int,   default=4,              help='how many layer types used to construct model')
parser.add_argument('--path',               type=str,   default='./')
parser.add_argument('--search_space',       type=str,   default='nasnet',       help='nasnet or mnasnet')

args = parser.parse_args()
args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
args.save = os.path.join(args.path, args.save)
#utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
'''
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
'''
if args.dataset=='cifar10':
    CLASSES = 10
else:
    CLASSES = 1000

def main():
    np.random.seed(args.seed)
    gpus = [int(i) for i in args.gpu.split(',')]

    if len(gpus)==1:
        torch.cuda.set_device(int(args.gpu))
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = {}'.format(args.gpu))
    logging.info("args = %s", args)
    writer = SummaryWriter(log_dir='./job_tboard')
    device = torch.device("cuda:"+args.gpu[0])

    env = envs(args)
    evo = evolution(args)
    evo.initialize()
    params = []
    # initialize score
    for i in range(args.population_size):
        model_code = evo.population[i][0]
        model = model_search.Network(C=args.init_channels, num_classes=CLASSES, num_blocks=args.num_blocks, net_code=model_code)
        if len(gpus)>1:
            model = nn.DataParallel(model, device_ids=gpus)
        model.to(device)
        logging.info("Model{} starts initializing. Model Code:".format(i))
        logging.info(str(model_code))
        fitness,_,param = env.train_test(model,model_code,is_init=True)
        params.append(param)
        evo.population[i][1] = fitness
    env.target_param = np.mean(params)
    print('average parameters',env.target_param)

    for epoch in range(args.epochs):
        sample          = evo.sample()
        best_index      = np.argmax([sample[i][1] for i in range(args.sample_size)])
        worst_index     = np.argmin([sample[i][1] for i in range(args.sample_size)])
        worst           = sample[worst_index]
        best            = sample[best_index]
        mutate          = evo.mutate(best)
        mutate_model    = model_search.Network(C=args.init_channels, num_classes=CLASSES, num_blocks=args.num_blocks, net_code=mutate[0]).cuda()
        if len(gpus)>1:
            mutate_model = nn.DataParallel(mutate_model, device_ids=gpus)

        logging.info("EPOCH {}  Mutated model".format(epoch))
        logging.info(str(mutate[0]))
        mutate[1],_,_   = env.train_test(mutate_model,mutate[0])

        evo.evolve(mutate,worst=worst)

        scores = [evo.population[i][1] for i in range(evo.population_size)]
        logging.info('EPOCH{}  mutated model valid acc {}'.format(epoch, mutate[1]))
        logging.info('EPOCH{}  population accuracy mean {}'.format(epoch, np.mean(scores)))
        logging.info('EPOCH{}  accuracy std {}'.format(epoch, np.std(scores)))
        logging.info('EPOCH{}  population'.format(epoch))
        logging.info(str(evo.population))

        writer.add_scalar('mutate_model_valid_acc', mutate[1], epoch)
        writer.add_scalar('population_accuracy_mean', np.mean(scores), epoch)
        writer.add_scalar('population_accuracy_std', np.std(scores), epoch)

if __name__ == '__main__':
    main()
