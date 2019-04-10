import os
import torch
import torch.nn as nn
import logging
import utils
import datasets
import torchvision.datasets as dset
from torch.autograd import Variable
from quantization.quantz import quantize_rl,quantize
from helpers import evaluate

class envs():
    def __init__(self,args):

        self.criterion      = nn.CrossEntropyLoss().cuda()
        self.init_channels  = args.init_channels
        self.num_blocks     = args.num_blocks
        self.momentum       = args.momentum
        self.weight_decay   = args.weight_decay
        self.learning_rate  = args.learning_rate
        self.grad_clip      = args.grad_clip
        self.epochs         = args.epochs_step
        #self.device         = torch.device("cuda:"+args.gpu[0])
        self.save           = args.save
        self.report_freq    = args.report_freq
        self.args           = args
        self.target_param   = 0.2 # mystery  改成初始化模型的平均值
        self.bucket_size    = 256
        num_workers         = 1
        pin_memory          = False
        shuffle             = True
        self.share_params   = {}
        self.search_space   = args.search_space

        if args.dataset=='cifar10':
            root = os.path.join(os.path.dirname(os.path.abspath(__file__)),args.data)
            print(root)
            train_transform, valid_transform = utils._data_transforms_cifar10(args)
            train_data = dset.CIFAR10(root=root, train=True, download=False, transform=train_transform)
            test_data  = dset.CIFAR10(root=root, train=False, download=False, transform=valid_transform)

            self.trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=shuffle,
                                                           num_workers=num_workers, pin_memory=pin_memory)
            self.testloader  = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=shuffle,
                                                      num_workers=num_workers, pin_memory=pin_memory)
        elif args.dataset=='Imagenet':
            trainFolder = os.path.join(args.data,'train/')
            testFolder = os.path.join(args.data,'val/')
            print('testFolder', testFolder)
            Imagenet = datasets.ImageNet12(trainFolder=trainFolder, testFolder=testFolder)
            self.trainloader= Imagenet.getTrainLoader(batch_size=args.batch_size, num_workers=56)
            print('train data done!')
            self.testloader = Imagenet.getTestLoader(batch_size=args.batch_size, num_workers=56)
            print('test data done!')
        else:
            raise ValueError('Unknown dataset {}'.format(args.dataset))

    def train_test(self, model, model_code=None, is_init=False):
        model.drop_path_prob = 0.0
        pretrained_dict = self.check_params(model)
        pre_params        = filter(lambda p: id(p) in pretrained_dict.keys(),model.parameters())
        new_params        = filter(lambda p: id(p) not in pretrained_dict.keys(),model.parameters())
        if not is_init:
            print('share parameters')
            model.load_state_dict(pretrained_dict,strict=False)

        optimizer = torch.optim.SGD(
            [
                {'params': pre_params},
                {'params': new_params, 'lr': self.learning_rate/10}
            ], lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs) # need to modify

        train_acc, valid_acc = 0, 0
        for epoch in range(self.epochs):
            scheduler.step()
            train_acc, train_obj = self.train(model, optimizer)
            with torch.no_grad():
                valid_acc, valid_obj = self.infer(model)

            logging.info('epoch%d: train_acc %f'%(epoch,train_acc))
            logging.info('epoch%d: valid_acc %f'%(epoch,valid_acc))

        self.share_params  = model.state_dict()
        if model_code is None:
            return valid_acc
        else:
            fitness, qAcc, params = self.quantz(model,model_code)
            return fitness, qAcc, params

    def quantz_actions(self,model, model_code):
        quantz_actions = {}
        # [0,1,2] --> [4,8,16]
        common_name = 'module.blocks.' if self.search_space=='mnasnet' else 'module.cells.'
        for name,_ in model.named_parameters():
            if common_name in name:
                #print('name.lstrip(common_name)',name.lstrip(common_name))
                quantz_actions[name] = 2**(model_code[-1][int(name.lstrip(common_name)[0])]+2)  # 4,8,16
                #quantz_actions[name] = 2**(model_code[-1][int(name.lstrip(common_name)[0])]+1)  # 1,2,4,8,16
            else:
                quantz_actions[name] = 8
            if 'bn' in name:
                quantz_actions[name] = 16
            #print(name,quantz_actions[name])
        return quantz_actions

    def quantz(self, model, model_code):
        quantz_actions = self.quantz_actions(model,model_code)

        qModel = quantize_rl(model,qtz_actions=quantz_actions,bucket_size=self.bucket_size)
        qAcc = evaluate.evaluateModel(qModel,self.testloader ,k=1).cpu().numpy()

        params = evaluate.get_size_quantized_model(qModel, numBits=quantz_actions, bucket_size=self.bucket_size)
        reward = (qAcc)*(params/self.target_param)**(-0.07)
        reward = qAcc
        print('qAcc:{} params:{} reward:{}'.format(qAcc, params, reward))
        return reward, qAcc, params

    def check_params(self,model):
        pretrained_dict = {}
        model_dict  = model.state_dict()
        for k, v in self.share_params.items():
            if k in model_dict:
                if model_dict[k].size() == v.size():
                    pretrained_dict[k] = v
        return pretrained_dict

    def train(self, model, optimizer):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()
        model.train()
        for step, (input, target) in enumerate(self.trainloader):

            input = Variable(input).cuda()
            target = Variable(target).cuda(async=True)

            optimizer.zero_grad()
            logits = model(input)
            loss = self.criterion(logits, target)

            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), self.grad_clip)
            optimizer.step()

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data[0], n)
            top1.update(prec1.data[0], n)
            top5.update(prec5.data[0], n)

            #if step % self.report_freq == 0:
            #    logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

        return top1.avg, objs.avg

    def infer(self, model):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()
        model.eval()

        for step, (input, target) in enumerate(self.testloader):
            input = Variable(input, volatile=True).cuda()
            target = Variable(target, volatile=True).cuda(async=True)

            logits = model(input)
            loss = self.criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data[0], n)
            top1.update(prec1.data[0], n)
            top5.update(prec5.data[0], n)

            #if step % self.report_freq == 0:
            #    logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

        return top1.avg, objs.avg
