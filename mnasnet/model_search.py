import numpy as np

import train_search
from controllers.environment import envs
from helpers import evaluate
from mnasnet.evolution import evolution
from mnasnet.operations import *
from quantization.quantz import quantize_rl


class Block(nn.Module):
    def __init__(self, C_in, kernel_size, skip_op, num_layers, conv_op, stride_first=2, affine=True): #  C_out
        super(Block, self).__init__()
        self._layers = nn.ModuleList()
        self._skips = nn.ModuleList()
        if conv_op==0:      #'conv'
            conv = ReLUConvBN
        elif conv_op==1:    #'sep_conv'
            conv = SepConv
        elif conv_op==2:    #'mib_conv3'
            conv = MBConv3
        elif conv_op==3:    #'mib_conv6'
            conv = MBConv6
        else:
            raise ValueError('Wrong conv layer type {}'.format(type))

        num_layers +=1                  # [0,1,2,3]-->[1,2,3,4]
        kernel_size=2*kernel_size+3     # [0,1]    -->[3,5]

        for layer in range(num_layers):
            if stride_first ==2:
                if layer==0:
                    in_channels = C_in
                    stride = stride_first
                else:
                    in_channels = 2*C_in
                    stride = 1
            else:
                in_channels = C_in
                stride = stride_first
            padding=int((kernel_size-1)/2)
            op = conv(C_in=in_channels, C_out=2*in_channels if layer==0 and stride_first==2 else in_channels,kernel_size=kernel_size,stride=stride,padding=padding,affine=affine)
            skip = Skip(C_in=in_channels, C_out=2*in_channels if layer==0 and stride_first==2 else in_channels,skip_op=skip_op,kernel_size=kernel_size,stride=stride,padding=padding)
            self._layers.append(op)
            self._skips.append(skip)

    def forward(self, x):
        for op, skip in zip(self._layers,self._skips):
            x = op(x) + skip(x)
        return x


class Network(nn.Module):
    def __init__(self, C, num_classes, num_blocks, net_code, stem_multiplier=1):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._num_blocks  = num_blocks
        self._net_code    = net_code

        if not len(net_code)==num_blocks:
            raise ValueError('blocks number {} incompatible with net code {}'.format(num_blocks,net_code))
        C_input = C*stem_multiplier
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=C_input, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(C_input)
        )
        self.blocks = nn.ModuleList()
        downsampleID = [0,2,4]

        for i in range(num_blocks):
            conv_op = net_code[i][0]
            kernel_size = net_code[i][1]
            skip_op = net_code[i][2]
            num_layers = net_code[i][3]
            stride_first = 2 if i in downsampleID else 1
            block = Block(C_in=C_input, kernel_size=kernel_size, skip_op=skip_op, num_layers=num_layers, conv_op=conv_op, stride_first=stride_first, affine=True)
            self.blocks.append(block)
            C_input *= 2 if i in downsampleID else 1
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_input, num_classes)

    def forward(self,x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        out = self.global_pooling(x)
        logits = self.classifier(out.view(out.size(0),-1))
        return logits


if __name__ == '__main__':
    #net_code = [[2, 0, 3, 2], [2, 0, 2, 0], [1, 1, 1, 0]]
    #net_code = [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]
    C = 16
    num_classes = 10
    num_blocks = 7
    args = train_search.args
    #x = Variable(torch.ones(1, 3, 32, 32)).cuda()
    env = envs(args)
    evo = evolution(args)
    evo.initialize()

    amount_params = []
    # initialize score
    for i in range(args.population_size):
        model_code = evo.population[i][0]
        model = Network(C=args.init_channels, num_classes=num_classes, num_blocks=args.num_blocks, net_code=model_code)
        model = model.cuda()
        quantz_actions = {}
        for name, _ in model.named_parameters():
            quantz_actions[name] = 8
            if 'bn' in name:
                quantz_actions[name] = 16

        qModel = quantize_rl(model, qtz_actions=quantz_actions, bucket_size=256)
        params = evaluate.get_size_quantized_model(qModel, numBits=quantz_actions, bucket_size=256)
        print(model_code,params)
        amount_params.append(params)
    print('mean',np.mean(params))
    #y = model(x)
    #print(y)
    #env = envs(args)
    #model.load_state_dict(torch.load('/data/yukang.chen/submit_rl_quantz/uploader/rl_quantization/pretrained_models/mnasnet2.pkl'))
    #top1_acc = env.infer(model)
    #print(top1_acc)
    #train_acc, valid_acc = env.train_test(model)
    #torch.save(model.state_dict(),'params.pkl')


