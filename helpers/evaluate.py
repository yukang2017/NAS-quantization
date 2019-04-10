import os
import torch
import datasets
import functools
import torch.nn as nn
from torch.autograd import Variable
from quantization.help_functions import get_huffman_encoding_mean_bit_length
from quantization.quant_functions import uniformQuantization

USE_CUDA = torch.cuda.is_available()


def getTestDataset(test_dataset='cifar10',batch_size=32):
    if test_dataset=='Imagenet':
        testFolder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../val/')
        print('testFolder',testFolder)
        Imagenet = datasets.ImageNet12(trainFolder=None,testFolder=testFolder)
        test_loader = Imagenet.getTestLoader(batch_size=batch_size,num_workers=0)
    else:
        datasets.BASE_DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
        try:
            os.mkdir(datasets.BASE_DATA_FOLDER)
        except:
            pass
        cifar10 = datasets.CIFAR10()
        test_loader = cifar10.getTestLoader(batch_size=batch_size,num_workers=0)
    return test_loader


def evaluateModel(model, testLoader, fastEvaluation=True, maxExampleFastEvaluation=10000, k=1):

    'if fastEvaluation is True, it will only check a subset of *maxExampleFastEvaluation* images of the test set'

    model.eval()
    correctClass = 0
    totalNumExamples = 0

    for idx_minibatch, data in enumerate(testLoader):

        # get the inputs
        inputs, labels = data
        inputs, labels = Variable(inputs, volatile=True), Variable(labels)
        if USE_CUDA:
            inputs = inputs.cuda()
            labels = labels.cuda()

        outputs = model(inputs)
        _, topk_predictions = outputs.topk(k, dim=1, largest=True, sorted=True)
        topk_predictions = topk_predictions.t()
        correct = topk_predictions.eq(labels.view(1, -1).expand_as(topk_predictions))

        correctClass += correct.view(-1).float().sum(0, keepdim=True).data[0]
        totalNumExamples += len(labels)

        if fastEvaluation is True and totalNumExamples > maxExampleFastEvaluation:
            break
    return correctClass / totalNumExamples


def get_size_quantized_model(model, numBits=8, bucket_size=256,
                             type_quantization='uniform', quantizeFirstLastLayer=True):
    numTensors = sum(1 for _ in model.parameters())
    if numBits is None:
        return sum(p.numel() for p in model.parameters()) * 4 / 1000000

    'Returns size in MB'
    if isinstance(numBits,int):
        quantization_functions = functools.partial(uniformQuantization, s=2 ** numBits, bucket_size=bucket_size)

        if quantizeFirstLastLayer is True:
            def get_quantized_params():
                return model.named_parameters()
            def get_unquantized_params():
                return iter(())
        else:
            def get_quantized_params():
                return  (p for idx, (name,p) in enumerate(model.named_parameters()) if idx not in (0, numTensors - 1))
            def get_unquantized_params():
                return (p for idx, (name,p) in enumerate(model.named_parameters()) if idx in (0, numTensors - 1))

        count_quantized_parameters = sum(p.numel() for _,p in get_quantized_params())
        count_unquantized_parameters = sum(p.numel() for _,p in get_unquantized_params())

        #Now get the best huffmann bit length for the quantized parameters
        actual_bit_huffmman = get_huffman_encoding_mean_bit_length(get_quantized_params(), quantization_functions,
                                                                       type_quantization, s=2**numBits)
    elif isinstance(numBits,dict):
        if not len(numBits) == numTensors:
            raise ValueError('The num of quantization actions should be same to the num of layers')
        'Returns size in MB'
        quantization_functions = {}
        for name, numBit in numBits.items():
            quantization_functions[name] = functools.partial(uniformQuantization, s=2 ** numBit,
                                                             bucket_size=bucket_size)

        if numBits is None:
            return sum(p.numel() for p in model.parameters()) * 4 / 1000000

        if quantizeFirstLastLayer is True:
            def get_quantized_params():
                return ((name, p) for idx, (name, p) in enumerate(model.named_parameters()) if numBits[name] > 0)

            def get_unquantized_params():
                return ((name, p) for idx, (name, p) in enumerate(model.named_parameters()) if numBits[name] <= 0)
        else:
            def get_quantized_params():
                return ((name, p) for idx, (name, p) in enumerate(model.named_parameters()) if
                        idx not in (0, numTensors - 1) and numBits[name] > 0)

            def get_unquantized_params():
                return ((name, p) for idx, (name, p) in enumerate(model.named_parameters()) if
                        idx in (0, numTensors - 1) or numBits[name] < 0)

        #count_quantized_parameters = sum(p.numel() for _, p in get_quantized_params())
        count_unquantized_parameters = sum(p.numel() for _, p in get_unquantized_params())

        '''
        s = {}
        #print('numBits.values()',numBits.values())
        for name, numBit in numBits.items():
            if numBit<0:
                continue
            s[name] = 2 ** numBit

        # Now get the best huffmann bit length for the quantized parameters
        actual_bit_huffmman = get_huffman_encoding_mean_bit_length(get_quantized_params(), quantization_functions,
                                                                   type_quantization, s=s) if len(s)>0 else 0
        '''

        actual_bit_huffmman = {}
        count_quantized_parameters = 0
        for name, p in get_quantized_params():
            count_quantized_parameters += p.numel()
            numBit = numBits[name]
            if numBit<0:
                continue
            s = 2**numBit
            actual_bit_huffmman[name] = get_huffman_encoding_mean_bit_length([(name,p)], quantization_functions,
                                                                   type_quantization, s=s)
    else:
        raise ValueError('numBits type is wrong {}.'.format(numBits))
    #Now we can compute the size.
    size_mb = 0
    size_mb += count_unquantized_parameters*4 #32 bits / 8 = 4 byte per parameter
    if isinstance(numBits,dict):
        for name, p in get_quantized_params():
            quantized_parameters = p.numel()
            size_mb += actual_bit_huffmman[name]*quantized_parameters/8
    else:
        size_mb += actual_bit_huffmman*count_quantized_parameters/8 #For the quantized parameters we use the mean huffman length
    if bucket_size is not None:
        size_mb += count_quantized_parameters/bucket_size*8  #for every bucket size, we have to save 2 parameters.
                                                             #so we multiply the number of buckets by 2*32/8 = 8
    size_mb = size_mb / 1000000 #to bring it in MB
    return size_mb



def get_size_unquantized_model(model):

    'Returns size in MB'

    count_unquantized_parameters = sum(p.numel() for p in model.parameters())

    #Now we can compute the size.
    size_mb = 0
    size_mb += count_unquantized_parameters*4 #32 bits / 8 = 4 byte per parameter
    size_mb = size_mb / 1000000 #to bring it in MB
    return size_mb
