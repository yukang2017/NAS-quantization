import torch
import quantization.quant_functions as q_f
import quantization.help_functions


def quantize(model,
             numBits=8,
             bucket_size=256,
             backprop_quantization_style='none',
             quantize_first_and_last_layer=True,
             quantizationFunctionToUse='uniformLinearScaling',
             ):
    quantizationFunctionToUse = quantizationFunctionToUse.lower()
    if backprop_quantization_style is None:
        backprop_quantization_style = 'none'
    backprop_quantization_style = backprop_quantization_style.lower()
    if quantizationFunctionToUse == 'uniformAbsMaxScaling'.lower():
        s = 2 ** (numBits - 1)
        type_of_scaling = 'absmax'
    elif quantizationFunctionToUse == 'uniformLinearScaling'.lower():
        s = 2 ** numBits
        type_of_scaling = 'linear'
    else:
        raise ValueError('The specified quantization function is not present')

    if backprop_quantization_style is None or backprop_quantization_style in ('none', 'truncated'):
        quantizeFunctions = lambda x: q_f.uniformQuantization(x, s,
                                                                       type_of_scaling=type_of_scaling,
                                                                       stochastic_rounding=False,
                                                                       max_element=False,
                                                                       subtract_mean=False,
                                                                       modify_in_place=False, bucket_size=bucket_size)[0]

    elif backprop_quantization_style == 'complicated':
        quantizeFunctions = [q_f.uniformQuantization_variable(s, type_of_scaling=type_of_scaling,
                                                                       stochastic_rounding=False,
                                                                       max_element=False,
                                                                       subtract_mean=False,
                                                                       modify_in_place=False, bucket_size=bucket_size)
                             for _ in model.parameters()]
    else:
        raise ValueError('The specified backprop_quantization_style not recognized')

    num_parameters = sum(1 for _ in model.parameters())

    def quantize_weights_model(model):
        for idx, (name, p) in enumerate(model.named_parameters()):
            #print(name)
            #if 'batch' in name:
            #    continue
            if quantize_first_and_last_layer is False:
                if idx == 0 or idx == num_parameters - 1:
                    continue  # don't quantize first and last layer
            if backprop_quantization_style == 'truncated':
                p.data.clamp_(-1, 1)
            if backprop_quantization_style in ('none', 'truncated'):
                p.data = quantizeFunctions(p.data)
            elif backprop_quantization_style == 'complicated':
                p.data = quantizeFunctions[idx].forward(p.data)
            else:
                raise ValueError

    quantize_weights_model(model)
    return model


def quantize_rl(model,
                qtz_actions={},
                bucket_size=256,
                backprop_quantization_style='none',
                quantizationFunctionToUse='uniformLinearScaling',
                ):
    num_parameters = sum(1 for _ in model.parameters())
    if not len(qtz_actions) == num_parameters:
        for name, _ in model.named_parameters():
            if not name in qtz_actions:
                qtz_actions[name] = -1

    quantizationFunctionToUse = quantizationFunctionToUse.lower()
    if backprop_quantization_style is None:
        backprop_quantization_style = 'none'
    backprop_quantization_style = backprop_quantization_style.lower()

    if quantizationFunctionToUse == 'uniformAbsMaxScaling'.lower():
        s = lambda numBits: 2 ** (numBits - 1)
        type_of_scaling = 'absmax'
    elif quantizationFunctionToUse == 'uniformLinearScaling'.lower():
        s = lambda numBits: 2 ** numBits
        type_of_scaling = 'linear'

    quantizeFunctions = lambda x,s: q_f.uniformQuantization(x, s,
                                                            type_of_scaling=type_of_scaling,
                                                            stochastic_rounding=False,
                                                            max_element=False,
                                                            subtract_mean=False,
                                                            modify_in_place=False, bucket_size=bucket_size)[0]

    for idx, (name, p) in enumerate(model.named_parameters()):
        numBit = qtz_actions[name]

        #if numBit not in [2,4,8,16,-1]:
        #    raise ValueError('Wrong numBit chosen:{}'.format(numBit))
        if numBit<0:
            continue
        s_num = int(s(numBits=numBit))
        if backprop_quantization_style == 'truncated':
            p.data.clamp_(-1, 1)
        p.data = quantizeFunctions(p.data, s_num)

    return model
