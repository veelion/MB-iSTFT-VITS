import time
import os
import torch
import torch.onnx
from torch import nn
import onnx

from models import SynthesizerTrn
import utils
from text.symbols import symbols


class ISTFT_VITS(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, x_lengths, scales, sid):
        res = self.model.infer(x, x_lengths,
                               noise_scale=scales[0],
                               length_scale=scales[1],
                               noise_scale_w=scales[2],)
        print(f'{res[0].shape = }')
        return res[0][0, 0]


def convert(model_name):
    config_path = f'configs/ljs_{model_name}.json'
    ckp_path = f'checkpoints/pretrained_{model_name}_ddp.pth'
    print(f'{config_path = }')
    hps = utils.get_hparams_from_file(config_path)
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model)
    _ = net_g.eval()
    _ = utils.load_checkpoint(ckp_path, net_g, None)
    model = ISTFT_VITS(net_g)
    model.eval()

    x = torch.randint(156, (1, 51))
    sid = torch.LongTensor([1])
    x_length = torch.LongTensor([51])
    scales = torch.tensor([0.667, 1, 0.8])
    print(f'{x.shape = }, {x = }')
    print(f'{x_length.shape = }, {x_length = }')
    onnx_name = f'onnx_models/{model_name}.onnx'
    # inputs = (x, x_length, noise_scale, noise_scale_w, length_scale)
    # input_names = ['feat', 'feat_len', 'noise_scale',
    #              'noise_scale_w', 'length_scale']
    inputs = (x, x_length, scales, sid)
    input_names = ['input', 'input_lengths', 'scales', 'sid']

    # Export the model
    torch.onnx.export(
        model,  # model being run
        inputs,  # model input (or a tuple for multiple inputs)
        onnx_name,  # where to save the model
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=17,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names = input_names,  # the model's input names
        output_names = ['output'], # the model's output names
        dynamic_axes={
            'input' : {0: 'batch', 1: 'phonemes'},    # variable length axes
            'input_lengths': {0: 'batch'},
            'scales': {0: 'batch'},
            'sid': {0: 'batch'},
            'output': {0: 'batch', 1: 'audio'},
        },
    )
    print(" ")
    print('Model has been converted to ONNX')
    onnxmodel = onnx.load(onnx_name)
    onnx.checker.check_model(onnxmodel)
    print(onnx.helper.printable_graph(onnxmodel.graph))
    print('check_model done')


if __name__ == "__main__":
    model_names = [
        'ms_istft_vits',
        'mini_mb_istft_vits',
    ]
    convert(model_names[1])
