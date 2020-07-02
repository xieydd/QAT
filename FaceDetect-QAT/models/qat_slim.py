#!/usr/bin/env python
# coding=utf-8
'''
@Author: xieydd
@since: 2020-06-09 11:26:57
@lastTime: 2020-06-16 18:43:31
@LastAuthor: Do not edit
@message: Site :
            https://github.com/pytorch/vision/blob/master/torchvision/models/quantization/shufflenetv2.py

'''
import torch
import torch.nn as nn
import sys
import time
from models.net_slim import Slim
import torch.nn.functional as F


class QuantizableSlim(Slim):
    def __init__(self, phase, *args, **kwargs):
        super(QuantizableSlim, self).__init__(*args, **kwargs)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.phase = phase

    def forward(self, x):
        x = self.quant(x)
        bbox_regressions, classifications, ldm_regressions = self._forward_impl(
            x)
        x1 = self.dequant(bbox_regressions)
        x2 = self.dequant(classifications)
        x3 = self.dequant(ldm_regressions)
        if self.phase == 'train':
            output = (x1, x2, x3)
        else:
            output = (x1, F.softmax(
                x2, dim=-1), x3)
        return output

    def fuse_model(self):
        r"""Fuse conv/bn/relu modules in  model
        Fuse conv+bn+relu/ conv+relu/conv+bn modules to prepare for quantization.
        Model is modified in place.  Note that this operation does not change numerics
        and the model after modification is in floating point

        Operator fusion: you can fuse multiple operations into a single operation, saving on memory access while also improving the operation’s numerical accuracy.
        """

        for name, m in self._modules.items():
            if name in ["conv1"]:
                torch.quantization.fuse_modules(
                    m, [["0", "1", "2"]], inplace=True)

            if name in ['conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7', 'conv8', 'conv9', 'conv10', 'conv11', 'conv12', 'conv13']:
                torch.quantization.fuse_modules(
                    m, [["0", "1", "2"]], inplace=True)
                torch.quantization.fuse_modules(
                    m, [["3", "4", "5"]], inplace=True)

            if name in ['conv14']:
                torch.quantization.fuse_modules(
                    m, [["0", "1"]], inplace=True)

                for m1 in m:
                    if type(m1) == nn.Sequential:
                        torch.quantization.fuse_modules(
                            m1, [["0", "1"]], inplace=True)

            if name in ['loc', 'conf', 'landm']:
                for m1 in m:
                    if type(m1) == nn.Sequential:
                        torch.quantization.fuse_modules(
                            m1, [["0", "1"]], inplace=True)


def _replace_relu(module):
    reassign = {}
    for name, mod in module.named_children():
        _replace_relu(mod)
        # Checking for explicit type instead of instance
        # as we only want to replace modules of the exact type
        # not inherited classes
        if type(mod) == nn.ReLU or type(mod) == nn.ReLU6:
            reassign[name] = nn.ReLU(inplace=False)

    for key, value in reassign.items():
        module._modules[key] = value


def quantize_model(model, backend):
    _dummy_input_data = torch.rand(1, 3, 299, 299)
    if backend not in torch.backends.quantized.supported_engines:
        raise RuntimeError("Quantized backend not supported ")
    torch.backends.quantized.engine = backend
    model.eval()
    # Make sure that weight qconfig matches that of the serialized models
    if backend == 'fbgemm':
        model.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.default_observer,
            weight=torch.quantization.default_per_channel_weight_observer)
    elif backend == 'qnnpack':
        model.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.default_observer,
            weight=torch.quantization.default_weight_observer)

    model.fuse_model()
    torch.quantization.prepare(model, inplace=True)
    model(_dummy_input_data)
    torch.quantization.convert(model, inplace=True)

    return


def slim(quantize_test=False, phase='train', *args, **kwargs):
    model = QuantizableSlim(phase, *args, **kwargs)
    _replace_relu(model)
    if quantize_test:
        # TODO use pretrained as a string to specify the backend
        backend = 'fbgemm'
        quantize_model(model, backend)

    return model


def compare_facedetect(model_file):
    elapsed = 0
    model = torch.jit.load(model_file)
    model.eval()
    num_images = 5
    num_batchs = 5
    # Run the scripted model on a few batches of images
    for i in range(num_batchs):
        images = torch.randn(num_batchs, 3, 300, 300)
        start = time.time()
        output = model(images)
        end = time.time()
        elapsed = elapsed + (end-start)

    print('Elapsed time: %3.0f ms' % (elapsed/(num_images*num_batch)*1000))
    return elapsed
