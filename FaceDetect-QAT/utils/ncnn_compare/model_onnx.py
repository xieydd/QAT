from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import time
from data.config import cfg_slim
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
import shutil
from models.net_slim import Slim
from models.qat_slim import slim
from utils.box_utils import decode, decode_landm
from utils.timer import Timer
import onnx
# pip3 install onnx-simplifier onnx
# python3 model_onnx.py --network=slim-qat -m ./weights/slim_Final.pth  --cpu
# python3 -m onnxsim xxx.onnx xxx_simple.onnx
# ncnn/build/tools/onnx/onnx2ncnn xxx.onnx xxx.param xxx.bin
# ncnn/build/tools/ncnnoptimize xxx.param xxx.bin xxx_opt.param xxx_opt.bin
'''
mkdir quantization_data
for i in `ls data/widerface/val/images`
do
    for j in `ls data/widerface/val/images/$i/*.jpg`
    do
        cp $j ./quantization_data/
    done
done
'''
# Attention: 320 is longest side of facedetect job, so need change ncnn2table code, like ncnn_forward
# WARN: swapRB false 不要指定
# ncnn/build/tools/quantization/ncnn2table --param xxx_opt.param --bin xxx_opt.bin --images ./quantization_data --output slim_320.table --mean 104,117,123 --norm 1,1,1 --size 320,320 --thread 8
# ncnn/build/tools/quantization/ncnn2int8 xxx_opt.param xxx_opt.bin xxx_int8.param xxx_int8.bin xxx.table
parser = argparse.ArgumentParser(description='Test')
parser.add_argument('-m', '--trained_model', default='./weights/slim_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='slim',
                    help='Backbone network slim-qat or  slim')
parser.add_argument('--long_side', default=320, type=int,
                    help='when origin_size is false, long_side is scaled size(320 or 640 for long side)')
parser.add_argument('--save_folder', default='./onnx/',
                    type=str, help='Dir to save onnx file')
parser.add_argument('--cpu', action="store_true",
                    default=False, help='Use cpu inference')
args = parser.parse_args()


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


def pad(image, stride=64):
    hasChange = False
    stdw = image.shape[1]
    if stdw % stride != 0:
        stdw += stride - (stdw % stride)
        hasChange = True

    stdh = image.shape[0]
    if stdh % stride != 0:
        stdh += stride - (stdh % stride)
        hasChange = True

    if hasChange:
        newImage = np.zeros((stdh, stdw, 3), np.uint8)
        newImage[:image.shape[0], :image.shape[1], :] = image
        return newImage
    else:
        return image


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    def f(x): return x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(
            pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == '__main__':
    torch.set_grad_enabled(False)

    cfg = None
    net = None
    cfg = cfg_slim
    if args.network == "slim":
        net = Slim(cfg=cfg, phase='test')
    elif args.network == "slim-qat":
        net = slim(False, phase='test')
    else:
        print("Don't support network!")
        exit(0)

    if args.network == "slim-qat":
        # backend is fbgemm for x86 and qnnpack for arm
        torch.backends.quantized.engine = "fbgemm"
        net.fuse_model()
        net.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
        torch.quantization.prepare_qat(net, inplace=True)
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    # print(net)
    cudnn.benchmark = True
    if args.network == "slim-qat":
        device = torch.device("cpu")
        print_size_of_model(net)
        #net.save(net.state_dict(), 'quantized_post_train_model.pth')
        # print(net)
    else:
        device = torch.device("cpu" if args.cpu else "cuda")
        net = net.to(device)
        print_size_of_model(net)

    x_numpy = np.random.rand(1, 3, args.long_side,
                             args.long_side).astype(np.float32)
    x = torch.from_numpy(x_numpy).to(dtype=torch.float)

    input_names = ["x"]
    outputs = net(x)
    saved_path = args.save_folder + "_" + \
        args.network + "_" + str(args.long_side)+".onnx"
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    if args.network == "slim-qat":
        traced = torch.jit.trace(net, x)
        buf = io.BytesIO()
        torch.jit.save(traced, buf)
        buf.seek(0)

        model = torch.jit.load(buf)
        f = io.BytesIO()
        torch.onnx.export(net, x, f, input_names=input_names, example_outputs=saved_path,
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
        f.seek(0)

        onnx_model = onnx.load(f)
        print(onnx_model)
        print('Successful save to onnx and load')
    else:
        torch.onnx.export(net, x, saved_path, verbose=True,
                          input_names=input_names)
        # Load the ONNX model
        # Need pip install onnx
        model = onnx.load(saved_path)

        # Check that the IR is well formed
        onnx.checker.check_model(model)

        # Print a human readable representation of the graph
        onnx.helper.printable_graph(model.graph)
        print('Successful save to onnx and load')
