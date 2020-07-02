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

# CUDA_VISUAL_DEVICES=-1 python3 detect.py -m ./weights/slim_epoch_40.pth --network=slim-qat  --cpu
# python detect.py -m ./slim_Final.pth
parser = argparse.ArgumentParser(description='Test')
parser.add_argument('-m', '--trained_model', default='./weights/slim_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='slim',
                    help='Backbone network mobile0.25 or slim or RFB')
parser.add_argument('--origin_size', default=False, type=str,
                    help='Whether use origin image size to evaluate')
parser.add_argument('--long_side', default=320, type=int,
                    help='when origin_size is false, long_side is scaled size(320 or 640 for long side)')
parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/',
                    type=str, help='Dir to save txt results')
parser.add_argument('--cpu', action="store_true",
                    default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02,
                    type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4,
                    type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('--save_image', action="store_true",
                    default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float,
                    help='visualization_threshold')
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
        newImage = np.zeros((stdh, std2, 3), np.uint8)
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
        torch.quantization.convert(net, inplace=True)
        print_size_of_model(net)
        net.save(net.state_dict(), 'quantized_post_train_model.pth')
        print(net)
    else:
        device = torch.device("cpu" if args.cpu else "cuda")
        net = net.to(device)
        print_size_of_model(net)

    # testing begin
    face_count = 0
    count = 0
    img_dir = './img/'
    result_dir = "./result"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    else:
        shutil.rmtree(result_dir)
        os.mkdir(result_dir)
    for img_path in os.listdir(img_dir):
        if img_path == "1.jpg":
            continue
        image_path = os.path.join(img_dir, img_path)
        #image_path = "./img/face1.jpg"
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        count += 1
        img = np.float32(img_raw)

        # testing scale
        target_size = args.long_side
        max_size = args.long_side
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        if args.origin_size:
            resize = 1

        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize,
                             fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape

        scale = torch.Tensor(
            [img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        tic = time.time()
        if args.network == 'slim-qat':
            torch.quantization.convert(net, inplace=True)
        loc, conf, landms = net(img)  # forward pass

        _, line, _ = conf.shape
        f = open("score.txt", 'w')
        for l in range(line):
            a, b = conf[:, l, :][0]
            a = a.item()
            b = b.item()
            f.write(str(a) + " " + str(b)+'\n')
        print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0),
                              prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(
            np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        landms = landms[:args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)

        # show image
        flag = False
        if args.save_image:
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                print("Score: ", b[4])
                if not flag:
                    face_count = face_count + 1
                    flag = True
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                print(b[0], b[1], b[2], b[3])
                cv2.rectangle(img_raw, (b[0], b[1]),
                              (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms
                cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
        # save image
        cv2.imwrite(result_dir+"/"+img_path, img_raw)
        print("face_count: ", face_count)
    print("count: ", count)
    print("recall: ", face_count/count)
