import utils
import numpy as np
import torch
import torch.nn as nn
import os
# from tqdm import tqdm
import torchvision.ops as ops
import torch.nn.functional as F
from layers.functions.prior_box import PriorBox
from data.config import cfg_slim

# site: https://github.com/supernotman/RetinaFace_Pytorch/blob/master/dataloader.py


class RegressionTransform(nn.Module):
    def __init__(self, mean=None, std_box=None, std_ldm=None):
        super(RegressionTransform, self).__init__()
        if mean is None:
            # self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).cuda()
            self.mean = torch.from_numpy(
                np.array([0, 0, 0, 0]).astype(np.float32))
        else:
            self.mean = mean
        if std_box is None:
            # self.std_box = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()
            self.std_box = torch.from_numpy(
                np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32))
        else:
            self.std_box = std_box
        if std_ldm is None:
            # self.std_ldm = (torch.ones(1,10) * 0.1).cuda()
            self.std_ldm = (torch.ones(1, 10) * 0.1)

    def forward(self, anchors, bbox_deltas, ldm_deltas, img):
        widths = anchors[:, :, 2] - anchors[:, :, 0]
        heights = anchors[:, :, 3] - anchors[:, :, 1]
        ctr_x = anchors[:, :, 0] + 0.5 * widths
        ctr_y = anchors[:, :, 1] + 0.5 * heights

        # Rescale
        ldm_deltas = ldm_deltas * self.std_ldm.cuda()
        bbox_deltas = bbox_deltas * self.std_box.cuda()

        bbox_dx = bbox_deltas[:, :, 0]
        bbox_dy = bbox_deltas[:, :, 1]
        bbox_dw = bbox_deltas[:, :, 2]
        bbox_dh = bbox_deltas[:, :, 3]

        # get predicted boxes
        pred_ctr_x = ctr_x + bbox_dx * widths
        pred_ctr_y = ctr_y + bbox_dy * heights
        pred_w = torch.exp(bbox_dw) * widths
        pred_h = torch.exp(bbox_dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack(
            [pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        # get predicted landmarks
        pt0_x = ctr_x + ldm_deltas[:, :, 0] * widths
        pt0_y = ctr_y + ldm_deltas[:, :, 1] * heights
        pt1_x = ctr_x + ldm_deltas[:, :, 2] * widths
        pt1_y = ctr_y + ldm_deltas[:, :, 3] * heights
        pt2_x = ctr_x + ldm_deltas[:, :, 4] * widths
        pt2_y = ctr_y + ldm_deltas[:, :, 5] * heights
        pt3_x = ctr_x + ldm_deltas[:, :, 6] * widths
        pt3_y = ctr_y + ldm_deltas[:, :, 7] * heights
        pt4_x = ctr_x + ldm_deltas[:, :, 8] * widths
        pt4_y = ctr_y + ldm_deltas[:, :, 9] * heights

        pred_landmarks = torch.stack([
            pt0_x, pt0_y, pt1_x, pt1_y, pt2_x, pt2_y, pt3_x, pt3_y, pt4_x, pt4_y
        ], dim=2)

        # clip bboxes and landmarks
        B, C, H, W = img.shape

        pred_boxes[:, :, ::2] = torch.clamp(
            pred_boxes[:, :, ::2], min=0, max=W)
        pred_boxes[:, :, 1::2] = torch.clamp(
            pred_boxes[:, :, 1::2], min=0, max=H)
        pred_landmarks[:, :, ::2] = torch.clamp(
            pred_landmarks[:, :, ::2], min=0, max=W)
        pred_landmarks[:, :, 1::2] = torch.clamp(
            pred_landmarks[:, :, 1::2], min=0, max=H)

        return pred_boxes, pred_landmarks


def get_detections(img_batch, model, score_threshold=0.5, iou_threshold=0.5):
    model.eval()
    with torch.no_grad():
        out = model(img_batch)
        bboxes, classifications, landmarks = out
        classifications = F.softmax(classifications, dim=-1)
        regressBoxes = RegressionTransform()
        anchors = []
        for img in img_batch:
            cfg = cfg_slim
            priorbox = PriorBox(cfg, image_size=(img.shape[1], img.shape[2]))
            anchors.append(priorbox.forward())
        anchors = torch.stack(anchors)
        bboxes, landmarks = regressBoxes(anchors, bboxes, landmarks, img_batch)
        batch_size = classifications.shape[0]
        picked_boxes = []
        picked_landmarks = []
        picked_scores = []

        for i in range(batch_size):
            #classification = torch.exp(classifications[i, :, :])
            classification = classifications[i, :, :]
            bbox = bboxes[i, :, :]
            landmark = landmarks[i, :, :]

            # choose positive and scores > score_threshold
            scores, argmax = torch.max(classification, dim=1)
            argmax_indice = argmax == 0
            scores_indice = scores > score_threshold
            positive_indices = argmax_indice & scores_indice

            scores = scores[positive_indices]

            if scores.shape[0] == 0:
                picked_boxes.append(None)
                picked_landmarks.append(None)
                picked_scores.append(None)
                continue

            bbox = bbox[positive_indices]
            landmark = landmark[positive_indices]

            keep = ops.boxes.nms(bbox, scores, iou_threshold)
            keep_boxes = bbox[keep]
            keep_landmarks = landmark[keep]
            keep_scores = scores[keep]
            keep_scores.unsqueeze_(1)
            picked_boxes.append(keep_boxes)
            picked_landmarks.append(keep_landmarks)
            picked_scores.append(keep_scores)

        return picked_boxes, picked_landmarks, picked_scores


def compute_overlap(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(
        a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(
        a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) *
                        (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    # (N, K) ndarray of overlap between boxes and query_boxes
    return torch.from_numpy(intersection / ua)


def evaluate_widerface(img_batch, annots, model, threshold=0.5):
    img_batch = img_batch.cuda()
    annots = [_annot.cuda() for _annot in annots]

    picked_boxes, _, _ = get_detections(img_batch, model)
    recall_iter = 0.
    precision_iter = 0.

    for j, boxes in enumerate(picked_boxes):
        annot_boxes = annots[j]
        annot_boxes = annot_boxes[annot_boxes[:, 0] != -1]

        if boxes is None and annot_boxes.shape[0] == 0:
            continue
        elif boxes is None and annot_boxes.shape[0] != 0:
            recall_iter += 0.
            precision_iter += 1.
            continue
        elif boxes is not None and annot_boxes.shape[0] == 0:
            recall_iter += 1.
            precision_iter += 0.
            continue

        overlap = ops.boxes.box_iou(annot_boxes, boxes)

        # compute recall
        max_overlap, _ = torch.max(overlap, dim=1)
        mask = max_overlap > threshold
        detected_num = mask.sum().item()
        recall_iter += detected_num/annot_boxes.shape[0]

        # compute precision
        max_overlap, _ = torch.max(overlap, dim=0)
        mask = max_overlap > threshold
        true_positives = mask.sum().item()
        precision_iter += true_positives/boxes.shape[0]

    return recall_iter/len(picked_boxes), precision_iter/len(picked_boxes)
