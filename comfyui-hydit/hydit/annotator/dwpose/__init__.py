# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import os
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
from . import util
from .wholebody import Wholebody


def draw_pose(pose, H, W, draw_body=True):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    if draw_body:
        canvas = util.draw_bodypose(canvas, candidate, subset)

    canvas = util.draw_handpose(canvas, hands)

    canvas = util.draw_facepose(canvas, faces)

    return canvas


def keypoint2bbox(keypoints):
    valid_keypoints = keypoints[keypoints[:, 0] >= 0]  # Ignore keypoints with confidence 0
    if len(valid_keypoints) == 0:
        return np.zeros(4)
    x_min, y_min = np.min(valid_keypoints, axis=0)
    x_max, y_max = np.max(valid_keypoints, axis=0)

    return np.array([x_min, y_min, x_max, y_max])

def expand_bboxes(bboxes, expansion_rate=0.5, image_shape=(0, 0)):
    expanded_bboxes = []
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = map(int, bbox)
        width = x_max - x_min
        height = y_max - y_min

        # 扩展宽度和高度
        new_width = width * (1 + expansion_rate)
        new_height = height * (1 + expansion_rate)

        # 计算新的边界框坐标
        x_min_new = max(0, x_min - (new_width - width) / 2)
        x_max_new = min(image_shape[1], x_max + (new_width - width) / 2)
        y_min_new = max(0, y_min - (new_height - height) / 2)
        y_max_new = min(image_shape[0], y_max + (new_height - height) / 2)

        expanded_bboxes.append([x_min_new, y_min_new, x_max_new, y_max_new])

    return expanded_bboxes

def create_mask(image_width, image_height, bboxs):
    mask = np.zeros((image_height, image_width), dtype=np.float32)
    for bbox in bboxs:
        x1, y1, x2, y2 = map(int, bbox)
        mask[y1:y2+1, x1:x2+1] = 1.0
    return mask

threshold = 0.4
class DWposeDetector:
    def __init__(self):

        self.pose_estimation = Wholebody()

    def __call__(self, oriImg, return_index=False, return_yolo=False, return_mask=False):
        oriImg = oriImg.copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, subset = self.pose_estimation(oriImg)
            candidate = np.zeros((1, 134, 2), dtype=np.float32) if candidate is None else candidate
            subset = np.zeros((1, 134), dtype=np.float32) if subset is None else subset
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            # import pdb; pdb.set_trace()
            if return_yolo:
                candidate[subset < threshold] = -0.1
                subset = np.expand_dims(subset >= threshold, axis=-1)
                keypoint = np.concatenate([candidate, subset], axis=-1)

                # return pose + hand
                return np.concatenate([keypoint[:, :18], keypoint[:, 92:]], axis=1)

            body = candidate[:, :18].copy()
            body = body.reshape(nums * 18, locs)
            score = subset[:, :18]
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > threshold:
                        score[i][j] = int(18 * i + j)
                    else:
                        score[i][j] = -1

            un_visible = subset < threshold
            candidate[un_visible] = -1

            foot = candidate[:, 18:24]

            faces = candidate[:, 24:92]

            hands1 = candidate[:, 92:113]
            hands2 = candidate[:, 113:]
            hands = np.vstack([hands1, hands2])

            # import pdb; pdb.set_trace()
            hands_ = hands[hands.max(axis=(1, 2)) > 0]
            if len(hands_) == 0:
                bbox = np.array([0, 0, 0, 0]).astype(int)
            else:
                hand_random = random.choice(hands_)
                bbox = (keypoint2bbox(hand_random) * H).astype(int)  # [0, 1] -> [h, w]



            bodies = dict(candidate=body, subset=score)
            pose = dict(bodies=bodies, hands=hands, faces=faces)

            if return_mask:
                bbox = [(keypoint2bbox(hand) * H).astype(int) for hand in hands_]
                # bbox = expand_bboxes(bbox, expansion_rate=0.5, image_shape=(H, W))
                mask = create_mask(W, H, bbox)
                return draw_pose(pose, H, W), mask

            if return_index:
                return pose
            else:
                return draw_pose(pose, H, W), bbox


