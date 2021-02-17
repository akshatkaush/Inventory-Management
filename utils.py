import cv2
import numpy as np
from torchvision import transforms
import torch


def frame_extract(path):
    vidObj = cv2.VideoCapture(path, 0)
    success = 1
    while success:
        success, image = vidObj.read()
        if success:
            yield image


def ymean(centroids):
    return np.mean([x[1] for x in centroids[:-1]])


def y_max(unique_ids: list):
    ys = []
    for (_, y) in unique_ids:
        ys.append(y)
    return max(ys)


def y_min(unique_ids: list):
    ys = []
    for (_, y) in unique_ids:
        ys.append(y)
    return min(ys)


def get_centroid(vertexes):
    _x_list = [vertex[0] for vertex in vertexes]
    _y_list = [vertex[1] for vertex in vertexes]
    _len = len(vertexes)
    _x = sum(_x_list) / _len
    _y = sum(_y_list) / _len
    return (int(_x), int(_y))


def get_count_from_dict(counted_object_ids):
    count_loaded = 0
    count_unloaded = 0
    for k, v in counted_object_ids.items():
        if v == "load":
            count_loaded += 1
        else:
            count_unloaded += 1
    return count_loaded, count_unloaded


class Config:
    # backbone
    pretrained = True
    freeze_stage_1 = True
    freeze_bn = True

    # fpn
    fpn_out_channels = 64
    use_p5 = False

    # head
    class_num = 2
    use_GN_head = True
    prior = 0.01
    add_centerness = True
    cnt_on_reg = False

    # training
    strides = [8, 16, 32, 64, 128]
    limit_range = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 999999]]

    # inference
    score_threshold = 0.3
    nms_iou_threshold = 0.2
    max_detection_boxes_num = 150


def count_object(disappeared_object, counted_object_ids, y_pos):
    for k, v in disappeared_object.items():

        direction = v["centroids"][-1][1] - ymean(v["centroids"])

        if (
            direction > 0
            and v["centroids"][-1][1] > y_pos
            and y_min(v["centroids"]) < y_pos
        ):
            counted_object_ids.update({k: "unload"})

        if (
            direction < 0
            and v["centroids"][-1][1] < y_pos
            and y_max(v["centroids"]) > y_pos
        ):
            counted_object_ids.update({k: "load"})

    return counted_object_ids


transformation = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])


def normalize(image):
    return transforms.Normalize(
        mean=[0.45086395, 0.45227149, 0.43730811],
        std=[0.16381127, 0.16430189, 0.1590376],
    )(transformation(image))


def preprocess_image(image):
    image = normalize(image)
    return torch.unsqueeze(image, dim=0)
