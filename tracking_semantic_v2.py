import os
import cv2
import argparse
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import time
from PIL import Image
from tracker import CentroidTracker
from semantic_segmentation_pipeline.config import get_cfg_defaults
from semantic_segmentation_pipeline.models.model import create_model
from utils import (
    frame_extract,
    transformation,
    preprocess_image,
    count_object,
    Config,
    get_count_from_dict,
)
from skimage.morphology import convex_hull_image
from skimage.measure import regionprops, label
from mask_operations import mask_to_polygon
from tqdm import tqdm


def detect_object(model, image):
    bboxes = []
    with torch.no_grad():
        out = model(image.cuda())
        scores, classes, boxes = out
        boxes = boxes[0].cpu().numpy().tolist()
        classes = classes[0].cpu().numpy().tolist()
        scores = scores[0].cpu().numpy().tolist()
    for i, bbox in enumerate(boxes):
        bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3], scores[i]])
    return bboxes


def merge_box(box1, box2):
    return [
        min(box1[0], box2[0]),
        min(box1[1], box2[1]),
        max(box1[2], box2[2]),
        max(box1[3], box1[3]),
        1,
    ]


def euclidean_distances(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def get_centroid(bbox):
    return (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))


def area_bbox(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def iou_bbox(bbox1, bbox2):
    xleft = max(bbox1[0], bbox2[0])
    yleft = max(bbox1[1], bbox2[1])
    xright = min(bbox1[2], bbox2[2])
    yright = min(bbox1[3], bbox2[3])
    if xright < xleft or yright < yleft:
        return 0
    intersection_area = area_bbox([xleft, yleft, xright, yright])
    return intersection_area / (area_bbox(bbox1) + area_bbox(bbox2) - intersection_area)


def join_boxes_iou(boxes, iou_match_threshold=0.3):
    iou_matrix = np.zeros((len(boxes), len(boxes)))
    for i, b1 in enumerate(boxes):
        for j, b2 in enumerate(boxes):
            if i == j:
                iou_matrix[i, j] = 0
                continue
            iou_matrix[i, j] = iou_bbox(b1, b2)
    max_indices = iou_matrix.argmax(axis=1)
    mergable = {}
    desired_boxes = []
    # print(iou_matrix)
    for idx, i in enumerate(max_indices):
        if iou_matrix[idx, i] > iou_match_threshold:
            if i not in list(mergable.keys()):
                desired_boxes.append(merge_box(boxes[idx], boxes[i]))
                mergable.update({idx: i})
                mergable.update({i: idx})

    for idx in range(len(boxes)):
        if idx not in mergable.keys():
            desired_boxes.append(boxes[idx])
    return desired_boxes


def create_mask(model, image, bbox_buffer=2):
    prediction = model(image.cuda(), (image.shape[2], image.shape[3]))
    prediction = (
        torch.argmax(prediction["output"][0], dim=1)
        .detach()
        .cpu()
        .squeeze(0)
        .numpy()
        .astype(np.uint8)
        * 255
    )

    frame1 = label(prediction)
    bboxes = regionprops(frame1)

    boxes = []

    for bbox in bboxes:
        area = bbox.area
        if area < 3000:
            continue

        bbox = bbox.bbox
        c = (bbox[1] + bbox[3]) // 2, (bbox[0] + bbox[2]) // 2
        w = abs(bbox[3] - bbox[1]) * bbox_buffer
        h = abs(bbox[2] - bbox[0]) * bbox_buffer
        x1 = int(max(0, c[0] - w // 2))
        y1 = int(max(0, c[1] - h // 2))
        x2 = int(min(prediction.shape[1], c[0] + w // 2))
        y2 = int(min(prediction.shape[0], c[1] + h // 2))
        p = prediction[y1:y2, x1:x2]
        p = convex_hull_image(p)
        frame1 = label(p)
        b = regionprops(frame1)

        if len(b) > 0:
            boxes.append(
                [
                    b[0].bbox[1] + x1,
                    b[0].bbox[0] + y1,
                    b[0].bbox[3] + x1,
                    b[0].bbox[2] + y1,
                ]
            )
    if len(boxes) != 0:
        boxes = join_boxes_iou(boxes, 0.2)
    all_polygons = []
    segmentation = mask_to_polygon(prediction, 1)
    for i in segmentation:
        poly = []
        for idx, _ in enumerate(i):
            if idx % 2 == 0:
                poly.append([abs(int(i[idx])), abs(int(i[idx + 1]))])
        all_polygons.append(poly)
    return all_polygons, prediction, boxes


def detect_vehicle(model, image):
    prediction = model(image.cuda(), (image.shape[2], image.shape[3]))
    prediction = (
        torch.argmax(prediction["output"][0], dim=1)
        .detach()
        .cpu()
        .squeeze(0)
        .numpy()
        .astype(np.uint8)
        * 255
    )
    plt.imshow(prediction)
    plt.show()
    frame1 = label(prediction)
    bboxes = regionprops(frame1)
    x_coordinates = []
    for bbox in bboxes:
        x_coordinates.extend(
            [int(bbox.bbox[1] * 2.5) + 400, int(bbox.bbox[3] * 2.5) + 400]
        )
    return sorted(x_coordinates)


# we can write algorithm based on trajectory maybe create some lstm kind of thing which takes all the lstm inputs and jugde if this object should be counted or not. But still complexity can be increased rather than just if the object crosses center


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "-p",
        "--path_to_video",
        required=True,
        help="path to video on which we are running",
    )
    ap.add_argument(
        "-c",
        "--path_to_config",
        required=True,
        help="path to config file for semantic segmentation model",
    )

    ap.add_argument(
        "-w", "--path_to_weights", required=True, help="path to weights of model"
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="",
    )
    args = ap.parse_args()

    cfg = get_cfg_defaults()
    cfg.defrost()
    cfg.merge_from_file(args.path_to_config)

    model_cylinder = create_model(cfg).cuda()
    model_cylinder.load_state_dict(
        torch.load(
            args.path_to_weights,
            map_location=torch.device("cpu"),
        )["state_dict"]
    )
    model_cylinder.eval()

    cfg1 = get_cfg_defaults()
    cfg1.defrost()
    cfg1.merge_from_file(
        "/media/sanchit/Workspace/idealai/Services_created/gas_project/semantic_segmentation_pipeline/config/vehicle_detection.yaml"
    )
    model_vehicle = create_model(cfg1).cuda()
    model_vehicle.load_state_dict(
        torch.load(
            "/media/sanchit/checkpoints/Vehicle_detect/hrnet18v1-c1-ohem/best.pth",
            map_location=torch.device("cpu"),
        )["state_dict"]
    )
    model_vehicle.eval()

    tracker = CentroidTracker()
    counted_object_ids = {}
    flag = 0
    count = 0

    for idx, frame in tqdm(enumerate(frame_extract(args.path_to_video))):

        # if idx % 120 == 0:
        #     vehicle_coordinates = detect_vehicle(
        #         model_vehicle,
        #         preprocess_image(cv2.resize(frame[200:600, 400:1900], (600, 160))),
        #     )
        vehicle_coordinates = [1000, 1800]
        vehicle_images = []
        for vehicle_id in range(len(vehicle_coordinates)):
            if vehicle_id % 2 == 0:
                vehicle_images.append(
                    frame[
                        650:1000,
                        vehicle_coordinates[vehicle_id] : vehicle_coordinates[
                            vehicle_id + 1
                        ],
                    ]
                )

        # add support for multiple vehicle
        vehicle_image = vehicle_images[-1]  # need to remove this and add a loop

        if count > 20:
            if idx % 2 == 0:
                continue

        if flag == 0:
            H, W, _ = vehicle_image.shape
            ypos = vehicle_image.shape[0] // 2
            flag = 1

        image = preprocess_image(vehicle_image)
        all_polygons, _, bboxes = create_mask(model_cylinder, image)

        centroids = tracker.get_centroid(bboxes)
        objects, disappeared_object = tracker.update(centroids)

        if len(list(disappeared_object.keys())) > 0:
            counted_object_ids = count_object(
                disappeared_object, counted_object_ids, ypos
            )

        if args.debug:
            for polygon in all_polygons:
                cv2.fillPoly(
                    vehicle_image,
                    [np.asarray(polygon).reshape(-1, 1, 2)],
                    (255, 0, 255),
                )
            for bbox in bboxes:
                cv2.rectangle(
                    vehicle_image,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    (0, 255, 0),
                    2,
                )

            cv2.line(vehicle_image, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
            for k, v in objects.items():
                centroid = v["centroids"][-1]
                cv2.putText(
                    vehicle_image,
                    str(k),
                    (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
                cv2.circle(
                    vehicle_image, (centroid[0], centroid[1]), 4, (0, 255, 0), -1
                )
                try:
                    centroid1 = v["centroids"][-2]
                    cv2.circle(
                        vehicle_image, (centroid1[0], centroid1[1]), 4, (255, 0, 0), -1
                    )
                except:
                    pass
            cv2.imshow("video", vehicle_image)
            key = cv2.waitKey(1) & 0xFF
            print(counted_object_ids)
            #            time.sleep(1 / 10)
            if key == ord("q"):
                break
    count_loaded, count_unloaded = get_count_from_dict(counted_object_ids)
    print("number of cylinders loaded", count_loaded)
    print("number of cylinders unloaded", count_unloaded)


if __name__ == "__main__":
    main()
