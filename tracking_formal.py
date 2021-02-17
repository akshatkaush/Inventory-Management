import os
import cv2
import torch
from torchvision import transforms
import argparse
from model.fcos import FCOSDetector
import scipy.spatial.distance as dist
import numpy as np


def frame_extract(path):
    vidObj = cv2.VideoCapture(path, 0)
    success = 1
    while success:
        success, image = vidObj.read()
        if success:
            yield image


transformation = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])


def normalize(image):
    return transforms.Normalize(
        mean=[0.45086395, 0.45227149, 0.43730811],
        std=[0.16381127, 0.16430189, 0.1590376],
    )(transformation(image))


def preprocess_image(image):
    image = normalize(image)
    return torch.unsqueeze(image, dim=0)


def detect_object(model, image):
    bboxes = []
    image = preprocess_image(image)
    with torch.no_grad():
        out = model(image.cuda())
        scores, classes, boxes = out
        boxes = boxes[0].cpu().numpy().tolist()
        classes = classes[0].cpu().numpy().tolist()
        scores = scores[0].cpu().numpy().tolist()
    for i, bbox in enumerate(boxes):
        bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3], scores[i]])
    return bboxes


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


class CentroidTracker:
    def __init__(self, maxDisappearedframes=20, centroid_distance_threshold=10):
        """constructor

        Args:
            maxDisappearedframes (int, optional): [number of frames to keep the object id saved after it disappeared]. Defaults to 50.
            centroid_distance_threshold (int, optional): [euclidean distance between two objects for which it is same object]. Defaults to 10.
        """
        self.max_disappeared = maxDisappearedframes
        self.centroid_distance_threshold = centroid_distance_threshold
        self.objects_tracked = (
            {}
        )  # key will be the object id, values will be list of all the centroids and number of frames for which it is disappeared
        self.count = 0

    def register(self, centroid):
        self.objects_tracked.update(
            {self.count: {"centroids": [centroid], "disappeared_count": 0}}
        )
        self.count += 1

    def deregister(self, objectid):
        self.disappeared_object[objectid] = self.objects_tracked[objectid]
        del self.objects_tracked[objectid]

    def update(self, boxes):
        self.disappeared_object = {}
        centroids = self.get_centroid(boxes)
        if len(boxes) == 0:
            for object_id in list(self.objects_tracked.keys()):
                self.objects_tracked[object_id]["disappeared_count"] += 1

                if (
                    self.objects_tracked[object_id]["disappeared_count"]
                    > self.max_disappeared
                ):
                    self.deregister(object_id)

            return self.objects_tracked, self.disappeared_object

        if len(self.objects_tracked) == 0:
            for i in centroids:
                self.register(i)
            return self.objects_tracked, self.disappeared_object

        objectIDs = list(self.objects_tracked.keys())
        objectCentroids = [x["centroids"][-1] for _, x in self.objects_tracked.items()]
        D = dist.cdist(np.array(objectCentroids), np.array(centroids))
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        usedRows = set()
        usedCols = set()

        for (row, col) in zip(rows, cols):

            if row in usedRows or col in usedCols:
                continue

            objectID = objectIDs[row]

            self.objects_tracked[objectID]["centroids"].append(centroids[col])
            self.objects_tracked[objectID]["disappeared_count"] = 0
            usedRows.add(row)
            usedCols.add(col)

        # compute both the row and column index we have NOT yet
        # examined
        unusedRows = set(range(0, D.shape[0])).difference(usedRows)
        unusedCols = set(range(0, D.shape[1])).difference(usedCols)

        if D.shape[0] >= D.shape[1]:

            for row in unusedRows:

                objectID = objectIDs[row]
                self.objects_tracked[objectID]["disappeared_count"] += 1
                if (
                    self.objects_tracked[objectID]["disappeared_count"]
                    > self.max_disappeared
                ):
                    self.deregister(objectID)

        else:
            for col in unusedCols:
                self.register(centroids[col])
        return self.objects_tracked, self.disappeared_object

    def get_centroid(self, boxes):
        return [(int((x[0] + x[2]) / 2), int((x[1] + x[3]) / 2)) for x in boxes]


# we can write algorithm based on trajectory maybe create some lstm kind of thing which takes all the lstm inputs and jugde if this object should be counted or not. But still complexity can be increased rather than just if the object crosses center
def ymean(centroids):
    return np.mean([x[1] for x in centroids[:-1]])


def count_object(disappeared_object, counted_object_ids, y_pos):
    for k, v in disappeared_object.items():

        direction = v["centroids"][-1][1] - ymean(v["centroids"])

        if direction > 0 and v["centroids"][-1][1] > y_pos:
            counted_object_ids.update({k: "unload"})

        if direction < 0 and v["centroids"][-1][1] < y_pos:
            counted_object_ids.update({k: "load"})

    return counted_object_ids


# centroid tracking not working properly
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "-p",
        "--path_to_video",
        required=True,
        help="path to Caffe 'deploy' prototxt file",
    )
    ap.add_argument(
        "-w", "--path_to_weights", required=True, help="path to Caffe pre-trained model"
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="",
    )
    args = ap.parse_args()

    # add vehicle and plate detector

    model = FCOSDetector(mode="inference", config=Config)
    model.load_state_dict(
        torch.load(
            args.path_to_weights,
            map_location=torch.device("cpu"),
        )
    )
    model.eval().cuda()

    tracker = CentroidTracker()
    counted_object_ids = {}
    flag = 0
    for frame in frame_extract(args.path_to_video):
        frame = frame[500:820, 580:1600]
        if flag == 0:
            H, W, _ = frame.shape
            ypos = frame.shape[0] // 2
            flag = 1
        bboxes = detect_object(model, frame)
        objects, disappeared_object = tracker.update(bboxes)

        if len(list(disappeared_object.keys())) > 0:
            counted_object_ids = count_object(
                disappeared_object, counted_object_ids, ypos
            )
        if args.debug:
            cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
            for k, v in objects.items():
                centroid = v["centroids"][-1]
                cv2.putText(
                    frame,
                    str(k),
                    (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            cv2.imshow("video", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            print(counted_object_ids)


if __name__ == "__main__":
    main()