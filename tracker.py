import numpy as np
import scipy.spatial.distance as dist


class CentroidTracker:
    def __init__(self, maxDisappearedframes=60, centroid_distance_threshold=150):
        """constructor

        Args:
            maxDisappearedframes (int, optional): [number of frames to keep the object id saved after it disappeared]. Defaults to 50.
            centroid_distance_threshold (int, optional): [euclidean distance between two objects for which it is same object]. Defaults to 10.
        """
        self.max_disappeared = maxDisappearedframes
        self.centroid_distance_threshold = centroid_distance_threshold
        self.objects_tracked = {}

        # key will be the object id, values will be list of all the centroids and number of frames for which it is disappeared
        self.count = 0

    def register(self, centroid):
        self.objects_tracked.update(
            {self.count: {"centroids": [centroid], "disappeared_count": 0}}
        )
        self.count += 1

    def deregister(self, objectid):
        self.disappeared_object[objectid] = self.objects_tracked[objectid]
        del self.objects_tracked[objectid]

    def update(self, centroids):
        self.disappeared_object = {}
        if len(centroids) == 0:
            for object_id in list(self.objects_tracked.keys()):
                self.objects_tracked[object_id]["disappeared_count"] += 1

                if (
                    self.objects_tracked[object_id]["disappeared_count"]
                    > self.max_disappeared
                ):
                    self.deregister(object_id)

            return self.objects_tracked, self.disappeared_object

        if len(self.objects_tracked) == 0:
            for idx, i in enumerate(centroids):
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
            if D[row, col] < self.centroid_distance_threshold:
                objectID = objectIDs[row]
                self.objects_tracked[objectID]["centroids"].append(centroids[col])
                self.objects_tracked[objectID]["disappeared_count"] = 0
                usedRows.add(row)
                usedCols.add(col)
            else:
                self.register(centroids[col])

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

    def euclidean_distances(self, pt1, pt2):
        x1, y1 = pt1
        x2, y2 = pt2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)