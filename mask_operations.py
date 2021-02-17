from skimage import measure
from shapely.geometry import Polygon, MultiPolygon
from PIL import Image
import numpy as np

def create_sub_masks(mask_image):
    width, height = mask_image.size
    sub_masks = {}

    for x in range(width):
        for y in range(height):
            pixel = mask_image.getpixel((x, y))
            pixel_str = str(pixel)
            sub_mask = sub_masks.get(pixel_str)
            if sub_mask is None:
                sub_masks[pixel_str] = Image.new("1", (width + 2, height + 2))
            sub_masks[pixel_str].putpixel((x + 1, y + 1), 1)
    return sub_masks


def mask_to_polygon(sub_mask, category_id):

    contours = measure.find_contours(sub_mask, 0.5, positive_orientation="low")
    segmentations = []
    polygons = []
    for contour in contours:
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        if isinstance(poly, type(MultiPolygon)):
            print("worked")
            continue

        polygons.append(poly)
        try:
            segmentation = np.array(poly.exterior.coords).ravel().tolist()
        except:
            segmentation = []
        if len(segmentation) != 0:
            segmentations.append(segmentation)
    return segmentations
