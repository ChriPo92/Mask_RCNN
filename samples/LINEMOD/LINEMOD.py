import numpy as np
import cv2
import sys
import os

ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)
from mrcnn import utils
from mrcnn.config import Config

########################################################################################################################
#                                                       Utility                                                        #
########################################################################################################################

camera_calibration_matrix = np.array([[572.41140, 0, 325.26110], [0, 573.57043, 242.04899], [0, 0, 1]])
camera_projection_matrix = np.array([[572.41140, 0, 325.26110, 0], [0, 573.57043, 242.04899, 0], [0, 0, 1, 0]])


def linemod_dpt(path):
    """read a depth image"""
    dpt = open(path, "rb")
    rows = np.frombuffer(dpt.read(4), dtype=np.int32)[0]
    cols = np.frombuffer(dpt.read(4), dtype=np.int32)[0]

    return np.fromfile(dpt, dtype=np.uint16).reshape((rows, cols))


def linemod_pose(path, i):
    """
    read a 3x3 rotation and 3x1 translation.

    can be done with np.loadtxt, but this is way faster
    """
    R = open("{}/data/rot{}.rot".format(path, i))
    R.readline()
    R = np.float32(R.read().split()).reshape((3, 3))

    t = open("{}/data/tra{}.tra".format(path, i))
    t.readline()
    t = np.float32(t.read().split())

    return R, t


def ply_vtx(path):
    """
    read all vertices from a ply file
    """
    f = open(path)
    assert f.readline().strip() == "ply"
    f.readline()

    N = int(f.readline().split()[-1])

    while f.readline().strip() != "end_header":
        continue

    pts = []

    for _ in range(N):
        pts.append(np.float32(f.readline().split()[:3]))

    return np.array(pts)


def linemod_point_cloud(path):
    "returns a PC in meters"
    pc = []

    def str_to_list_of_floats(line):
        m = map(float, line[:-2].split(" "))
        return list(m)

    with open(path) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            l = str_to_list_of_floats(line)
            pc.append(l[:3])
    return np.array(pc)


########################################################################################################################
#                                                   Configurations                                                     #
########################################################################################################################


class BalloonConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "balloon"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


########################################################################################################################
#                                                        Dataset                                                       #
########################################################################################################################

class LINEMOD_Dataset(utils.Dataset):
    def load_linemod(self, dataset_dir, subset, class_ids=None):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        class_ids: If provided, only loads images that have the given classes
        """
        # Add classes.
        classes = {1: "ape", 2: "benchviseblue", 3: "bowl", 4: "cam", 5: "can", 6: "cat", 7: "cup", 8: "driller",
                   9: "duck", 10: "eggbox", 11: "glue", 12: "holepuncher", 13: "iron", 14: "lamp", 15: "phone"}
        if class_ids is None:
            class_ids = np.arange(1, 16, 1)
        for i in class_ids:
            self.add_class("linemod", i, classes[i])

        # Train or validation dataset?
        # TODO: implement consistent splitting of the datasets
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "balloon",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "balloon":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "balloon":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
