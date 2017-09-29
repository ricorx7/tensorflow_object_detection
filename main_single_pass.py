"""
Give a video file to detect objects.  This will process the video file and
create a new video file with the objects shown with a box around them.

Usage:
  main_single_pass.py (--video=<video_path>) [--output=<output>] [--model=<model>]

"""

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import glob

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

# This is needed to display the images.
#%matplotlib inline

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

from tqdm import tqdm
from moviepy.editor import *
from docopt import docopt


class main():
    """
    Give a video path.  This will detect the objects in the video
    and create a new video file.
    """

    def __init__(self, docopt_args):

        # Path of video to detect objects
        self.video_path = docopt_args["--video"]

        # Path to store results
        if docopt_args["--output"]:
            self.video_output_path = docopt_args["--output"]
        else:
            self.video_output_path = "../../../video_output/"

        # Path to model file
        if docopt_args["--model"]:
            self.model_path = docopt_args["--model"]
        else:
            self.model_path = "../../../ssd_inception_v2_coco_11_06_2017"

        # Run the app
        self.run()


    def run(self):
        # Print summary
        print("Model: " + self.model_path)
        print("Video: " + self.video_path)
        print("Output Dir: " + self.video_output_path)

        # What model to download.
        #MODEL_NAME = '../../../ssd_mobilenet_v1_coco_11_06_2017'                           # Fast
        #MODEL_NAME = '../../../faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017'      # Slow Best results
        #MODEL_NAME = '../../../ssd_inception_v2_coco_11_06_2017'                            # Fast
        MODEL_NAME = self.model_path
        MODEL_FILE = MODEL_NAME + '.tar.gz'
        DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

        # List of the strings that is used to add correct label for each box.
        PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

        NUM_CLASSES = 90

        #opener = urllib.request.URLopener()
        #opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
        tar_file = tarfile.open(MODEL_FILE)
        for file in tar_file.getmembers():
          file_name = os.path.basename(file.name)
          if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())


        detection_graph = tf.Graph()
        with detection_graph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')


        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        def load_image_into_numpy_array(image):
          (im_width, im_height) = image.size
          return np.array(image.getdata()).reshape(
              (im_height, im_width, 3)).astype(np.uint8)

        # Open the video file
        clip = VideoFileClip(self.video_path)

        # Size, in inches, of the output images.
        IMAGE_SIZE = (12, 8)

        with detection_graph.as_default():
          with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # Create an image index to keep the images in order
            # when converted back to a video
            img_idx = 0

            # Iterate through all frames in the video
            # use tqdm to give progress in the console
            # Use a frame rate of 30 fps
            for frame in tqdm(clip.iter_frames(fps=30)):

              # the array based representation of the image will be used later in order to prepare the
              # result image with boxes and labels on it.
              image_np = frame

              # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
              image_np_expanded = np.expand_dims(image_np, axis=0)

              # Actual detection.
              (boxes, scores, classes, num) = sess.run(
                  [detection_boxes, detection_scores, detection_classes, num_detections],
                  feed_dict={image_tensor: image_np_expanded})

              # Visualization of the results of a detection.
              vis_util.visualize_boxes_and_labels_on_image_array(
                  image_np,
                  np.squeeze(boxes),
                  np.squeeze(classes).astype(np.int32),
                  np.squeeze(scores),
                  category_index,
                  use_normalized_coordinates=True,
                  line_thickness=8)
              #plt.figure(figsize=IMAGE_SIZE)

              # Save the image to a folder
              im = Image.fromarray(image_np)
              im.save(os.path.join(self.video_output_path, (str(img_idx).zfill(7) + ".jpg")))
              img_idx += 1

        # Convert all the images in the folder to a video
        clip = ImageSequenceClip(self.video_output_path, fps=30)
        clip.to_videofile(os.path.join(self.video_path, "output.mp4"), fps=30)



if __name__ == '__main__':
    args = docopt(__doc__)
    main(args)
