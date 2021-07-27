import argparse
from utils import util

def arg_parse():

    parser=argparse.ArgumentParser()
    parser.add_argument('-g','--gpu',type=str,default='0',help='Use which gpu?')
    parser.add_argument('-d','--dataset',type=str,help='Train on which dataset')
    parser.add_argument('--dataset_folder','--image_dataset_path',type=str,help='Dataset Fodlder Path')
    parser.add_argument("--graph_path",'--frozen_graph',type=str,help='The path of object detection,frozen graph is used')
    parser.add_argument('--box_imgs_npy_path',type=str,help='Path for npy file that store the \(box,img_path\)') 
    args=parser.parse_args()
    return args

import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


PATH_TO_MODEL_DIR = "D:\Paper\efficientdet_d1_coco17_tpu-32"

PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "\saved_model"

LABEL_FILENAME = 'mscoco_label_map.pbtxt'

PATH_TO_LABELS="D:\\Paper\\data\\"+LABEL_FILENAME

####Image Paths####
base_url = "D:\\Paper\\dataset\\avenue\\training\\frames\\02\\"

filenames = '1421.jpg'

IMAGE_PATHS = [base_url+filenames]


####----------------------####

file_path = IMAGE_PATHS[0]
print(os.path.isfile(file_path))



####Testing model loads####

print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

####----------------------####

####Loading labels####

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

####----------------------####

####Testing it on images####

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    
    return np.array(Image.open(path))


for image_path in IMAGE_PATHS:

    print('Running inference for {}... '.format(image_path), end='')

    #image_np = load_image_into_numpy_array(image_path)
    image=util.data_preprocessing(image_path,target_size=640)
    #image=np.expand_dims(image,axis=0)

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.


    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)

    plt.figure()
    plt.imshow(image_np_with_detections)
    result = Image.fromarray(image_np_with_detections)
    result.save("det.jpg")
    
    print('Done')

plt.show()

# sphinx_gallery_thumbnail_number = 2

####inference_per_image####
def run_inference_for_images_per_image(image_folder,np_boxes_path,score_threshold):
  frame_lists=util.get_frames_paths(image_folder,gap=2)
  print(frame_lists)
  path_box_lists=[]

  for image_path in frame_lists:
    
    print('Running inference for {}... '.format(image_path), end='')
    #image_np = load_image_into_numpy_array(image_path)
    
    image=util.data_preprocessing(image_path,target_size=640)
    #image=np.expand_dims(image,axis=0)

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.


    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)

    #plt.figure()
    #plt.imshow(image_np_with_detections)
    #result = Image.fromarray(image_np_with_detections)
    #result.save("det.jpg")
      
    #print('Done')

    #plt.show()
 
    for score,box,_class in zip(detections['detection_scores'],detections['detection_boxes'],detections['detection_classes']):
      #print(score)
      if score>=score_threshold:
        print(score)
        path_box_lists.append([image_path,box[0],box[1],box[2],box[3],_class])
    
    #print(image_path)


  np.save(np_boxes_path,path_box_lists)

  print('finish boxes detection!')


####----------------------####

#### Main ####
if __name__=='__main__':
    
    args=arg_parse()

    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    
    np_paths_boxes_path = args.box_imgs_npy_path

    
    #run_inference_for_images_per_image(args.dataset_folder,np_paths_boxes_path,0.5)

####----------------------####