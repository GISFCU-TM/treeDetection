#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
#import tensorflow as tf
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import cv2

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = '/inference_graph/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '/training/labelmap.pbtxt'

#Number of classes
NUM_CLASSES = 1


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
  
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
        
  return output_dict


def group_output_dict(output_dict, intersection_percentage=0.5, union_percentage=0.9):
  for i in range(output_dict['num_detections']):
    for j in range(i+1,output_dict['num_detections']):
      area_A = output_dict['detection_masks'][i].sum()
      area_B = output_dict['detection_masks'][j].sum()

      intersection = ( output_dict['detection_masks'][i] & output_dict['detection_masks'][j] ).sum()
      union     = ( output_dict['detection_masks'][i] | output_dict['detection_masks'][j] ).sum()

      # intersection>=intersection_percentage AND union<=union_percentage
      if ((intersection/area_A>=intersection_percentage)|(intersection/area_B>=intersection_percentage)) & (union/(area_A+area_B)<=union_percentage): 
        output_dict['detection_masks'][i] = output_dict['detection_masks'][i] | output_dict['detection_masks'][j]
        output_dict['detection_masks'][j] = output_dict['detection_masks'][j] * 0

  return output_dict



def output_dict_remove2small(output_dict, area_percentage=0.015):
    img_size = output_dict['detection_masks'][0].shape[0] * output_dict['detection_masks'][0].shape[1]
    output_dict['area_percentage'] = output_dict['detection_scores']
    for i in range(output_dict['num_detections']):
        output_dict['area_percentage'][i] = output_dict['detection_masks'][i].sum()/img_size

    output_dict['area_percentage'] = output_dict['area_percentage'][:output_dict['num_detections']]
    output_dict['detection_boxes'] = output_dict['detection_boxes'][:output_dict['num_detections']]
    output_dict['detection_classes']= output_dict['detection_classes'][:output_dict['num_detections']]
    output_dict['detection_scores'] = output_dict['detection_scores'][:output_dict['num_detections']]

    mask = output_dict['area_percentage'] >= area_percentage
    output_dict['detection_boxes'] = output_dict['detection_boxes'][mask]
    output_dict['detection_classes'] = output_dict['detection_classes'][mask]
    output_dict['detection_masks'] = output_dict['detection_masks'][mask]
    output_dict['detection_scores'] = output_dict['detection_scores'][mask]
    output_dict['num_detections'] = len(output_dict['detection_scores'])

    return output_dict
  
  
def detect_forest(img_folder, filename):
    # load model
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        tf.disable_v2_behavior()
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(os.path.dirname(__file__)+PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    
    label_map = label_map_util.load_labelmap(os.path.dirname(__file__)+PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    
    # load image
    image_path = os.path.join(img_folder, filename)
    image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)

    # Only keep lager componet on each object detected result
    im_height, im_width = output_dict['detection_masks'][0].shape[0], output_dict['detection_masks'][0].shape[1]
    for i in range(output_dict['num_detections']):
        num_objects, labels = cv2.connectedComponents(output_dict['detection_masks'][i]*255)

        if num_objects > 1:
            _, counts = np.unique(labels, return_counts=True)
            max_count_label = np.argmax(counts[1:]) + 1
            labels = (labels==max_count_label).astype(int)
            binary_img = np.array(labels*255, dtype = np.uint8)
            contours, _ = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            xmin,ymin,w,h = cv2.boundingRect(contours[0])
            xmax,ymax = xmin+w,ymin+h
            output_dict['detection_masks'][i] = labels
            output_dict['detection_boxes'][i] = np.array([ymin/im_height, xmin/im_width, ymax/im_height, xmax/im_width])

    output_dict = group_output_dict(output_dict)
    output_dict = output_dict_remove2small(output_dict, 0.01)


    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        min_score_thresh=0,
        skip_scores=True,
        line_thickness=0)
    
    del output_dict['detection_boxes']
    del output_dict['detection_scores']
    del output_dict['detection_classes']
    del output_dict['area_percentage']
    #return(image_np, output_dict)  
 

    #image_result, output_dict = detect_forest(args.folder, args.name)    
    #print(output_dict)
    cv2.imwrite(os.path.join(img_folder, filename)+'_detect.jpg', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    return 0