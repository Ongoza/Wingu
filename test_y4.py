import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from tf2_yolov4.anchors import YOLOV4_ANCHORS
from tf2_yolov4.model import YOLOv4

import os
# tf.debugging.set_log_device_placement(True)

path_root = os.path.dirname(os.path.abspath(__file__))
#print(path_root)

HEIGHT, WIDTH = (416, 416)

devices = tf.config.experimental.list_physical_devices()
#if gpus:
#    # Restrict TensorFlow to only use the first GPU
#    try:
#        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
#    except RuntimeError as e:
#        # Visible devices must be set at program startup
#        print(e)
with tf.device('/GPU:1'):
    image1 = tf.io.read_file(path_root + "/video/cars.jpg")
    image1 = tf.image.decode_image(image1)
    print("img", type(image1))
    image = tf.image.resize(image1, (HEIGHT, WIDTH))
    print(image)
    #images = tf.convert_to_tensor(images, dtype=tf.float32)

    gpus = tf.config.experimental.list_logical_devices('GPU')
    print("gpus:", gpus)
    if not gpus:
        print("\n\n!!!!!======= any GPUs don't use ========!!!!!!!!!!!!!!!!!!\n\n")
        images = tf.expand_dims(image, axis=0) / 255.0
        print('images.shape=', images.shape)
        model = YOLOv4(
            input_shape=(HEIGHT, WIDTH, 3),
            anchors=YOLOV4_ANCHORS,
            num_classes=80,
            training=False,
            yolo_max_boxes=100,
            yolo_iou_threshold=0.5,
            yolo_score_threshold=0.5,
        )

        model.load_weights(path_root+"/models/yolov4.h5")
        #model.summary()
    else:
        with tf.device(device):
            images = tf.expand_dims(image, axis=0) / 255.0
            print('images.shape=', images.shape)
            model = YOLOv4(
                input_shape=(HEIGHT, WIDTH, 3),
                anchors=YOLOV4_ANCHORS,
                num_classes=80,
                training=False,
                yolo_max_boxes=100,
                yolo_iou_threshold=0.5,
                yolo_score_threshold=0.5,
            )

            model.load_weights(path_root+"/models/yolov4.h5")

    boxes, scores, classes, valid_detections = model.predict(images)
    print("!!!!===================================")
    #print("result=", boxes, scores, classes, valid_detections)
    # COCO classes
    CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table',
        'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]

    # colors for visualization
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
              [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


    # %config InlineBackend.figure_format = 'retina'

    def plot_results(pil_img, boxes, scores, classes):
        plt.figure(figsize=(16,10))
        plt.imshow(pil_img)
        ax = plt.gca()
        print("d=" , boxes, scores, classes)
        for (xmin, ymin, xmax, ymax), score, cl in zip(boxes.tolist(), scores.tolist(), classes.tolist()):
            print("d2=", score, cl, xmin, ymin, xmax, ymax)
            if score > 0:
              ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=COLORS[cl % 6], linewidth=3))
              text = f'{CLASSES[cl]}: {score:0.2f}'
              ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
        plt.axis('off')
        plt.savefig(path_root+'/video/car_y4.jpg')
        #plt.show()

    plot_results(    images[0], boxes[0] * [WIDTH, HEIGHT, WIDTH, HEIGHT], scores[0],  classes[0].astype(int),)
    print(images[0][0][0])
    # plt.savefig('video/car_y4.jpg')
    # arr = images[0].numpy()
    # print('arr=', arr.shape, arr.dtype)
    # (Image.fromarray(np.uint8(arr))).save("video/car_y4.jpg", "JPEG")

