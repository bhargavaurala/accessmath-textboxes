import os
from copy import deepcopy
import cv2
import numpy as np
import caffe

from nms import nms

caffe_root = '/home/buralako/git/TextBoxes'
caffe.set_device(0)
caffe.set_mode_gpu()

scales = ((300, 300), (700, 700), (700, 500), (700, 300), (1600, 1600))

class TextDetection(object):
    def __init__(self,
                 model_def=os.path.join(caffe_root, 'examples/TextBoxes/deploy.prototxt'),
                 model_weights=os.path.join(caffe_root, 'examples/TextBoxes/TextBoxes_icdar13.caffemodel'),
                 detection_threshold=0.6,
                 nms_threshold=0.3,
                 format='xyxy',
                 visualize=False):
        self.net = caffe.Net(model_def,  # defines the structure of the model
                        model_weights,  # contains the trained weights
                        caffe.TEST)  # use test mode (e.g., don't perform dropout)
        print('Initialization complete - text detection model loaded')
        self.text_bbox = {}
        self.detection_threshold = detection_threshold
        self.nms_threshold = nms_threshold
        self.visualize = visualize
        self.visualize_dir = 'text_detection_debug({})'.format(self.detection_threshold)
        if not os.path.isdir(self.visualize_dir):
            os.makedirs(self.visualize_dir)
        self.bbox_format = 'xyxy' if format not in ['xyxy', 'xywh'] else format

    def initialize(self, width, height):
        pass

    def handleFrame(self, frame, last_frame, v_index, abs_time, rel_time, abs_frame_idx):
        image = frame
        frameID = abs_frame_idx

        img = deepcopy(image).astype(np.float32)
        image_height, image_width, channels = img.shape
        dt_results = []
        for scale in scales:
            # print(scale)
            image_resize_height = scale[0]
            image_resize_width = scale[1]
            transformer = caffe.io.Transformer({'data': (1, 3, image_resize_height, image_resize_width)})
            transformer.set_transpose('data', (2, 0, 1))
            transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
            transformer.set_raw_scale('data',
                                      255)  # the reference model operates on images in [0,255] range instead of [0,1]
            transformer.set_channel_swap('data',
                                         (2, 1, 0))  # the reference model has channels in BGR order instead of RGB

            self.net.blobs['data'].reshape(1, 3, image_resize_height, image_resize_width)
            transformed_image = transformer.preprocess('data', img)
            self.net.blobs['data'].data[...] = transformed_image
            # Forward pass.
            dets = self.net.forward()['detection_out']
            # Parse the outputs.
            det_label = dets[0, 0, :, 1]
            det_conf = dets[0, 0, :, 2]
            det_xmin = dets[0, 0, :, 3]
            det_ymin = dets[0, 0, :, 4]
            det_xmax = dets[0, 0, :, 5]
            det_ymax = dets[0, 0, :, 6]
            top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.detection_threshold]
            top_conf = det_conf[top_indices]
            top_xmin = det_xmin[top_indices]
            top_ymin = det_ymin[top_indices]
            top_xmax = det_xmax[top_indices]
            top_ymax = det_ymax[top_indices]
            for i in xrange(top_conf.shape[0]):
                xmin = int(round(top_xmin[i] * img.shape[1]))
                ymin = int(round(top_ymin[i] * img.shape[0]))
                xmax = int(round(top_xmax[i] * img.shape[1]))
                ymax = int(round(top_ymax[i] * img.shape[0]))
                xmin = max(1, xmin)
                ymin = max(1, ymin)
                xmax = min(img.shape[1] - 1, xmax)
                ymax = min(img.shape[0] - 1, ymax)
                score = top_conf[i]
                dt_result = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax, score]
                dt_results.append(dt_result)
        dt_results = sorted(dt_results, key=lambda x: -float(x[8]))
        nms_flag = nms(dt_results, 0.3)
        detections = []
        for k, dt in enumerate(dt_results):
            if nms_flag[k]:
                name = '%.2f' % (dt[8])
                xmin = dt[0]
                ymin = dt[1]
                xmax = dt[2]
                ymax = dt[5]
                detections.append((xmin, ymin, xmax, ymax))
        self.text_bbox[frameID] = {}
        self.text_bbox[frameID]['abs_time'] = abs_time
        self.text_bbox[frameID]['bboxes'] = []
        self.text_bbox[frameID]['visible'] = False
        for i in range(len(detections)):
            pts = detections[i]
            if self.visualize:
                cv2.rectangle(image, (pts[0], pts[1]), (pts[2], pts[3]), color=(0, 0, 255))
            if self.bbox_format == 'xywh':
                coords = (pts[0], pts[1], pts[2] - pts[0], pts[3] - pts[1])
            else:
                coords = (pts[0], pts[1], pts[2], pts[3])
            self.text_bbox[frameID]['visible'] = True
            self.text_bbox[frameID]['bboxes'].append(coords)
        if self.visualize and len(detections) > 0:
            cv2.imwrite('{}/{}.jpg'.format(self.visualize_dir, frameID), img)
            print(frameID, detections[0], len(detections), len(dt_results), self.getArea(detections[0], 'xyxy'))

    def getWorkName(self):
        return "Text Detector Caffe"

    def finalize(self):
        pass

    @staticmethod
    def getCentroid(bbox, bbox_format):
        if bbox_format == 'xyxy':
            x, y, x2, y2 = bbox
            w = x2 - x
            h = y2 - y
        else:
            x, y, w, h = bbox
        return (x + (w // 2), y + (h // 2))

    @staticmethod
    def getArea(bbox, bbox_format):
        if bbox_format == 'xyxy':
            x, y, x2, y2 = bbox
            w = x2 - x
            h = y2 - y
        else:
            x, y, w, h = bbox
        return w * h