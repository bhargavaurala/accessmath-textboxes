import numpy as np
import cv2
import skimage

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

from math import ceil
from random import shuffle

class CaffeTrainer(object):
    def __init__(self,
                 model_def='models/VGGNet/text/longer_conv_300x300/train.prototxt',
                 model_weights='examples/TextBoxes/TextBoxes_icdar13.caffemodel',
                 solver_def='models/VGGNet/text/longer_conv_300x300/solver.prototxt',
                 nepochs=100,
                 batch_size=32,
                 scale=(300, 300),
                 train_list='/home/buralako/dataset-txt/ICDAR13/trainval.txt'):
        self.net = caffe.Net(model_def,  # defines the structure of the model
                             model_weights,  # contains the trained weights
                             caffe.TRAIN)  # use test mode (e.g., don't perform dropout)
        self.solver = caffe.get_solver('models/VGGNet/text/longer_conv_300x300/solver.prototxt')
        self.nepochs = nepochs
        self.batch_size = batch_size
        self.scale = scale
        self.train_list = []        
        with open(self.train_list, 'r') as f:
            for line in f.readlines():
                if len(line) <= 0:
                    continue
                else:
                    self.train_list += [tuple(line.split(' '))]

    def __len__(self):
        return ceil(len(self.train_list) / float(self.batch_size))

    def __getitem__(self, index):
        start = index * batch_size
        end = (index + 1) * batch_size
        if end < len(self.train_list):
            this_batch = self.train_list[start : end]
        else:
            end = len(train_list)
            this_batch = self.train_list[start : end]
            remaining = 32 - end + start
            this_batch += self.train_list[: remaining]
            shuffle(train_list)
        return self.get_batch(this_batch)

    def get_batch(this_batch):
        batch_ims = []
        batch_bboxes = []
        for im_file, bboxes_file in this_batch:
            img = cv2.imread(im_file)
            batch_ims += [self.convertImageCaffe(img, self.scale)]
            batch_bboxes += [self.parseAnnotationTextBoxes(bboxes_file)]
        batch_ims = np.concatenate(batch_ims, axis=0)
        batch_bboxes = np.concatenate(batch_bboxes, axis=0).reshape(1, 1, -1, 8)
        return batch_ims, batch_bboxes

    @staticmethod
    def convertImageCaffe(raw_image, scale):
        input_img = skimage.img_as_float(cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)).astype(np.float32)
        image_height, image_width, channels = input_img.shape
        image_resize_height = scale[0]
        image_resize_width = scale[1]
        transformer = caffe.io.Transformer({'data': (1, 3, image_resize_height, image_resize_width)})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
        transformer.set_raw_scale('data',
                                  255)  # the reference model operates on images in [0,255] range instead of [0,1]
        transformer.set_channel_swap('data',
                                     (2, 1, 0))  # the reference model has channels in BGR order instead of RGB

        transformed_image = transformer.preprocess('data', input_img)
        return transformed_image

    @staticmethod
    def parseAnnotationTextBoxes(bboxes_file):
        pass

    def train(self):
        for i in range(self.nepochs):
            for b in range(len(self)):
                ims, bboxes = self[b]
                self.net.blobs['data'].data[...] = ims

if __name__ == '__main__':
    tr = CaffeTrainer()
    print tr.net.blobs['data'].data.shape

    
















