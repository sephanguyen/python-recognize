# UltraFace does not have landmark detection

import cv2
import torch
from PIL import Image
from ultraface_package.vision.ssd.config.fd_config import define_img_size

input_img_size = 320 # 128/160/320/480/640/1280
define_img_size(input_img_size)  # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'

from ultraface_package.vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor

#from ultraface_pytorch.vision.utils.misc import Timer
#import sys

class Ultraface_detect():
    def __init__(self, model_path='models/pretrained/version-RFB-320.pth', device='cpu'):
        self.candidate_size = 1000
        self.threshold = 0.7
        self.det_device = device #cuda:0 or cpu
        num_classes = 2 # 2 class --> BACKGROUND or face
        #model_path  = "models/pretrained/version-RFB-320.pth"
        self.net = create_Mb_Tiny_RFB_fd(num_classes, is_test=True, device=self.det_device)
        self.predictor = create_Mb_Tiny_RFB_fd_predictor(self.net, candidate_size=self.candidate_size, device=self.det_device)
        self.net.load(model_path)
    
    def detect_face(self, img):
        # img is cv2 image
        # Output should be PIL images
        boxes, labels, probs = self.predictor.predict(img, self.candidate_size / 2, self.threshold)
        if torch.is_tensor(boxes):
            boxes = boxes.numpy()
        detected_faces = []
        for i in range(boxes.shape[0]):
            box = boxes[i,:]
            box = [int(x) for x in box]
            prob = probs[i] # Indeed prob will be > self.threshold
            crop_img = img[box[1]:box[3], box[0]:box[2]]
            crop_img_pil = Image.fromarray(crop_img)
            detected_faces.append(crop_img_pil)
        return detected_faces, boxes