import numpy as np
import torch
from PIL import Image
import cv2
from torch.autograd import Variable
from mtcnn_package.src.get_nets import ONet
from mtcnn_package.src.box_utils import nms, calibrate_box, get_image_boxes
from mtcnn_package.src.align_trans import get_reference_facial_points, warp_and_crop_face
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')

class MTCNN_Alignment():
    def __init__(self, device='cpu', landmarks_threshold=0.5):
        self.align_device = torch.device(device)
        self.onet = ONet().to(self.align_device)
        self.onet.eval()
        self.reference = get_reference_facial_points(default_square= True)
        self.landmarks_threshold = landmarks_threshold
        
    def align(self, img):
        _, landmarks = self.find_landmarks(img)
        facial5points = [[landmarks[0][j],landmarks[0][j+5]] for j in range(5)]
        warped_face = warp_and_crop_face(np.array(img), facial5points, self.reference, crop_size=(112,112))
        return Image.fromarray(warped_face), facial5points

    def show_landmarks(self, img, landmarks):
        # landmarks is facial5points above
        # img is PIL image
        if len(landmarks) != 5:
            landmarks = [[landmarks[0][j],landmarks[0][j+5]] for j in range(5)]
        img_cv2 = np.asarray(img) 
        cv2.circle(img_cv2, (landmarks[0][0], landmarks[0][1]), 1, (0, 0, 255), 4)
        cv2.circle(img_cv2, (landmarks[1][0], landmarks[1][1]), 1, (0, 255, 255), 4)
        cv2.circle(img_cv2, (landmarks[2][0], landmarks[2][1]), 1, (255, 0, 255), 4)
        cv2.circle(img_cv2, (landmarks[3][0], landmarks[3][1]), 1, (0, 255, 0), 4)
        cv2.circle(img_cv2, (landmarks[4][0], landmarks[4][1]), 1, (255, 0, 0), 4)
        img_pil = Image.fromarray(img_cv2)
        return img_pil

    def find_landmarks(self, image, nms_threshold=0.7):
        """
        Arguments:
            image: an instance of PIL.Image.
            thresholds: scalar. --> self.landmarks_threshold
            nms_thresholds: scalar.
        Returns:
            two float numpy arrays of shapes [n_boxes, 4] and [n_boxes, 10], # Indeed n_boxes = 1
            bounding boxes and facial landmarks.
        """

        # BUILD AN IMAGE PYRAMID
        width, height = image.size
        
        threshold = self.landmarks_threshold

        # it will be returned
        bounding_boxes = np.asarray([[0, 0, width, height, 1]])
        with torch.no_grad():
            img_boxes = get_image_boxes(bounding_boxes, image, size=48)
            if len(img_boxes) == 0: 
                return [], []
            img_boxes = torch.FloatTensor(img_boxes).to(self.align_device)
            output = self.onet(img_boxes)
            landmarks = output[0].cpu().data.numpy()  # shape [n_boxes, 10]
            offsets = output[1].cpu().data.numpy()  # shape [n_boxes, 4]
            probs = output[2].cpu().data.numpy()  # shape [n_boxes, 2]
            print(probs)
            keep = np.where(probs[:, 1] > threshold)[0]
            print(probs[:, 1])

            bounding_boxes = bounding_boxes[keep]
            bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
            offsets = offsets[keep]
            landmarks = landmarks[keep]

            # compute landmark points
            width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
            height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
            xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
            landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1)*landmarks[:, 0:5]
            landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1)*landmarks[:, 5:10]

            bounding_boxes = calibrate_box(bounding_boxes, offsets)
            keep = nms(bounding_boxes, nms_threshold, mode='min')
            bounding_boxes = bounding_boxes[keep]
            landmarks = landmarks[keep]


        return bounding_boxes, landmarks