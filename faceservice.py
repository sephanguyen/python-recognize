from flask_restful import Resource, Api
from seesawfacenet.seesaw import Seesaw_Recognise
from PIL import Image
from utils import convert_pil_rgb2bgr
import cv2

# seesaw_model = Seesaw_Recognise(pretrained_path='pretrained_model/DW_SeesawFaceNetv2.pth', save_facebank_path='facebank/', device='cpu') # device='cuda:0' if GPU available

# # img = Image.open('/Users/nguyenpha/WorkPlace/work-main/Omnigo/general_images/test_01.jpg') # Should be RGB
# image = cv2.imread('/Users/nguyenpha/WorkPlace/work-main/Omnigo/general_images/test_01.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB --> dont know why the default is BGR
# faces, boxes = seesaw_model.detect_model.detect_face(image)
# img = faces[0] # should only has 1 face in an image
#             # align
# img,_ = seesaw_model.alignment_model.align(img)
#             # Convert to BGR (IMPORTANT)
# img = convert_pil_rgb2bgr(img)
#             # recognise
# name, distance = seesaw_model.infer([img])
# print(name)

# # Alignment
# aligned_img,_ = seesaw_model.alignment_model.align(img)

# # Convert to BGR
# aligned_img_bgr = convert_pil_rgb2bgr(aligned_img) # Convert to BGR (recognise step only work on BGR format)

# # Recognition
# name, distance = seesaw_model.infer([aligned_img_bgr]) # Input is a list, then need to put in [ ... ]
# predicted_name = name[0] # 1st and the only element in the output list
# predicted_distance = distance[0].numpy() # 1st and the only element in the output list


class FaceService(Resource):
    def get(self):
        # img = Image.open('path/to/image.jpg') # Should be RGB


        return {'employees': [1, 2]} # Fetches first column that is Employee ID