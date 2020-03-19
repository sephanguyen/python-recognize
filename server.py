from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from json import dumps
from faceservice import FaceService
import json
from seesawfacenet.seesaw import Seesaw_Recognise
from PIL import Image
from utils import convert_pil_rgb2bgr
import cv2
import numpy as np
app = Flask(__name__)
api = Api(app)

ALLOWED_EXTENSIONS = set([ 'png', 'jpg', 'jpeg', 'gif'])


seesaw_model = Seesaw_Recognise(pretrained_path='pretrained_model/DW_SeesawFaceNetv2.pth', save_facebank_path='facebank/', device='cpu') # device='cuda:0' if GPU available

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	

# api.add_resource(FaceService, '/faceservice') # Route_1

@app.route('/faceservice', methods=['POST'])
def faceservice():

     file = request.files['file']
     res = {}
     if file.filename != '':
          if file and allowed_file(file.filename):
               filestr = file.read()
               npimg = np.frombuffer(filestr, np.uint8)
               image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
               image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB --> dont know why the default is BGR
               faces, boxes = seesaw_model.detect_model.detect_face(image)
               img = faces[0] # should only has 1 face in an image
               # align
               img,_ = seesaw_model.alignment_model.align(img)
               # Convert to BGR (IMPORTANT)
               img = convert_pil_rgb2bgr(img)
               # recognise
               name, distance = seesaw_model.infer([img])
               print(name)
               idName = name[0] if len(name) >= 1 else 'water' 'UNKNOW'
               res = {'id': idName}
     
     response = app.response_class(
          response=json.dumps(res),
          status=200,
          mimetype='application/json'
     )

     return response


if __name__ == '__main__':
     app.run(port=5002)