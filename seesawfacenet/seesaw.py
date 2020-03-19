import torch.backends.cudnn as cudnn
from seesawfacenet.seesaw_package.DW_SeesawFaceNetv2 import DW_SeesawFaceNetv2, l2_norm
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as trans
import math
from collections import OrderedDict
from mtcnn import MTCNN_Alignment
from seesawfacenet.ultra_face.ultraface import Ultraface_detect
from utils import convert_pil_rgb2bgr, draw_box_name
from pathlib import Path
import cv2

class Seesaw_Recognise(object):
    def __init__(self, pretrained_path = '/directory/to/model.pth', save_facebank_path = 'directory/to/facebank.pth and .npy folder', device='cpu'):
        self.alignment_model = MTCNN_Alignment(device=device, landmarks_threshold=0.4)
        self.detect_model = Ultraface_detect(model_path='ultraface_package/models/pretrained/version-RFB-320.pth', device=device)        
        
        self.embedding_size = 512
        self.device=torch.device(device)
        self.threshold = 1.35
        self.model = DW_SeesawFaceNetv2(self.embedding_size).to(self.device)
        self.model.eval()
        print('seesawFaceNet model generated')
        
        # load pretrained models
        if pretrained_path != '':
            new_state_dict = OrderedDict()
            state_dict = torch.load(pretrained_path, map_location=torch.device('cpu')) 
            for k, v in state_dict.items():
                if 'module.' in k:
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                else:
                    new_state_dict[k] = v
                    
            self.model.load_state_dict(new_state_dict)
            print('seesawFaceNet model weight loaded')
        else:
            print('Not found pretrained seesaw model')

        # Augumentation from loading images
        self.test_transform = trans.Compose([
                    trans.ToTensor(),
                    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])

        # load facebank
        if save_facebank_path != '':
            self.targets, self.names = self.load_facebank(save_facebank_path)
            print('Facebank loaded')
        else:
            self.targets = None
            self.names = None
            print('No facebank Detected')

    def infer(self, faces, tta=False):
        '''
        faces : list of PIL Image (BGR format)
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        '''
        target_embs = self.targets
        names = self.names
        if target_embs is None or names is None:
            print("No facebank Detected ==>  CANT infering!")
            return None
        embs = []
        for img in faces:
            with torch.no_grad():
                if tta:
                    mirror = trans.functional.hflip(img)
                    emb = self.model(self.test_transform(img).to(self.device).unsqueeze(0))
                    emb_mirror = self.model(self.test_transform(mirror).to(self.device).unsqueeze(0))
                    embs.append(l2_norm(emb + emb_mirror))
                else:                        
                    embs.append(self.model(self.test_transform(img).to(self.device).unsqueeze(0)))
        source_embs = torch.cat(embs)
        
        diff = source_embs.unsqueeze(-1) - target_embs.transpose(1,0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        min_idx[minimum > self.threshold] = -1 # if no match, set idx to -1
        recognise_id = [self.names[x+1] for x in min_idx] # convert to real name
        return recognise_id, minimum # min_idx is index of recognise, minimum is distance of that idx

    def infer_general_image(self, image, plot_result=True, tta=False):
        # image should be in cv2 format
        # If no facebank --> return None
        # If plot_result = True --> return annotated image
        # If plot_result = False --> return cropped faces and their predicted ID
        target_embs = self.targets
        names = self.names
        if target_embs is None or names is None:
            print("No facebank Detected ==>  CANT infering!")
            return None
        
        origin_image = image
        faces, boxes = self.detect_model.detect_face(image)
        list_imgs_to_recognise = []
        for face in faces:
            # align
            img,_ = self.alignment_model.align(face)
            # Convert to BGR (IMPORTANT)
            img = convert_pil_rgb2bgr(img)
            list_imgs_to_recognise.append(img)
        # recognise
        predicted_names, predicted_distances = self.infer(list_imgs_to_recognise, tta=tta)

        if plot_result:
            boxes = boxes.astype(int)
            boxes = boxes + [-1,-1,1,1] # personal choice
            for idx,box in enumerate(boxes):
                image = draw_box_name(box, predicted_names[idx], origin_image)
            return Image.fromarray(image)

        return faces, names

    def create_facebank(self, facebank_path = '/directory/to/facebank/folder/images/', save_facebank_path = '/directory/to/save/folder/', run_detect=True, tta=False):
        # Create facebank from scratch
        # Output is target_emns, and names (like infer function) and save to database
        facebank_path = Path(facebank_path)
        #self.model.eval()
        embeddings =  []
        names = ['Unknown']
        for path in facebank_path.iterdir():
            if path.is_file():
                continue
            else:
                embs = []
                for file in path.iterdir():
                    if not file.is_file():
                        continue
                    else:
                        if run_detect:
                            try:
                                image = cv2.imread(str(file))
                                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB --> dont know why the default is BGR
                                faces, boxes = self.detect_model.detect_face(image)
                                img = faces[0] # should only has 1 face in an image
                            except:
                                continue
                        else:
                            try:
                                img = Image.open(file) # It should return RGB image
                            except:
                                continue
                        # Alignment
                        img,_ = self.alignment_model.align(img)
                        # Convert to BGR
                        img = convert_pil_rgb2bgr(img)
                        # Run embedding
                        with torch.no_grad():
                            if tta:
                                mirror = trans.functional.hflip(img)
                                emb = self.model(self.test_transform(img).to(self.device).unsqueeze(0))
                                emb_mirror = self.model(self.test_transform(mirror).to(self.device).unsqueeze(0))
                                embs.append(l2_norm(emb + emb_mirror))
                            else:                        
                                embs.append(self.model(self.test_transform(img).to(self.device).unsqueeze(0)))
            if len(embs) == 0:
                continue
            embedding = torch.cat(embs).mean(0,keepdim=True)
            embeddings.append(embedding)
            names.append(path.name)

        embeddings = torch.cat(embeddings)
        names = np.array(names)
        if save_facebank_path != '' and save_facebank_path is not None:
            torch.save(embeddings, save_facebank_path+'facebank.pth')
            np.save(save_facebank_path+'names.npy', names)
        return embeddings, names

    def update_facebank(self, new_facebank_path = '/directory/to/new facebank/folder/images/', save_facebank_path = '/directory/to/save/folder/', run_detect=True, tta=False):
        if self.targets is None or self.names is None: # No current facebank --> run create new facebank
            t1, t2 = self.create_facebank(facebank_path=new_facebank_path, save_facebank_path=save_facebank_path, run_detect=run_detect, tta=tta)
            return t1, t2
        # Read new
        new_embeddings, new_names = self.create_facebank(facebank_path=new_facebank_path, save_facebank_path=None, run_detect=run_detect, tta=tta)
        # Concat to current
        final_embeddings = torch.cat([self.targets, new_embeddings])
        new_names = new_names[1:]
        final_names = np.append(self.names, new_names)
        # Save it
        torch.save(final_embeddings, save_facebank_path+'facebank.pth')
        np.save(save_facebank_path+'names.npy', final_names)
        return final_embeddings, final_names

    def load_facebank(self, save_facebank_path='directory/to/facebank.pth and .npy folder/'):
        embeddings = torch.load(save_facebank_path+'facebank.pth')
        names = np.load(save_facebank_path+'names.npy')
        return embeddings, names