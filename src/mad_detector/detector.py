#!/usr/bin/env python

import torch
import os
import cv2
from mad_detector.yolo_infer import load_infer_from_file


class RFSignsDetector(object):
    def __init__(self, model_path):
        
        self.infer = load_infer_from_file(
            model_path=model_path,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            conf_threshold=0.6,
            nms_threshold=0.3,
            # use_half_precision=True,  # use it only for GPU device!
        )
        self.labels = self.infer.get_labels()
    
    def find_signs(self, image):
        # Must be RGB image!
        bboxes, labels, scores = self.infer.infer_image(image)
        
        label_names = [self.labels[int(labels)] for lbl_idx in labels]
        
        return bboxes, label_names, scores
        

def get_images_from_directory(dirpath):
    fpaths = [os.path.join(dirpath, fname) 
                for fname in os.listdir(dirpath) 
                    if fname.split('.')[-1].lower() in ['png', 'jpg', 'jpeg']]

    return fpaths

        
if __name__ == '__main__':
    import rospy
    rospy.init_node('test_node')
    
    # Get args 
    model_path = rospy.get_param('~model_path')
    input_path = rospy.get_param('~input')
    
    RESULT_DIRECTORY = os.path.join(input_path, 'predicted')
    
    try:
        os.makedirs(RESULT_DIRECTORY)
    except:
        pass
    
    # Execute
    det = RFSignsDetector(model_path)
    im_fpaths = get_images_from_directory(input_path)
    
    for im_fpath in im_fpaths:
        img = cv2.imread(im_fpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to minimal 480
        TARGET_MIN_SIDE = 480.
        rsz_ratio = TARGET_MIN_SIDE/min(img.shape[:2])
        img = cv2.resize(img, None, fx=rsz_ratio, fy=rsz_ratio)
        
        # Predict (can be any size of input)
        bboxes, labels, scores = det.find_signs(img)

        # Render boxes
        for i_p, (x, y, w, h) in enumerate(bboxes):
            score = str(round(scores[i_p], 2))
            
            cv2.rectangle(
                img,
                (int(x), int(y)),
                (int(x+w), int(y+h)),
                color=(0, 0, 255),
                thickness=2
            )
            
            font_sz = 0.7
            font_width = 1
            cv2.putText(
                img, 
                text='{} {}'.format(labels[i_p], score),  
                org=(int(x), int(y-5)),  
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=font_sz,
                color=(0, 0, 255),
                lineType=cv2.LINE_AA, 
                thickness=font_width)

        result_img_fpath = os.path.join(RESULT_DIRECTORY, os.path.basename(im_fpath))        
        # To BGR for saving
        cv2.imwrite(result_img_fpath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
