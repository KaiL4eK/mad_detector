import os
import cv2
import torch

from dcvt.common.utils.fs import get_images_from_directory
from dcvt.detection.infer import InferDetection


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Test script for detector')
    parser.add_argument('-m', action="store", dest="model", required=True)
    parser.add_argument('-i', action="store", dest="input", required=True)

    args = parser.parse_args()
    return vars(args)   # As dictionary


def init_ros():
    import rospy
    rospy.init_node('test_node')
    # Get args 
    args = {
        'model': rospy.get_param('~model_path'),
        'input': rospy.get_param('~input')
    }
    

class RFSignsDetector(object):
    def __init__(self, model_path):
        
        self.infer = InferDetection.from_file(
            model_filepath=model_path,
            conf_threshold=0.5,
            nms_threshold=0.3,
            # use_half_precision=True,  # use it only for GPU device!
        )
        self.label_names = self.infer.get_labels()
    
    def find_signs(self, image):
        # Must be RGB image!
        bboxes, label_ids, scores = self.infer.infer_image(image)
        labels = self.infer.map_labels(label_ids)
        return bboxes, labels, scores
    

if __name__ == '__main__':
    args = get_args()
    input_path = args['input']
    model_path = args['model']
    
    RESULT_DIRECTORY = os.path.join(input_path, 'predicted')
    
    try:
        os.makedirs(RESULT_DIRECTORY)
    except:
        pass
    
    # Execute
    det = RFSignsDetector(model_path)
    im_fpaths = get_images_from_directory(input_path)
    
    for im_fpath in im_fpaths:
        print(f'Processing file: {im_fpath}')
        
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
