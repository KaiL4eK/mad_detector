from .yolo4tiny import Yolov4Tiny

def get_model_type(model_name):
    if model_name.lower() == 'yolov4tiny':
        return Yolov4Tiny
    else:
        raise NotImplementedError
