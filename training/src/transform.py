import albumentations as albu
import numpy as np


class AlbuAugmentation():
    def __init__(self, **config):
        self.fill_value = config['fill_value']
        self.min_area_px = config['min_area_px']
        self.min_visibility = config['min_visibility']
        self.target_sz_hw = config['target_sz_hw']

        if not isinstance(self.fill_value, (list, tuple, np.ndarray)):
            self.fill_value = [self.fill_value]*3

        self.description = [
            albu.ToGray(p=.3),
            albu.HorizontalFlip(p=.5),
            albu.OneOf([
                albu.IAAAdditiveGaussianNoise(p=.5),
                albu.GaussNoise(p=.5),
                albu.MultiplicativeNoise(per_channel=True, p=.3),
            ], p=0.4),
            albu.ImageCompression(quality_lower=90, quality_upper=100, p=.5),
            albu.OneOf([
                albu.MotionBlur(blur_limit=3, p=0.2),
                albu.MedianBlur(blur_limit=3, p=0.2),
                albu.GaussianBlur(blur_limit=3, p=0.2),
                albu.Blur(blur_limit=3, p=0.2),
            ], p=0.2),
            albu.OneOf([
                albu.CLAHE(),
                albu.IAASharpen(),
                albu.IAAEmboss(),
                albu.RandomBrightnessContrast(),
            ], p=0.3),
            albu.HueSaturationValue(p=.3),
            albu.RandomSunFlare(p=.5, src_radius=int(min(self.target_sz_hw)*0.5)),
            albu.ShiftScaleRotate(
                shift_limit=.3,
                scale_limit=.1,
                rotate_limit=10, 
                interpolation=3,
                border_mode=0, 
                value=self.fill_value,
                p=.5),
        ]

        self.compose = albu.Compose(
            self.description, p=1,
            bbox_params=albu.BboxParams(
                # COCO bbox ~ [ul_x, ul_y, w, h]
                format='coco', 
                label_fields=['labels'],
                min_area=self.min_area_px,
                # NOTE - set visibility to filter out of border bboxes
                min_visibility=self.min_visibility
            )
        )
        
    def transform(self, img, ann):
        transformed = self.compose(
            image=img,
            bboxes=ann['bboxes'],
            labels=ann['labels']
        )

        ann['bboxes'] = np.array(transformed['bboxes'])
        ann['labels'] = np.array(transformed['labels'])
        img = transformed['image']

        return img, ann
        
    def serialize(self):
        return albu.to_dict(self.compose)
