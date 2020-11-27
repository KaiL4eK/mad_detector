from dcvt.utils.transform import ResizeKeepingRatio

import albumentations as albu


def get_transform(target_sz_hw, config):
    transform_cfg = {
        'fill_value': config.model.fill_value,
        'is_tiled': config.model.tiled,
        'min_side_size': config.model.min_side_size,

        'min_visibility': config.dataset.augmentation.min_visibility,
        'min_area_px': config.dataset.augmentation.min_area_px,
    }

    fill_value = transform_cfg['fill_value']
    is_tiled = transform_cfg['is_tiled']
    min_visibility = transform_cfg.get('min_visibility', 0)
    min_area_px = transform_cfg.get('min_area_px', 0)
    
    # if is_tiled:
    #     return _get_transform_tiled(target_sz_hw, transform_cfg)
    
    train_preprocess = [
        albu.ToGray(p=.3),
        # albu.HorizontalFlip(p=.5),
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
        albu.RandomSunFlare(p=.5, src_radius=int(min(target_sz_hw)*0.5)),
        
        ResizeKeepingRatio(
            target_wh=target_sz_hw[::-1],
            always_apply=True
        ),
        # cv::BORDER_CONSTANT = 0
        albu.PadIfNeeded(min_height=target_sz_hw[0],
                        min_width=target_sz_hw[1],
                        border_mode=0,
                        value=fill_value,
                        always_apply=True),
        
        albu.CoarseDropout(
            max_width=int(target_sz_hw[1]*0.1),
            max_height=int(target_sz_hw[0]*0.1),
            min_width=1,
            min_height=1,
            max_holes=15,
            min_holes=1,
            fill_value=fill_value,
            p=0.9
        ),
        albu.ShiftScaleRotate(
            shift_limit=.3,
            scale_limit=.4,
            rotate_limit=10, 
            interpolation=3,
            border_mode=0, value=fill_value, p=.5),
    ]

    val_preprocess = [
        ResizeKeepingRatio(
            target_wh=target_sz_hw[::-1], always_apply=True),
        # cv::BORDER_CONSTANT = 0
        albu.PadIfNeeded(min_height=target_sz_hw[0],
                        min_width=target_sz_hw[1],
                        border_mode=0,
                        value=fill_value,
                        always_apply=True),
    ]

    train_transform = albu.Compose(
        train_preprocess, p=1,
        bbox_params=albu.BboxParams(
            # COCO bbox ~ [ul_x, ul_y, w, h]
            format='coco', 
            # label_fields=['id'],
            min_area=min_area_px,
            # NOTE - set visibility to filter out of border bboxes
            min_visibility=min_visibility
        )
    )
    
    val_transform = albu.Compose(
        val_preprocess, p=1,
        bbox_params=albu.BboxParams(
            # COCO bbox ~ [ul_x, ul_y, w, h]
            format='coco', 
            # label_fields=['id'],
            min_area=min_area_px,
            # NOTE - set visibility to filter out of border bboxes
            min_visibility=min_visibility
        )
    )
    
    return train_transform, val_transform


# def _get_transform_tiled(target_sz_hw, transform_cfg):
#     fill_value = transform_cfg['fill_value']
#     min_side_size = transform_cfg['min_side_size']

#     train_preprocess = [
#         albu.ToGray(p=.3),
#         albu.HorizontalFlip(p=.5),
#         albu.OneOf([
#             albu.IAAAdditiveGaussianNoise(p=.5),
#             albu.GaussNoise(p=.5),
#             albu.MultiplicativeNoise(per_channel=True, p=.3),
#         ], p=0.4),
#         albu.ImageCompression(quality_lower=80, quality_upper=100, p=.5),
#         albu.OneOf([
#             albu.MotionBlur(blur_limit=3, p=0.2),
#             albu.MedianBlur(blur_limit=3, p=0.2),
#             albu.GaussianBlur(blur_limit=3, p=0.2),
#             albu.Blur(blur_limit=3, p=0.2),
#         ], p=0.2),
#         albu.OneOf([
#             albu.CLAHE(),
#             albu.IAASharpen(),
#             albu.IAAEmboss(),
#             albu.RandomBrightnessContrast(),
#         ], p=0.3),
#         albu.HueSaturationValue(p=0.3),
#         albu.SmallestMaxSize(
#             max_size=min_side_size, 
#             always_apply=True,
#             p=1),
#         albu.CoarseDropout(
#             max_width=int(target_sz_hw[1]*0.1),
#             max_height=int(target_sz_hw[0]*0.1),
#             min_width=1,
#             min_height=1,
#             max_holes=10,
#             min_holes=1,
#             fill_value=fill_value,
#             p=0.9
#         ),
#         albu.ShiftScaleRotate(
#             shift_limit=.20,
#             scale_limit=.20,
#             rotate_limit=.10, 
#             interpolation=3,
#             border_mode=0, value=fill_value, p=.5),
#         albu.RandomCrop(
#             height=target_sz_hw[0],
#             width=target_sz_hw[1],
#             always_apply=True
#         )
#     ]

#     val_preprocess = [
#         albu.SmallestMaxSize(
#             max_size=min_side_size, 
#             always_apply=True,
#             p=1),
#         albu.RandomCrop(
#             height=target_sz_hw[0],
#             width=target_sz_hw[1],
#             always_apply=True
#         )
#     ]

#     return train_preprocess, val_preprocess

