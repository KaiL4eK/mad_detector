import torch
from torch.nn import BatchNorm2d
import numpy as np
import cv2

from .models import get_model_type


def letterbox(img, target_sz_hw, fill_value):
    target_sz = np.array(target_sz_hw)
    current_sz = np.array(img.shape[:2]).astype(np.float32)
    scale = min(target_sz/current_sz)

    new_sz = current_sz * scale
    # Must be width, height
    img = cv2.resize(img, tuple(map(int, new_sz[::-1])), None)

    padding = target_sz - new_sz
    pad = padding.astype(int)//2

    new_img = np.ones((target_sz[0], target_sz[1], 3), dtype=np.uint8)*fill_value
    new_img[pad[0]:pad[0]+img.shape[0],
            pad[1]:pad[1]+img.shape[1], :] = img

    return new_img, (pad, scale)


def diou_xywh(bboxes_a, bboxes_b):

    a_br = bboxes_a[:, None, :2] + bboxes_a[:, None, 2:]
    b_br = bboxes_b[:, :2] + bboxes_b[:, 2:]

    a_cntr = bboxes_a[:, None, :2] + bboxes_a[:, None, 2:]/2
    b_cntr = bboxes_a[:, None, :2] + bboxes_a[:, None, 2:]/2

    # intersection top left
    tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
    # intersection bottom right
    br = torch.min(a_br, b_br)

    # convex (smallest enclosing box) top left and bottom right
    con_tl = torch.min(bboxes_a[:, None, :2], bboxes_b[:, :2])
    con_br = torch.max(a_br, b_br)
    # centerpoint distance squared
    rho2 = ((a_cntr - b_cntr) ** 2 / 4).sum(dim=-1)

    area_a = torch.prod(bboxes_a[:, 2:], 1)
    area_b = torch.prod(bboxes_b[:, 2:], 1)

    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    area_u = area_a[:, None] + area_b - area_i
    iou = area_i / area_u

    # convex diagonal squared
    c2 = torch.pow(con_br - con_tl, 2).sum(dim=2) + 1e-16
    return iou - rho2 / c2  # DIoU


def nms_diou(preds, nms_thresh=0.5):
    # preds ~ [[x, y, w, h, score, label]]
    # NMS for one image (prediction ~ bboxes with labels and scores)
    assert preds.shape[1] == 6

    if not isinstance(preds, torch.Tensor):
        preds = torch.FloatTensor(preds)

    ordered_idxs = (-preds[:, 4]).argsort()

    iou_matrix = diou_xywh(
        preds[:, :4],
        preds[:, :4]
    )

    keep = []
    while ordered_idxs.shape[0] > 0:
        idx_self = ordered_idxs[0]
        keep.append(idx_self)

        # ious = iou_matrix[idx_self]
        # assert ious[0] > 0.99

        preds_other = preds[ordered_idxs, :]
        ious = iou_matrix[idx_self, ordered_idxs]
        # preds_check = preds[None, idx_self, :]

        check_label = preds[idx_self, 5]

        # ious = diou_xywh(
        #     preds_check[:, :4],
        #     preds_other[:, :4]
        # )
        # ious = ious[0]
        # [1, other_preds]

        high_iou_inds = (ious >= nms_thresh)
        same_classes_inds = preds_other[:, 5] == check_label
        mask = ~(high_iou_inds & same_classes_inds)

        ordered_idxs = ordered_idxs[mask]

    return torch.LongTensor(keep)


def load_infer_from_file(model_path, device, nms_threshold=0.4, conf_threshold=0.6, use_half_precision=False):
    loaded_data = torch.load(model_path)
    model_config = loaded_data['model_config']
    model_state = loaded_data['model_state']
    
    # Only for Python 2 =(
    # model_config['anchors'] = np.array(model_config['anchors'], dtype=np.float32)
    model_config['strides'] = np.array(model_config['strides'], dtype=np.float32)

    return InferDetection(
        model_state=model_state,
        model_config=model_config,
        device=device,
        nms_threshold=nms_threshold,
        conf_threshold=conf_threshold,
        use_half_precision=use_half_precision
    )


class InferDetection():
    def __init__(self, model_config, device, model_state=None, nms_threshold=0.4, conf_threshold=0.6, use_half_precision=False):
        model_type = get_model_type(model_config['name'])
        self.model_wh = model_config['infer_sz_hw'][::-1]
        self.use_half_precision = use_half_precision

        self.config = model_config
        self.device = device
        self.model = model_type(config=model_config,
                                inference=True).to(device).eval()

        if self.use_half_precision:
            self.model.half()
            for layer in self.model.modules():
                if isinstance(layer, BatchNorm2d):
                    layer.float()

        if model_state is not None:
            self.update_model_state(model_state)

        self.nms_threshold = nms_threshold
        self.conf_threshold = conf_threshold
        self.tiled = model_config.get('tiled', False)
        self.labels = model_config['labels']

        self.fill_value = model_config['fill_value']

        self._size_tnsr = torch.FloatTensor([
            self.model_wh[0],
            self.model_wh[1],
            self.model_wh[0],
            self.model_wh[1],
        ]).view(1, 1, 4).to(self.device)

    def get_labels(self):
        return self.labels

    def _image_2_tensor(self, img, target_wh, fill_value):
        image, (pad, scale) = letterbox(
            img=img,
            target_sz_hw=target_wh[::-1],
            fill_value=fill_value
        )

        image = image.astype(np.float32)/255.
        tnsr_image = torch.from_numpy(image.transpose(2, 0, 1))

        return tnsr_image, (pad[1], pad[0]), scale

    def update_model_state(self, model_state):
        self.model.load_state_dict(model_state)

    def infer_image(self, image):
        if self.tiled:
            return self._infer_image_tiled(image)
        else:
            return self.infer_batch([image])[0]

    def infer_batch(self, imgs_list):
        batch_tensor = []
        _scale_list = []
        _padding_list = []

        result_list = []

        for img in imgs_list:
            tensor, paddings, rsz_scale = self._image_2_tensor(
                img=img,
                target_wh=self.model_wh,
                fill_value=self.fill_value
            )
            batch_tensor.append(tensor)

            _scale_list.append(rsz_scale)
            _padding_list.append(paddings)

        _scale_tnsr = torch.FloatTensor(
            _scale_list).view(-1, 1, 1).to(self.device)
        _pad_tnsr = torch.FloatTensor(
            _padding_list).view(-1, 1, 2).to(self.device)

        batch_tensor = torch.stack(batch_tensor, axis=0)

        with torch.no_grad():
            batch_tensor = batch_tensor.to(self.device)
            if self.use_half_precision:
                batch_tensor = batch_tensor.half()

            outputs = self.model(batch_tensor)
            
            outputs[..., :4] *= self._size_tnsr
            outputs[..., :2] -= _pad_tnsr
            outputs[..., :4] /= _scale_tnsr

            if self.use_half_precision:
                outputs = outputs.float()

            outputs = outputs.cpu()

        # Go through batches
        for i, output in enumerate(outputs):
            bboxes = np.array([])
            scores = np.array([])
            labels = np.array([])

            preds = output[output[..., 4] > self.conf_threshold]
            if preds.shape[0] > 0:
                keep = nms_diou(preds, nms_thresh=self.nms_threshold)
                preds = preds[keep].numpy()

                bboxes = preds[:, :4]
                scores = preds[:, 4]
                labels = preds[:, 5].astype(int)

            result_list.append(
                (
                    bboxes,
                    labels,
                    scores,
                )
            )

        return result_list
