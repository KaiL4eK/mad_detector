import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import sys

def get_region_boxes(boxes_and_confs):
    boxes_list = []
    confs_list = []

    for item in boxes_and_confs:
        boxes_list.append(item[0])
        confs_list.append(item[1])

    # boxes: [batch, num1 + num2 + num3, 4]
    # confs: [batch, num1 + num2 + num3, num_classes]
    boxes = torch.cat(boxes_list, dim=1)
    confs = torch.cat(confs_list, dim=1)

    # NOTE - Changed format for easier evaluation
    boxes[..., :2] = boxes[..., :2]-boxes[..., 2:]/2 # to coco format
    scores, labels = confs.max(-1, keepdim=True)
    output = torch.cat(
        (
            boxes,
            scores,
            labels.type(boxes.dtype)
        ),
        dim=-1
    )
    return output


def yolo_forward(output, num_classes, 
                 anchors_wh, num_anchors):
    # Output would be invalid if it does not satisfy this assert
    assert (output.size(1) == (5 + num_classes) * num_anchors)

    # print(output.size())

    # Slice the second dimension (channel) of output into:
    # [ 2, 2, 1, num_classes, 2, 2, 1, num_classes, 2, 2, 1, num_classes ]
    # And then into
    # bxy = [ 6 ] bwh = [ 6 ] det_conf = [ 3 ] cls_conf = [ num_classes * 3 ]
    batch = output.size(0)
    H = output.size(2)
    W = output.size(3)
    dtype = output.dtype

    bxy_list = []
    bwh_list = []
    det_confs_list = []
    cls_confs_list = []

    for i in range(num_anchors):
        begin = i * (5 + num_classes)
        end = (i + 1) * (5 + num_classes)

        bxy_list.append(output[:, begin: begin + 2])
        bwh_list.append(output[:, begin + 2: begin + 4])
        det_confs_list.append(output[:, begin + 4: begin + 5])
        cls_confs_list.append(output[:, begin + 5: end])

    # Shape: [batch, num_anchors * 2, H, W]
    bxy = torch.cat(bxy_list, dim=1)
    # Shape: [batch, num_anchors * 2, H, W]
    bwh = torch.cat(bwh_list, dim=1)

    # Shape: [batch, num_anchors, H, W]
    det_confs = torch.cat(det_confs_list, dim=1)
    # Shape: [batch, num_anchors * H * W]
    # det_confs = det_confs.view(batch, num_anchors * H * W)
    det_confs = det_confs.reshape(batch, num_anchors * H * W)

    # Shape: [batch, num_anchors * num_classes, H, W]
    cls_confs = torch.cat(cls_confs_list, dim=1)
    # Shape: [batch, num_anchors, num_classes, H * W]
    cls_confs = cls_confs.view(batch, num_anchors, num_classes, H * W)
    # Shape: [batch, num_anchors, num_classes, H * W] --> [batch, num_anchors * H * W, num_classes]
    cls_confs = cls_confs.permute(0, 1, 3, 2).reshape(
        batch, num_anchors * H * W, num_classes)

    # Apply sigmoid(), exp() and softmax() to slices
    #
    bxy = torch.sigmoid(bxy) # * scale_x_y - 0.5 * (scale_x_y - 1)
    bwh = torch.exp(bwh)
    det_confs = torch.sigmoid(det_confs)
    cls_confs = torch.nn.Softmax(dim=2)(cls_confs)

    # Prepare C-x, C-y, P-w, P-h (None of them are torch related)
    grid_x = np.expand_dims(np.expand_dims(np.expand_dims(
        np.linspace(0, W - 1, W), axis=0).repeat(H, 0), axis=0), axis=0)
    grid_y = np.expand_dims(np.expand_dims(np.expand_dims(
        np.linspace(0, H - 1, H), axis=1).repeat(W, 1), axis=0), axis=0)
    # grid_x = torch.linspace(0, W - 1, W).reshape(1, 1, 1, W).repeat(1, 1, H, 1)
    # grid_y = torch.linspace(0, H - 1, H).reshape(1, 1, H, 1).repeat(1, 1, 1, W)

    anchor_w, anchor_h = anchors_wh

    device = None
    cuda_check = output.is_cuda
    if cuda_check:
        device = output.get_device()

    bx_list = []
    by_list = []
    bw_list = []
    bh_list = []

    # Apply C-x, C-y, P-w, P-h
    for i in range(num_anchors):
        ii = i * 2
        # Shape: [batch, 1, H, W]
        # grid_x.to(device=device, dtype=torch.float32)
        bx = bxy[:, ii: ii + 1] + \
            torch.tensor(grid_x, device=device, dtype=dtype)
        # Shape: [batch, 1, H, W]
        # grid_y.to(device=device, dtype=torch.float32)
        by = bxy[:, ii + 1: ii + 2] + \
            torch.tensor(grid_y, device=device, dtype=dtype)
        # Shape: [batch, 1, H, W]
        bw = bwh[:, ii: ii + 1] * anchor_w[i]
        # Shape: [batch, 1, H, W]
        bh = bwh[:, ii + 1: ii + 2] * anchor_h[i]

        bx_list.append(bx)
        by_list.append(by)
        bw_list.append(bw)
        bh_list.append(bh)

    ########################################
    #   Figure out bboxes from slices     #
    ########################################

    # Shape: [batch, num_anchors, H, W]
    bx = torch.cat(bx_list, dim=1)
    # Shape: [batch, num_anchors, H, W]
    by = torch.cat(by_list, dim=1)
    # Shape: [batch, num_anchors, H, W]
    bw = torch.cat(bw_list, dim=1)
    # Shape: [batch, num_anchors, H, W]
    bh = torch.cat(bh_list, dim=1)

    # Shape: [batch, 2 * num_anchors, H, W]
    bx_bw = torch.cat((bx, bw), dim=1)
    # Shape: [batch, 2 * num_anchors, H, W]
    by_bh = torch.cat((by, bh), dim=1)

    # normalize coordinates to [0, 1]
    bx_bw /= W
    by_bh /= H

    # Shape: [batch, num_anchors * H * W, 1]
    bx = bx_bw[:, :num_anchors].view(batch, num_anchors * H * W, 1)
    by = by_bh[:, :num_anchors].view(batch, num_anchors * H * W, 1)
    bw = bx_bw[:, num_anchors:].view(batch, num_anchors * H * W, 1)
    bh = by_bh[:, num_anchors:].view(batch, num_anchors * H * W, 1)

    # Shape: [batch, num_anchors * h * w, 4]
    boxes = torch.cat((bx, by, bw, bh), dim=2).view(
        batch, num_anchors * H * W, 4)

    # boxes:     [batch, num_anchors * H * W, num_classes, 4]
    # cls_confs: [batch, num_anchors * H * W, num_classes]
    # det_confs: [batch, num_anchors * H * W]

    det_confs = det_confs.view(batch, num_anchors * H * W, 1)

    if num_classes > 1:
        confs = cls_confs * det_confs
    else:
        confs = det_confs

    # boxes: [batch, num_anchors * H * W, 4]
    # confs: [batch, num_anchors * H * W, num_classes]

    return boxes, confs


class YoloLayer(nn.Module):
    def __init__(self, anchor_mask=[], num_classes=0, 
                anchors=[], stride=32, inference=True):
        super(YoloLayer, self).__init__()
        self.anchor_mask = anchor_mask
        self.num_classes = num_classes
        self.anchors = anchors
        self.anchor_step = 2
        self.num_anchors = len(anchors) // 2
        self.stride = stride
        
        self.inference = inference

        self.masked_anchors = []
        for m in self.anchor_mask:
            self.masked_anchors += self.anchors[m *
                                           self.anchor_step:(m + 1) * self.anchor_step]
        self.masked_anchors = [anchor / self.stride for anchor in self.masked_anchors]

        self.anchors_wh = (
            self.masked_anchors[::2], 
            self.masked_anchors[1::2]
        )

    def forward(self, output, target=None):
        # if self.training:
        if not self.inference:
            return output

        return yolo_forward(output, 
                            self.num_classes, 
                            self.anchors_wh, 
                            len(self.anchor_mask))


class ResConv2dBatchLeaky(nn.Module):
    def __init__(self, in_channels, return_extra=False):
        super(ResConv2dBatchLeaky, self).__init__()
        # Output = in_ch * 2

        half_ch = in_channels//2
        self.return_extra = return_extra

        self.conv1 = Conv_Bn_Activation(half_ch, half_ch, 3, 1, 'leaky')
        self.conv2 = Conv_Bn_Activation(half_ch, half_ch, 3, 1, 'leaky')
        self.conv3 = Conv_Bn_Activation(
            in_channels, in_channels, 1, 1, 'leaky')

    def forward(self, x):
        # Groups=2, group_id=1 (second part of channels)
        h_ch = x.shape[1]//2
        x0 = x[:, h_ch:, ...]

        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x2 = torch.cat([x1, x2], dim=1)

        x3 = self.conv3(x2)
        out = torch.cat([x, x3], dim=1)

        if self.return_extra:
            return out, x3
        else:
            return out


class Mish(torch.nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x



class Conv_Bn_Activation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, bn=True, bias=False):
        super(Conv_Bn_Activation, self).__init__()
        pad = (kernel_size - 1) // 2

        self.conv = nn.ModuleList()
        if bias:
            self.conv.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad))
        else:
            self.conv.append(nn.Conv2d(in_channels, out_channels,
                                       kernel_size, stride, pad, bias=False))
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))
        if activation == "mish":
            self.conv.append(Mish())
        elif activation == "relu":
            self.conv.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "linear":
            pass
        else:
            print("activate error !!! {} {} {}".format(sys._getframe().f_code.co_filename,
                                                       sys._getframe().f_code.co_name, sys._getframe().f_lineno))

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x



class Upsample(nn.Module):
    def __init__(self, inference):
        super(Upsample, self).__init__()
        self.inference = inference

    def forward(self, x, target_size):
        assert (x.data.dim() == 4)
        _, _, tH, tW = target_size

        # if not self.training:
        if self.inference:
            B = x.data.size(0)
            C = x.data.size(1)
            H = x.data.size(2)
            W = x.data.size(3)

            return x.view(B, C, H, 1, W, 1).expand(B, C, H, tH // H, W, tW // W).contiguous().view(B, C, tH, tW)
        else:
            return F.interpolate(x, size=(tH, tW), mode='nearest')


