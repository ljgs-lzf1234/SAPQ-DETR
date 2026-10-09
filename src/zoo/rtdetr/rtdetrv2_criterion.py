"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.

SCB-P2Q additions:
  - P2 recall heatmap supervision for query proposal calibration
  - Optional scale-balanced detection losses for small objects
"""

import torch
import torch.nn as nn
import torch.distributed
import torch.nn.functional as F
import torchvision

import copy
import math
import numpy as np

from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from ...misc.dist_utils import get_world_size, is_dist_available_and_initialized
from ...core import register


# ========================================================================
# P2 recall helpers
# ========================================================================

def _draw_gaussian_numpy(heatmap_np, center_y, center_x, sigma):
    """
    Draw a 2D Gaussian onto heatmap_np at (center_y, center_x), in-place,
    taking element-wise maximum with the existing values.

    Args:
        heatmap_np: np.ndarray of shape [H, W], float32, modified in-place.
        center_y, center_x: float, pixel coordinates of the Gaussian center.
        sigma: float, std of the Gaussian.
    """
    H, W = heatmap_np.shape
    radius = int(3 * sigma) + 1
    y_min = max(0, int(center_y) - radius)
    y_max = min(H, int(center_y) + radius + 1)
    x_min = max(0, int(center_x) - radius)
    x_max = min(W, int(center_x) + radius + 1)
    if y_max <= y_min or x_max <= x_min:
        return
    yy, xx = np.meshgrid(
        np.arange(y_min, y_max, dtype=np.float32),
        np.arange(x_min, x_max, dtype=np.float32),
        indexing='ij'
    )
    gauss = np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma ** 2))
    np.maximum(heatmap_np[y_min:y_max, x_min:x_max], gauss,
               out=heatmap_np[y_min:y_max, x_min:x_max])

@torch.no_grad()
def generate_p2_recall_gt(
    targets,
    p2_h, p2_w,
    small_threshold=0.12,
    small_tau=0.03,
    alpha_min=0.25,
    sigma_scale=0.25,
    device='cpu',
):
    """Generate scale-aware center recall targets on the P2 grid."""
    B = len(targets)
    heatmap_np = np.zeros((B, p2_h, p2_w), dtype=np.float32)

    for b, target in enumerate(targets):
        boxes = target['boxes']
        if len(boxes) == 0:
            continue
        boxes_np = boxes.detach().cpu().numpy() if isinstance(boxes, torch.Tensor) else np.asarray(boxes, dtype=np.float32)

        for box in boxes_np:
            cx, cy, w, h = box
            wh_max = max(float(w), float(h))
            small_gate = 1. / (1. + np.exp((wh_max - small_threshold) / max(small_tau, 1e-6)))
            amplitude = alpha_min + (1. - alpha_min) * small_gate

            cx_p = float(cx) * p2_w
            cy_p = float(cy) * p2_h
            sigma = max(1.0, sigma_scale * max(float(w) * p2_w, float(h) * p2_h))

            single = np.zeros_like(heatmap_np[b])
            _draw_gaussian_numpy(single, cy_p, cx_p, sigma)
            np.maximum(heatmap_np[b], single * amplitude, out=heatmap_np[b])

    return torch.from_numpy(heatmap_np).unsqueeze(1).to(device)


# ========================================================================
# Criterion
# ========================================================================

@register()
class RTDETRCriterionv2(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)

    """
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self, \
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.0,
        num_classes=80,
        boxes_weight_format=None,
        share_matched_indices=False,
        sa_p2q_small_threshold=0.12,
        sa_p2q_small_tau=0.03,
        sa_p2q_det_scale_weight=0.2,
        sa_p2q_recall_alpha_min=0.25,
        sa_p2q_recall_sigma_scale=0.25,
        sa_p2q_recall_gamma=2.0,
        sa_p2q_aux_warm_iters=97000,
        sa_p2q_aux_anneal_iters=56000,
        sa_p2q_aux_min_factor=0.10):
        """Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            num_classes: number of object categories, omitting the special no-object category
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            boxes_weight_format: format for boxes weight (iou, )
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma

        self.sa_p2q_small_threshold = float(sa_p2q_small_threshold)
        self.sa_p2q_small_tau = float(sa_p2q_small_tau)
        self.sa_p2q_det_scale_weight = float(sa_p2q_det_scale_weight)
        self.sa_p2q_recall_alpha_min = float(sa_p2q_recall_alpha_min)
        self.sa_p2q_recall_sigma_scale = float(sa_p2q_recall_sigma_scale)
        self.sa_p2q_recall_gamma = float(sa_p2q_recall_gamma)
        self.sa_p2q_aux_warm_iters = int(sa_p2q_aux_warm_iters)
        self.sa_p2q_aux_anneal_iters = int(sa_p2q_aux_anneal_iters)
        self.sa_p2q_aux_min_factor = float(sa_p2q_aux_min_factor)
        self.register_buffer('_sa_p2q_step', torch.zeros((), dtype=torch.long), persistent=False)

    def _scale_gate(self, boxes):
        wh_max = boxes[:, 2:].max(dim=-1).values
        return torch.sigmoid((self.sa_p2q_small_threshold - wh_max) / max(self.sa_p2q_small_tau, 1e-6))

    def _det_scale_weight(self, boxes):
        return 1. + self.sa_p2q_det_scale_weight * self._scale_gate(boxes)

    def _sa_p2q_aux_factor(self):
        if not self.training:
            return 1.0
        step = int(self._sa_p2q_step.item())
        if self.sa_p2q_aux_anneal_iters <= 0:
            return 1.0
        if step < self.sa_p2q_aux_warm_iters:
            return 1.0
        progress = min(
            1.0,
            max(0.0, (step - self.sa_p2q_aux_warm_iters) / float(self.sa_p2q_aux_anneal_iters)),
        )
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.sa_p2q_aux_min_factor + (1.0 - self.sa_p2q_aux_min_factor) * factor

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        if target_boxes.numel() > 0 and self.sa_p2q_det_scale_weight > 0:
            pos_scale = torch.ones_like(target_score_o, dtype=src_logits.dtype)
            pos_scale[idx] = self._det_scale_weight(target_boxes).to(dtype=src_logits.dtype)
            weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + \
                target_score * pos_scale.unsqueeze(-1)
        else:
            weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        scale_weight = None
        if target_boxes.numel() > 0 and self.sa_p2q_det_scale_weight > 0:
            scale_weight = self._det_scale_weight(target_boxes).to(dtype=loss_bbox.dtype)
            loss_bbox = loss_bbox * scale_weight.unsqueeze(-1)
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(\
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        if scale_weight is not None:
            loss_giou = loss_giou * scale_weight
        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_p2_recall(self, outputs, targets, indices=None, num_boxes=None, **kwargs):
        """P2 center-recall heatmap loss.

        This is a proposal prior for query selection, not a hard routing
        target. It supervises all objects with a smooth scale-aware amplitude,
        so small objects receive stronger recall pressure while medium objects
        still leave a weak signal.
        """
        if 'p2_recall_logits' not in outputs:
            return {}

        logits = outputs['p2_recall_logits']
        _, _, h, w = logits.shape
        target = generate_p2_recall_gt(
            targets,
            h,
            w,
            small_threshold=self.sa_p2q_small_threshold,
            small_tau=self.sa_p2q_small_tau,
            alpha_min=self.sa_p2q_recall_alpha_min,
            sigma_scale=self.sa_p2q_recall_sigma_scale,
            device=logits.device,
        ).to(dtype=logits.dtype)

        pred = logits.sigmoid()
        pos_mask = (target > 1e-3).to(dtype=logits.dtype)
        neg_mask = 1. - pos_mask
        pos_weight = 1. + 4. * target
        neg_weight = (1. - target).pow(4.)

        bce = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction='none')
        pred_fp32 = pred.float()
        target_fp32 = target.float()
        pos_focal = (1. - pred_fp32).clamp_min(1e-4).pow(self.sa_p2q_recall_gamma)
        neg_focal = pred_fp32.clamp_min(1e-4).pow(self.sa_p2q_recall_gamma)

        pos_loss = bce * pos_focal * pos_weight.float() * pos_mask.float()
        neg_loss = bce * neg_focal * neg_weight.float() * neg_mask.float()

        pos_norm = pos_mask.flatten(1).sum(dim=1).clamp_min(1.)
        neg_norm = neg_mask.flatten(1).sum(dim=1).clamp_min(1.)
        loss = (pos_loss.flatten(1).sum(dim=1) / pos_norm +
                neg_loss.flatten(1).sum(dim=1) / neg_norm).mean()
        return {'loss_p2_recall': loss}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
            'p2_recall': self.loss_p2_recall,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Retrieve the matching between the outputs of the last layer and the targets
        matched = self.matcher(outputs_without_aux, targets)
        indices = matched['indices']

        # Compute all the requested losses
        losses = {}
        once_only_losses = ('p2_recall',)
        for loss in self.losses:
            # Auxiliary proposal losses are handled once at the end.
            if loss in once_only_losses:
                continue
            meta = self.get_loss_meta_info(loss, outputs, targets, indices)
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes, **meta)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if not self.share_matched_indices:
                    matched = self.matcher(aux_outputs, targets)
                    indices = matched['indices']
                for loss in self.losses:
                    if loss in once_only_losses:
                        continue
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of cdn auxiliary losses. For rtdetr
        if 'dn_aux_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']
            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                for loss in self.losses:
                    if loss in once_only_losses:
                        continue
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of encoder auxiliary losses. For rtdetr v2
        if 'enc_aux_outputs' in outputs:
            assert 'enc_meta' in outputs, ''
            class_agnostic = outputs['enc_meta']['class_agnostic']
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t['labels'] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                matched = self.matcher(aux_outputs, targets)
                indices = matched['indices']
                for loss in self.losses:
                    if loss in once_only_losses:
                        continue
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices)
                    l_dict = self.get_loss(loss, aux_outputs, enc_targets, indices, num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            if class_agnostic:
                self.num_classes = orig_num_classes

        aux_factor = self._sa_p2q_aux_factor()
        if 'p2_recall_logits' in outputs and 'loss_p2_recall' in self.weight_dict:
            l_dict = self.loss_p2_recall(outputs, targets)
            if 'loss_p2_recall' in l_dict:
                losses['loss_p2_recall'] = l_dict['loss_p2_recall'] * self.weight_dict['loss_p2_recall'] * aux_factor

        if self.training:
            self._sa_p2q_step += 1

        return losses

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None:
            return {}

        src_boxes = outputs['pred_boxes'][self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if self.boxes_weight_format == 'iou':
            iou, _ = box_iou(box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes))
            iou = torch.diag(iou)
        elif self.boxes_weight_format == 'giou':
            iou = torch.diag(generalized_box_iou(\
                box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)))
        else:
            raise AttributeError()

        if loss in ('boxes', ):
            meta = {'boxes_weight': iou}
        elif loss in ('vfl', ):
            meta = {'values': iou}
        else:
            meta = {}

        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices
        """
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device
        
        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                    torch.zeros(0, dtype=torch.int64,  device=device)))
        
        return dn_match_indices
