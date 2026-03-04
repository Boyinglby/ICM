# Obtained from: https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License
# A copy of the license is available at resources/license_dacs

import kornia
import numpy as np
import torch
import torch.nn as nn


def strong_transform(param, data=None, target=None):
    assert ((data is not None) or (target is not None))
    data, target = one_mix(mask=param['mix'], data=data, target=target)
    data, target = color_jitter(
        color_jitter=param['color_jitter'],
        s=param['color_jitter_s'],
        p=param['color_jitter_p'],
        mean=param['mean'],
        std=param['std'],
        data=data,
        target=target)
    data, target = gaussian_blur(blur=param['blur'], data=data, target=target)
    return data, target


def get_mean_std(img_metas, dev):
    mean = [
        torch.as_tensor(img_metas[i]['img_norm_cfg']['mean'], device=dev)
        for i in range(len(img_metas))
    ]
    mean = torch.stack(mean).view(-1, 3, 1, 1)
    std = [
        torch.as_tensor(img_metas[i]['img_norm_cfg']['std'], device=dev)
        for i in range(len(img_metas))
    ]
    std = torch.stack(std).view(-1, 3, 1, 1)
    return mean, std


def denorm(img, mean, std):
    return img.mul(std).add(mean) / 255.0


def denorm_(img, mean, std):
    img.mul_(std).add_(mean).div_(255.0)


def renorm_(img, mean, std):
    img.mul_(255.0).sub_(mean).div_(std)


def color_jitter(color_jitter, mean, std, data=None, target=None, s=.25, p=.2):
    # s is the strength of colorjitter
    if not (data is None):
        if data.shape[1] == 3:
            if color_jitter > p:
                if isinstance(s, dict):
                    seq = nn.Sequential(kornia.augmentation.ColorJitter(**s))
                else:
                    seq = nn.Sequential(
                        kornia.augmentation.ColorJitter(
                            brightness=s, contrast=s, saturation=s, hue=s))
                denorm_(data, mean, std)
                data = seq(data)
                renorm_(data, mean, std)
    return data, target


def gaussian_blur(blur, data=None, target=None):
    if not (data is None):
        if data.shape[1] == 3:
            if blur > 0.5:
                sigma = np.random.uniform(0.15, 1.15)
                kernel_size_y = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[2]) - 0.5 +
                        np.ceil(0.1 * data.shape[2]) % 2))
                kernel_size_x = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[3]) - 0.5 +
                        np.ceil(0.1 * data.shape[3]) % 2))
                kernel_size = (kernel_size_y, kernel_size_x)
                seq = nn.Sequential(
                    kornia.filters.GaussianBlur2d(
                        kernel_size=kernel_size, sigma=(sigma, sigma)))
                data = seq(data)
    return data, target


def get_class_masks(labels):
    class_masks = []
    for label in labels:
        classes = torch.unique(labels)
        nclasses = classes.shape[0]
        class_choice = np.random.choice(
            nclasses, int((nclasses + nclasses % 2) / 2), replace=False)
        classes = classes[torch.Tensor(class_choice).long()]
        class_masks.append(generate_class_mask(label, classes).unsqueeze(0))
    return class_masks


# def get_class_masks(labels, exclude_classes=(0, 2, 8, 10), ignore_index=255):
#     """
#     Build a binary mask per sample selecting ALL pixels whose class is NOT in exclude_classes.
#     Important: returns mask shape (1,H,W) per sample (NOT (1,1,H,W)) to keep one_mix output 4D.

#     labels can be:
#       - Tensor (B,H,W) or (B,1,H,W)
#       - iterable of (H,W) or (1,H,W)
#     """
#     # Normalize input to iterable of per-sample label maps
#     if isinstance(labels, torch.Tensor):
#         if labels.dim() == 4:  # (B,1,H,W)
#             labels_iter = [labels[i, 0] for i in range(labels.size(0))]  # -> (H,W)
#         elif labels.dim() == 3:  # (B,H,W)
#             labels_iter = [labels[i] for i in range(labels.size(0))]      # -> (H,W)
#         else:
#             raise ValueError(f"labels must be (B,H,W) or (B,1,H,W), got {labels.shape}")
#     else:
#         labels_iter = []
#         for lab in labels:
#             if lab.dim() == 3 and lab.size(0) == 1:  # (1,H,W)
#                 labels_iter.append(lab[0])
#             elif lab.dim() == 2:  # (H,W)
#                 labels_iter.append(lab)
#             else:
#                 raise ValueError(f"Each label must be (H,W) or (1,H,W), got {lab.shape}")

#     class_masks = []
#     for label in labels_iter:
#         # label: (H,W)
#         valid = label.ne(ignore_index)

#         excl = torch.tensor(exclude_classes, device=label.device, dtype=label.dtype)  # (K,)
#         in_excl = (label.unsqueeze(-1) == excl.view(1, 1, -1)).any(dim=-1)            # (H,W)

#         keep = valid & (~in_excl)  # keep everything except excluded classes

#         # CRITICAL: return (1,H,W) so one_mix returns 4D image
#         class_masks.append(keep.float().unsqueeze(0))  # (1,H,W)

#     return class_masks


def generate_class_mask(label, classes):
    label, classes = torch.broadcast_tensors(label,
                                             classes.unsqueeze(1).unsqueeze(2))
    class_mask = label.eq(classes).sum(0, keepdims=True)
    return class_mask


def one_mix(mask, data=None, target=None):
    if mask is None:
        return data, target
    if not (data is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
        data = (stackedMask0 * data[0] +
                (1 - stackedMask0) * data[1]).unsqueeze(0)
    if not (target is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
        target = (stackedMask0 * target[0] +
                  (1 - stackedMask0) * target[1]).unsqueeze(0)
    return data, target
# def one_mix(mask, data=None, target=None):
#     if mask is None:
#         return data, target

#     # ---- data branch (expects B,C,H,W) ----
#     if data is not None:
#         d0, d1 = data[0], data[1]              # typically (C,H,W)
#         m0 = mask[0]                           # should be (1,H,W)

#         # Ensure data slices are (C,H,W)
#         if d0.dim() == 4:  # (1,C,H,W) -> (C,H,W)
#             d0 = d0.squeeze(0)
#             d1 = d1.squeeze(0)

#         # Ensure mask is (1,H,W)
#         if m0.dim() == 2:  # (H,W) -> (1,H,W)
#             m0 = m0.unsqueeze(0)

#         stackedMask0, _ = torch.broadcast_tensors(m0, d0)  # -> (C,H,W)
#         mixed = stackedMask0 * d0 + (1 - stackedMask0) * d1
#         data = mixed.unsqueeze(0)  # -> (1,C,H,W)

#     # ---- target branch (must end as B,1,H,W) ----
#     if target is not None:
#         t0, t1 = target[0], target[1]
#         m0 = mask[0]

#         # Accept both (H,W) and (1,H,W) incoming target slices
#         if t0.dim() == 2:          # (H,W) -> (1,H,W)
#             t0 = t0.unsqueeze(0)
#             t1 = t1.unsqueeze(0)
#         elif t0.dim() == 4:        # (1,1,H,W) -> (1,H,W)
#             t0 = t0.squeeze(0)
#             t1 = t1.squeeze(0)

#         # Ensure mask is (1,H,W)
#         if m0.dim() == 2:
#             m0 = m0.unsqueeze(0)

#         stackedMask0, _ = torch.broadcast_tensors(m0, t0)  # -> (1,H,W)
#         mixed = stackedMask0 * t0 + (1 - stackedMask0) * t1

#         # CRITICAL: return as (1,1,H,W)
#         target = mixed.unsqueeze(0)  # (1,1,H,W)

#     return data, target
