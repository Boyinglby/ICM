import random
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
from torch.nn import Module

from mmseg.models.uda.teacher_module import EMATeacher
from mmseg.models.utils.dacs_transforms import get_mean_std, strong_transform


class IntraClassMixConsistencyModule(Module):
    """

      Confusion-based intra-class swap (within same pseudo class) using clean logits:
         - Select most "confused" patches for each class (margin)

    """

    def __init__(self, require_teacher, cfg):
        super().__init__()

        self.source_only = cfg.get("source_only", False)
        self.max_iters = cfg["max_iters"]
        self.color_jitter_s = cfg["color_jitter_strength"]
        self.color_jitter_p = cfg["color_jitter_probability"]

        self.mask_mode = cfg["mask_mode"]
        self.mask_alpha = cfg["mask_alpha"]
        self.mask_pseudo_threshold = cfg["mask_pseudo_threshold"]
        self.mask_lambda = cfg["mask_lambda"]
        
        assert self.mask_mode in [
            "separate",
            "separatesrc",
            "separatetrg",
            "separateaug",
            "separatesrcaug",
            "separatetrgaug",
        ]


        self.stuff_classes = set([0, 1, 2, 3, 4, 8, 9, 10])
        self.ignore_index = int(cfg.get("ignore_index", 255))

        # mask out non-keep pixels in the image (mean fill)
        self.mask_out_image_nonkeep = cfg.get("mask_out_image_nonkeep", True)

        # Intra-swap
        self.enable_intra_swap = cfg.get("enable_intra_swap", True)
        self.swap_mode = cfg.get("swap_mode", "grid")  

        # Grid mode
        self.grid_n = cfg.get("swap_grid_n", 8)  # 8x8 patches
        self.min_class_patch_frac = cfg.get("min_class_patch_frac", 0.08)
        self.permute_ratio = cfg.get("permute_ratio", 1.0)


        # Intra-class mix consistency
        self.intra_mix_lambda = float(cfg.get("intra_mix_lambda", 0.5))
        self.intra_class_mix_consistency_weight = float(
            cfg.get("intra_class_mix_consistency_weight", 0.5)
        )
        self.intra_class_mix_consistency_temperature = float(
            cfg.get("intra_class_mix_consistency_temperature", 1.0)
        )

        # ----  confusion-based swap knobs ----
        self.enable_confusion_swap = cfg.get("enable_confusion_swap", True)
        self.confusion_metric = cfg.get("confusion_metric", "margin")  
        self.confusion_topk_frac = float(cfg.get("confusion_topk_frac", 0.5))
        self.confusion_pair_with_easy = cfg.get("confusion_pair_with_easy", True)

        # Teacher
        self.teacher = None
        if require_teacher or self.mask_alpha != "same" or self.mask_pseudo_threshold != "same":
            self.teacher = EMATeacher(use_mask_params=True, cfg=cfg)

        self.debug = False
        self.debug_output = {}

    def update_weights(self, model, iter):
        if self.teacher is not None:
            self.teacher.update_weights(model, iter)

    def update_debug_state(self):
        if self.teacher is not None:
            self.teacher.debug = self.debug

    def _sync_ignore_index_from_model(self, model):
        if hasattr(model, "decode_head") and hasattr(model.decode_head, "ignore_index"):
            self.ignore_index = int(model.decode_head.ignore_index)

    # -------------------------
    # Debug: logits -> confidence maps
    # -------------------------
    def _logits_to_confidence_map(self, logits: torch.Tensor, mode: str = "margin") -> torch.Tensor:
        """
        logits: (B,K,H,W)
        returns: (B,H,W) float ~[0,1]
        """
        prob = F.softmax(logits, dim=1)

        if mode == "maxprob":
            return prob.max(dim=1).values

        if mode == "entropy":
            eps = 1e-8
            ent = -(prob * (prob + eps).log()).sum(dim=1)  # (B,H,W)
            # optional normalization (not strictly needed for viz)
            return ent

        # default: margin
        top2 = torch.topk(prob, k=2, dim=1).values  # (B,2,H,W)
        return top2[:, 0] - top2[:, 1]

    # -------------------------
    # (A1) supervision masking
    # -------------------------
    def _stuff_only_classes(
        self,
        lbl: torch.Tensor,  # (B,1,H,W) long
        seg_weight: Optional[torch.Tensor] = None,  # (B,H,W) float or None
    ):
        if not self.stuff_classes:
            return lbl, seg_weight

        lbl = lbl.clone()
        keep_mask = torch.zeros_like(lbl, dtype=torch.bool)
        for cid in self.stuff_classes:
            keep_mask |= lbl == int(cid)

        lbl[~keep_mask] = self.ignore_index

        if seg_weight is not None:
            seg_weight = seg_weight.clone()
            seg_weight[~keep_mask.squeeze(1)] = 0.0

        return lbl, seg_weight

    # -------------------------
    # (A2) image masking (TOTAL)
    # -------------------------
    def _mask_image_nonkeep(
        self,
        img: torch.Tensor,  # (B,3,H,W) normalized
        lbl: torch.Tensor,  # (B,1,H,W) keep classes, others ignore_index
        fill: Optional[torch.Tensor] = None,  # (1,3,1,1) or None -> 0
    ):
        if (not self.mask_out_image_nonkeep) or (not self.stuff_classes):
            return img

        img = img.clone()
        keep = lbl != self.ignore_index  # (B,1,H,W)
        keep3 = keep.expand(-1, img.size(1), -1, -1)

        if fill is None:
            img[~keep3] = 0.0
        else:
            img[~keep3] = fill.expand_as(img)[~keep3]

        return img

    # -------------------------
    # Confusion scoring (per patch)
    # -------------------------
    def _patch_confusion_score(
        self,
        prob_patch: torch.Tensor,      # (K,ph,pw) probabilities
        mask_patch: torch.Tensor,      # (ph,pw) bool (pseudo==cid & optional conf_mask)
        cid: int,
    ) -> torch.Tensor:
        if mask_patch.sum() == 0:
            return torch.tensor(0.0, device=prob_patch.device)

        if self.confusion_metric == "1-p":
            conf_map = 1.0 - prob_patch[cid]
            return conf_map[mask_patch].mean()

        if self.confusion_metric == "entropy":
            eps = 1e-8
            ent = -(prob_patch * (prob_patch + eps).log()).sum(dim=0)  # (ph,pw)
            return ent[mask_patch].mean()

        # default: margin confusion = 1 - (p1 - p2)
        top2 = torch.topk(prob_patch, k=2, dim=0).values  # (2,ph,pw)
        margin = top2[0] - top2[1]
        conf_map = 1.0 - margin
        return conf_map[mask_patch].mean()

    # ----------------------------------
    # (B) intra-swap: grid permutation (confusion-guided)
    # ----------------------------------
    def _grid_permute_same_class(
        self,
        img: torch.Tensor,
        lbl: torch.Tensor,
        seg_weight: Optional[torch.Tensor] = None,
        conf_mask: Optional[torch.Tensor] = None,
        clean_logits: Optional[torch.Tensor] = None,  # (B,K,H,W) from clean forward
    ):
        if not self.stuff_classes:
            return img, lbl, seg_weight

        B, C, H, W = img.shape
        out_img = img.clone()
        out_lbl = lbl
        out_w = seg_weight.clone() if seg_weight is not None else None

        n = int(self.grid_n)
        ph = H // n
        pw = W // n
        if ph <= 0 or pw <= 0:
            return out_img, out_lbl, out_w

        for b in range(B):
            prob_b = None
            if self.enable_confusion_swap and (clean_logits is not None):
                # (K,H,W)
                prob_b = F.softmax(clean_logits[b], dim=0)

            for cid in self.stuff_classes:
                cid = int(cid)

                # Collect eligible patches with optional confusion score
                eligible: List[Tuple[int, int, int, int, torch.Tensor]] = []
                for gy in range(n):
                    y1, y2 = gy * ph, (gy + 1) * ph
                    for gx in range(n):
                        x1, x2 = gx * pw, (gx + 1) * pw

                        patch_lbl = out_lbl[b, 0, y1:y2, x1:x2]
                        m = patch_lbl == cid
                        if conf_mask is not None:
                            m = m & conf_mask[b, y1:y2, x1:x2]

                        if float(m.float().mean()) >= float(self.min_class_patch_frac):
                            if prob_b is not None:
                                prob_patch = prob_b[:, y1:y2, x1:x2]
                                score = self._patch_confusion_score(prob_patch, m, cid)
                            else:
                                score = torch.tensor(0.0, device=img.device)
                            eligible.append((y1, y2, x1, x2, score))

                if len(eligible) < 2:
                    continue

                # Optional subsample eligible
                if float(self.permute_ratio) < 1.0:
                    ksub = max(2, int(len(eligible) * float(self.permute_ratio)))
                    idx = torch.randperm(len(eligible), device=img.device)[:ksub].tolist()
                    eligible = [eligible[i] for i in idx]
                    if len(eligible) < 2:
                        continue

                # Choose src/dst boxes
                if prob_b is not None:
                    scores = torch.stack([e[4] for e in eligible])  # (N,)
                    N = len(eligible)
                    k = max(2, int(N * float(self.confusion_topk_frac)))
                    k = min(k, N)

                    idx_sorted = torch.argsort(scores, descending=True)
                    hard_idx = idx_sorted[:k].tolist()

                    if self.confusion_pair_with_easy and N >= 4:
                        easy_idx = idx_sorted[-k:].tolist()
                        hard = [eligible[i] for i in hard_idx]
                        easy = [eligible[i] for i in easy_idx]
                        src_boxes = [(y1, y2, x1, x2) for (y1, y2, x1, x2, _) in easy]
                        dst_boxes = [(y1, y2, x1, x2) for (y1, y2, x1, x2, _) in hard]
                        mlen = min(len(src_boxes), len(dst_boxes))
                        src_boxes, dst_boxes = src_boxes[:mlen], dst_boxes[:mlen]
                    else:
                        hard = [eligible[i] for i in hard_idx]
                        src_boxes = [(y1, y2, x1, x2) for (y1, y2, x1, x2, _) in hard]
                        perm = torch.randperm(len(src_boxes), device=img.device).tolist()
                        dst_boxes = [src_boxes[i] for i in perm]
                else:
                    src_boxes = [(y1, y2, x1, x2) for (y1, y2, x1, x2, _) in eligible]
                    perm = torch.randperm(len(src_boxes), device=img.device).tolist()
                    dst_boxes = [src_boxes[i] for i in perm]

                if len(src_boxes) < 2:
                    continue

                # Cache source patches
                src_img_patches = [
                    out_img[b, :, y1:y2, x1:x2].clone() for (y1, y2, x1, x2) in src_boxes
                ]
                src_w_patches = None
                if out_w is not None:
                    src_w_patches = [
                        out_w[b, y1:y2, x1:x2].clone() for (y1, y2, x1, x2) in src_boxes
                    ]

                # Paste into destinations (only where both src and dst are class==cid [+ conf])
                for i, ((sy1, sy2, sx1, sx2), (dy1, dy2, dx1, dx2)) in enumerate(
                    zip(src_boxes, dst_boxes)
                ):
                    src_patch = src_img_patches[i]

                    src_m = out_lbl[b, 0, sy1:sy2, sx1:sx2] == cid
                    dst_m = out_lbl[b, 0, dy1:dy2, dx1:dx2] == cid
                    if conf_mask is not None:
                        src_m = src_m & conf_mask[b, sy1:sy2, sx1:sx2]
                        dst_m = dst_m & conf_mask[b, dy1:dy2, dx1:dx2]

                    swap_m = src_m & dst_m
                    if swap_m.sum() == 0:
                        continue

                    sm = swap_m.unsqueeze(0).expand(C, -1, -1)
                    out_img[b, :, dy1:dy2, dx1:dx2][sm] = src_patch[sm]

                    if out_w is not None and src_w_patches is not None:
                        src_w = src_w_patches[i]
                        out_w[b, dy1:dy2, dx1:dx2][swap_m] = src_w[swap_m]

        return out_img, out_lbl, out_w

    def _apply_intra_swap(self, img, lbl, seg_weight, clean_logits=None):
        if (not self.enable_intra_swap) or (not self.stuff_classes):
            return img, lbl, seg_weight

        conf_mask = None

        if self.swap_mode == "grid":
            return self._grid_permute_same_class(
                img, lbl, seg_weight, conf_mask=conf_mask, clean_logits=clean_logits
            )

        raise ValueError(f"Unknown swap_mode={self.swap_mode}. Use 'grid'.")

    # -------------------------
    # (C) Intra-class mix consistency loss
    # -------------------------
    def _intra_class_mix_consistency_loss(
        self,
        mixed_logits: torch.Tensor,  # (B,K,H,W)
        clean_logits: torch.Tensor,  # (B,K,H,W)
        keep_mask: torch.Tensor,     # (B,H,W) bool
        temperature: float = 1.0,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        mixed_logp = F.log_softmax(mixed_logits / temperature, dim=1)
        clean_p = F.softmax(clean_logits / temperature, dim=1)

        kl = F.kl_div(mixed_logp, clean_p, reduction="none").sum(dim=1)  # (B,H,W)

        keep_mask_f = keep_mask.float()
        denom = keep_mask_f.sum().clamp_min(eps)
        return (kl * keep_mask_f).sum() / denom

    # -------------------------
    # Main forward
    # -------------------------
    def __call__(
        self,
        model,
        img,
        img_metas,
        gt_semantic_seg,
        target_img,
        target_img_metas,
        valid_pseudo_mask,
        pseudo_label=None,
        pseudo_weight=None,
    ):
        self.update_debug_state()
        self.debug_output = {}
        model.debug_output = {}

        self._sync_ignore_index_from_model(model)

        dev = img.device
        means, stds = get_mean_std(img_metas, dev)
        fill_mean = means[0].view(1, 3, 1, 1)

        # --- teacher pseudo labels for target, if needed ---
        if not self.source_only:
            if self.teacher is None:
                assert self.mask_alpha == "same"
                assert self.mask_pseudo_threshold == "same"
                assert pseudo_label is not None
                assert pseudo_weight is not None
                masked_plabel = pseudo_label
                masked_pweight = pseudo_weight
            else:
                masked_plabel, masked_pweight = self.teacher(
                    target_img, target_img_metas, valid_pseudo_mask
                )

            if self.debug:
                self.debug_output["Mask Teacher"] = {
                    "Img": target_img.detach(),
                    "Pseudo Label": masked_plabel.detach().cpu().numpy(),
                    "Pseudo Weight": masked_pweight.detach().cpu().numpy(),
                    # NEW: treat pweight as a confidence-like map for viz
                    "Conf (pweight)": masked_pweight.detach().cpu().numpy(),
                }

        # --- Build training batch for this module (and aligned clean batch) ---
        if self.source_only:
            train_img = img
            train_lbl = gt_semantic_seg
            train_seg_weight = None
            train_metas = img_metas

            clean_img = img
            clean_lbl_full = gt_semantic_seg
            clean_metas = img_metas

        elif self.mask_mode in ["separate", "separateaug"]:
            assert img.shape[0] == 2
            train_img = torch.stack([img[0], target_img[0]])
            train_lbl = torch.stack([gt_semantic_seg[0], masked_plabel[0].unsqueeze(0)])
            gt_pixel_weight = torch.ones(masked_pweight[0].shape, device=dev)
            train_seg_weight = torch.stack([gt_pixel_weight, masked_pweight[0]])
            train_metas = img_metas

            clean_img = torch.stack([img[0], target_img[0]])
            clean_lbl_full = torch.stack([gt_semantic_seg[0], masked_plabel[0].unsqueeze(0)])
            clean_metas = img_metas

        elif self.mask_mode in ["separatesrc", "separatesrcaug"]:
            train_img = img
            train_lbl = gt_semantic_seg
            train_seg_weight = None
            train_metas = img_metas

            clean_img = img
            clean_lbl_full = gt_semantic_seg
            clean_metas = img_metas

        elif self.mask_mode in ["separatetrg", "separatetrgaug"]:
            train_img = target_img
            train_lbl = masked_plabel.unsqueeze(1)
            train_seg_weight = masked_pweight
            train_metas = target_img_metas

            clean_img = target_img
            clean_lbl_full = masked_plabel.unsqueeze(1)
            clean_metas = target_img_metas

        else:
            raise NotImplementedError(self.mask_mode)

        # --- (A) supervision masking: apply loss to stuff_classes ---
        train_lbl, train_seg_weight = self._stuff_only_classes(train_lbl, train_seg_weight)


        # ---  compute clean logits (used for confusion swap + debug + consistency) ---
        with torch.no_grad():
            clean_logits = model.encode_decode(clean_img, clean_metas)  # (B,K,H,W)

        if self.debug:
            conf_max = self._logits_to_confidence_map(clean_logits, mode="maxprob")
            conf_margin = self._logits_to_confidence_map(clean_logits, mode="margin")

            self.debug_output.setdefault("Clean", {})
            self.debug_output["Clean"]["Img"] = clean_img.detach()
            self.debug_output["Clean"]["Pred"] = clean_logits.argmax(dim=1).detach().cpu().numpy()
            self.debug_output["Clean"]["Conf (maxprob)"] = conf_max.detach().cpu().numpy()
            self.debug_output["Clean"]["Conf (margin)"] = conf_margin.detach().cpu().numpy()

        # --- (B) intra-swap among keep classes (confusion-guided if enabled) ---
        train_img, train_lbl, train_seg_weight = self._apply_intra_swap(
            train_img, train_lbl, train_seg_weight, clean_logits=clean_logits
        )

        # --- Color augmentation (optional) ---
        if "aug" in self.mask_mode:
            strong_parameters = {
                "mix": None,
                "color_jitter": random.uniform(0, 1),
                "color_jitter_s": self.color_jitter_s,
                "color_jitter_p": self.color_jitter_p,
                "blur": random.uniform(0, 1),
                "mean": means[0].unsqueeze(0),
                "std": stds[0].unsqueeze(0),
            }
            train_img, _ = strong_transform(strong_parameters, data=train_img.clone())

        # ==========================================================
        # (C) Intra-class mix consistency: compare ONLY stuff pixels
        # ==========================================================
        loss_mix_cons = None
        w_cons = float(self.intra_class_mix_consistency_weight)
        if w_cons > 0:
            keep_mask = train_lbl[:, 0] != self.ignore_index  # (B,H,W)

            mixed_logits = model.encode_decode(train_img, train_metas)  # grads flow
            loss_mix_cons = self._intra_class_mix_consistency_loss(
                mixed_logits=mixed_logits,
                clean_logits=clean_logits,  # no-grad
                keep_mask=keep_mask,
                temperature=float(self.intra_class_mix_consistency_temperature),
            )

        # --- Train supervised seg on masked/swapped/aug batch (keep-only labels) ---
        intra_mix_loss = model.forward_train(
            train_img,
            train_metas,
            train_lbl,
            seg_weight=train_seg_weight,
        )
        if self.intra_mix_lambda != 1:
            intra_mix_loss["decode.loss_seg"] *= self.intra_mix_lambda

        if loss_mix_cons is not None:
            intra_mix_loss["decode.loss_intra_class_mix_consistency"] = w_cons * loss_mix_cons

        if self.debug:
            self.debug_output["Intra_mix"] = model.debug_output
            if train_seg_weight is not None:
                self.debug_output["Intra_mix"]["PL Weight"] = train_seg_weight.detach().cpu().numpy()

        return intra_mix_loss
