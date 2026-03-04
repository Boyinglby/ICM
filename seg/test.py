# 


import argparse
import os

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, wrap_fp16_model)
from mmcv.utils import DictAction



from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor


import re
import torch

def _strip_any_prefix(k: str):
    # strip known wrappers repeatedly
    changed = True
    while changed:
        changed = False
        for p in ("module.", "model.", "_orig_mod."):
            if k.startswith(p):
                k = k[len(p):]
                changed = True
    return k

def _detect_backbone_prefix(model):
    """Detect what prefix the model expects for the timm backbone params."""
    mkeys = list(model.state_dict().keys())

    # Most likely cases:
    candidates = [
        "backbone.model.",       # your TimmBackbone wrapper (self.model)
        "backbone.",             # plain backbone
        "backbone.model.model.", # sometimes wrappers nest again
    ]
    for c in candidates:
        if any(k.startswith(c) for k in mkeys):
            return c

    # Fallback: find any key containing '.stem_0.' under backbone
    for k in mkeys:
        if k.startswith("backbone.") and ".stem_0." in k:
            # return the prefix up to stem_0
            i = k.find("stem_0.")
            return k[:i]

    # default
    return "backbone.model."

def load_ckpt_with_remap(model, ckpt_path, map_location="cpu", use_ema=False, strict=True):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    expected_bb = _detect_backbone_prefix(model)
    # expected_bb ends with "backbone." or "backbone.model." etc.
    # We will map checkpoint backbone keys to this.

    new_sd = {}

    for k, v in sd.items():
        k = _strip_any_prefix(k)

        # Choose EMA or non-EMA source
        if use_ema:
            if k.startswith("ema_backbone."):
                kk = expected_bb + k[len("ema_backbone."):]
                new_sd[kk] = v
                continue
            if k.startswith("ema_decode_head."):
                new_sd["decode_head." + k[len("ema_decode_head."):]] = v
                continue
            if k.startswith("ema_auxiliary_head."):
                new_sd["auxiliary_head." + k[len("ema_auxiliary_head."):]] = v
                continue
            continue  # ignore non-EMA keys entirely
        else:
            if k.startswith("ema_"):
                continue  # drop EMA keys

        # ---- Backbone remap ----
        if k.startswith("backbone."):
            kk = expected_bb + k[len("backbone."):]
            new_sd[kk] = v
            continue

        # Everything else (decode_head, neck, etc.)
        new_sd[k] = v

    incompatible = model.load_state_dict(new_sd, strict=strict)
    missing = getattr(incompatible, "missing_keys", None)
    unexpected = getattr(incompatible, "unexpected_keys", None)
    if missing is None or unexpected is None:
        missing, unexpected = incompatible

    # Print a small sanity check so you can confirm it’s working
    print(f"[load_ckpt_with_remap] expected backbone prefix: {expected_bb}")
    print(f"[load_ckpt_with_remap] loaded keys: {len(new_sd)}")
    if missing:
        print(f"[load_ckpt_with_remap] missing ({len(missing)}), first 10:", missing[:10])
    if unexpected:
        print(f"[load_ckpt_with_remap] unexpected ({len(unexpected)}), first 10:", unexpected[:10])

    return ckpt if isinstance(ckpt, dict) else {"state_dict": sd}

def update_legacy_cfg(cfg):
    # The saved json config does not differentiate between list and tuple
    cfg.data.test.pipeline[1]['img_scale'] = tuple(
        cfg.data.test.pipeline[1]['img_scale'])
    cfg.data.val.pipeline[1]['img_scale'] = tuple(
        cfg.data.val.pipeline[1]['img_scale'])
    # Support legacy checkpoints
    if cfg.model.decode_head.type == 'UniHead':
        cfg.model.decode_head.type = 'DAFormerHead'
        cfg.model.decode_head.decoder_params.fusion_cfg.pop('fusion', None)
    if cfg.model.type == 'MultiResEncoderDecoder':
        cfg.model.type = 'HRDAEncoderDecoder'
    if cfg.model.decode_head.type == 'MultiResAttentionWrapper':
        cfg.model.decode_head.type = 'HRDAHead'
    cfg.model.backbone.pop('ema_drop_path_rate', None)
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument(
        '--inference-mode',
        choices=[
            'same',
            'whole',
            'slide',
        ],
        default='same',
        help='Inference mode.')
    parser.add_argument(
        '--test-set',
        action='store_true',
        help='Run inference on the test set')
    parser.add_argument(
        '--hrda-out',
        choices=['', 'LR', 'HR', 'ATT'],
        default='',
        help='Extract LR and HR predictions from HRDA architecture.')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument('--local_rank', type=int, default=0)

    # NEW: optionally load EMA weights (ema_backbone / ema_decode_head) into normal modules.
    parser.add_argument(
        '--use-ema',
        action='store_true',
        help='Load EMA weights from checkpoint (ema_backbone/ema_decode_head) '
             'into the normal model modules. If not set, EMA keys are ignored.')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def _strip_prefix(s: str, prefix: str) -> str:
    return s[len(prefix):] if s.startswith(prefix) else s


def load_checkpoint_robust(model, ckpt_path, map_location='cpu', use_ema=False, strict=True):
    """
    Robust loader for MIC/mmseg 0.x style checkpoints with optional EMA.
    Fixes:
      - drops EMA keys by default
      - optional EMA remap to normal modules
      - strips module. and model. prefixes
      - maps backbone.* -> backbone.model.* for TimmBackbone wrapper
    Returns: checkpoint dict (for meta), missing_keys, unexpected_keys
    """
    ckpt = torch.load(ckpt_path, map_location=map_location)
    state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt

    new_sd = {}

    for k, v in state_dict.items():
        # strip common wrappers
        k = _strip_prefix(k, 'module.')
        k = _strip_prefix(k, 'model.')

        if use_ema:
            # Only consume EMA params; ignore non-EMA params
            if k.startswith('ema_backbone.'):
                kk = 'backbone.' + k[len('ema_backbone.'):]
                # TimmBackbone keeps timm model under backbone.model
                if kk.startswith('backbone.') and not kk.startswith('backbone.model.'):
                    kk = 'backbone.model.' + kk[len('backbone.'):]
                new_sd[kk] = v
                continue

            # Many mmseg models name the main head as "decode_head"
            # If your segmentor uses a different attribute name, adjust here.
            if k.startswith('ema_decode_head.'):
                kk = 'decode_head.' + k[len('ema_decode_head.'):]
                new_sd[kk] = v
                continue

            # Optional: some setups might save EMA for auxiliary head
            if k.startswith('ema_auxiliary_head.'):
                kk = 'auxiliary_head.' + k[len('ema_auxiliary_head.'):]
                new_sd[kk] = v
                continue

            # ignore any other keys when using EMA
            continue

        else:
            # drop EMA keys entirely
            if k.startswith('ema_'):
                continue

            # Fix your reported mismatch:
            # checkpoint has backbone.<timm_keys>, model expects backbone.model.<timm_keys>
            if k.startswith('backbone.') and not k.startswith('backbone.model.'):
                k = 'backbone.model.' + k[len('backbone.'):]

            new_sd[k] = v

    incompatible = model.load_state_dict(new_sd, strict=strict)

    # torch returns an _IncompatibleKeys object in newer versions;
    # in older versions, it might be a tuple.
    missing_keys = getattr(incompatible, 'missing_keys', None)
    unexpected_keys = getattr(incompatible, 'unexpected_keys', None)
    if missing_keys is None or unexpected_keys is None:
        missing_keys, unexpected_keys = incompatible

    return ckpt if isinstance(ckpt, dict) else {'state_dict': state_dict}, missing_keys, unexpected_keys


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    cfg = update_legacy_cfg(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    if args.inference_mode == 'same':
        # Use pre-defined inference mode
        pass
    elif args.inference_mode == 'whole':
        print('Force whole inference.')
        cfg.model.test_cfg.mode = 'whole'
    elif args.inference_mode == 'slide':
        print('Force slide inference.')
        cfg.model.test_cfg.mode = 'slide'
        crsize = cfg.data.train.get('sync_crop_size', cfg.crop_size)
        cfg.model.test_cfg.crop_size = crsize
        cfg.model.test_cfg.stride = [int(e / 2) for e in crsize]
        cfg.model.test_cfg.batched_slide = True
    else:
        raise NotImplementedError(args.inference_mode)

    if args.hrda_out == 'LR':
        cfg['model']['decode_head']['fixed_attention'] = 0.0
    elif args.hrda_out == 'HR':
        cfg['model']['decode_head']['fixed_attention'] = 1.0
    elif args.hrda_out == 'ATT':
        cfg['model']['decode_head']['debug_output_attention'] = True
    elif args.hrda_out == '':
        pass
    else:
        raise NotImplementedError(args.hrda_out)

    if args.test_set:
        for k in cfg.data.test:
            if isinstance(cfg.data.test[k], str):
                cfg.data.test[k] = cfg.data.test[k].replace('val', 'test')

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    checkpoint, missing_keys, unexpected_keys = load_checkpoint_robust(
        model,
        args.checkpoint,
        map_location='cpu',
        use_ema=args.use_ema,
        strict=True  # keep strict for full-model checkpoints you trained
    )

    # Print a short summary (optional but helpful)
    if missing_keys:
        print(f'[Checkpoint] Missing keys ({len(missing_keys)}). First 20:\n  ' +
              '\n  '.join(missing_keys[:20]))
    if unexpected_keys:
        print(f'[Checkpoint] Unexpected keys ({len(unexpected_keys)}). First 20:\n  ' +
              '\n  '.join(unexpected_keys[:20]))

    # meta
    if isinstance(checkpoint, dict) and 'meta' in checkpoint and isinstance(checkpoint['meta'], dict):
        meta = checkpoint['meta']
    else:
        meta = {}

    if 'CLASSES' in meta:
        model.CLASSES = meta['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES

    if 'PALETTE' in meta:
        model.PALETTE = meta['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE

    efficient_test = False
    if args.eval_options is not None:
        efficient_test = args.eval_options.get('efficient_test', False)

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  efficient_test, args.opacity)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect, efficient_test)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            res = dataset.evaluate(outputs, args.eval, **kwargs)
            print([k for k, v in res.items() if 'IoU' in k])
            print([round(v * 100, 1) for k, v in res.items() if 'IoU' in k])


if __name__ == '__main__':
    main()
