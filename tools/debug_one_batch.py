import os
import argparse
import torch
from mmengine import Config
from mmengine.registry import build_from_cfg
from mmengine.runner import load_checkpoint

# Minimal script to load one batch and run one forward pass deterministically.
# Usage: CUDA_LAUNCH_BLOCKING=1 python tools/debug_one_batch.py --config <config-file>


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--checkpoint', default=None, help='checkpoint (optional)')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # Force dataloader workers to 0 for deterministic stacktraces
    if 'train_dataloader' in cfg:
        cfg.train_dataloader.workers_per_gpu = 0
        cfg.train_dataloader.num_workers = 0
    if 'train' in cfg and isinstance(cfg.train, dict):
        # fallback
        cfg.data.train.num_workers = 0

    # Build dataset and dataloader using mmengine's API if available
    # We try to import mmdet3d utilities; fallback to mmengine construction.
    from mmdet3d.datasets import build_dataloader, build_dataset
    from mmdet3d.models import build_segmentor

    dataset_cfg = cfg.data.train
    dataset = build_dataset(dataset_cfg)

    # build dataloader (batch_size = 1 to minimize memory)
    loader_cfg = dict(
        dataset=dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=None,
    )

    # mmcv/mmengine build_dataloader helper expects some kwargs; use mmdet3d's wrapper
    dataloader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0)

    print('Dataloader built; getting one batch...')
    it = iter(dataloader)
    batch = next(it)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model
    model = build_segmentor(cfg.model)
    model.to(device)
    model.eval()

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint)

    # Move tensors in batch to device safely
    def to_device(x):
        if isinstance(x, dict):
            return {k: to_device(v) for k, v in x.items()}
        if isinstance(x, list):
            return [to_device(v) for v in x]
        if isinstance(x, tuple):
            return tuple(to_device(list(x)))
        if hasattr(x, 'to'):
            try:
                return x.to(device)
            except Exception:
                return x
        return x

    batch = to_device(batch)

    # Run a forward (loss mode) in a try/except to capture errors
    try:
        print('Running one forward...')
        with torch.no_grad():
            out = model(**batch, mode='loss')
        print('Forward succeeded; output keys:', out.keys() if isinstance(out, dict) else type(out))
    except Exception as e:
        print('Exception during forward:')
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
