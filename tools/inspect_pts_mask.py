import argparse
import pickle
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--pkl', required=True, help='path to nuscenes infos pkl')
parser.add_argument('--idx', type=int, default=0, help='which sample index to inspect')
args = parser.parse_args()

with open(args.pkl, 'rb') as f:
    infos = pickle.load(f)

print('Total samples in pkl:', len(infos))
info = infos[args.idx]

print('\nSample index:', args.idx)
for k, v in info.items():
    if k == 'pts_semantic_mask' or 'mask' in k:
        mask = None
        try:
            mask = v
            if isinstance(mask, (list, tuple)):
                mask = np.array(mask)
        except Exception as e:
            print('Could not coerce mask to array for key', k, 'error:', e)
            continue
        print('Key:', k)
        print(' dtype:', mask.dtype)
        try:
            print(' min:', mask.min(), ' max:', mask.max())
            unique = np.unique(mask)
            print(' unique count:', len(unique))
            print(' some uniques:', unique[:50])
        except Exception as e:
            print('Error inspecting mask values:', e)
    else:
        # print keys and small repr
        s = str(v)
        if len(s) > 200:
            s = s[:200] + '...'
        print(f"{k}: {s}")

print('\nDone')
