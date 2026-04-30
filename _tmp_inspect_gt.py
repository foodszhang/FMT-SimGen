#!/usr/bin/env python3
import numpy as np
import json

gt = np.load('data/default/samples/sample_0000/gt_voxels.npy')
print('gt_voxels shape:', gt.shape)
print('gt_voxels dtype:', gt.dtype)
print('max:', gt.max(), 'min:', gt.min())
unique = np.unique(gt)
print('unique values count:', len(unique))
print('first 20 unique:', unique[:20])

# Quick stats on non-zero
nz = gt > 0
print('\nNon-zero count:', nz.sum(), 'of', gt.size)
print('Non-zero max:', gt[nz].max() if nz.any() else 0)
print('Non-zero min:', gt[nz].min() if nz.any() else 0)

# Sample values
vals = gt[nz]
print('\nValue distribution of non-zero:')
for pct in [1, 5, 25, 50, 75, 95, 99]:
    print(f'  p{pct}: {np.percentile(vals, pct):.4f}')
