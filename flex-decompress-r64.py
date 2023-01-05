#!/usr/bin/env python2

import argparse
import numpy as np
import json
import h5py
import os

# .............................................................................
# decompress()
#
# Decompresses a compressed item's features in a modality
# .............................................................................
def decompress(idx,
               comp_init,
               comp_ids,
               comp_feat,
               bit_shift_t,
               multiplier_t,
               decomp_mask_t,
               n_feat_per_int,
               bit_shift_ir,
               multiplier_ir,
               decomp_mask_ir):

    decomp = []
    comp_init = np.uint64(comp_init)
    comp_ids = np.uint64(comp_ids)
    comp_feat = np.uint64(comp_feat)

    feat_init_id = comp_init >> bit_shift_t
    feat_init_score = (comp_init & decomp_mask_t) / float(multiplier_t)
    feat_score = feat_init_score
    decomp.append((int(feat_init_id), feat_init_score))
    for f_pos in range(n_feat_per_int):
        feat_i = (comp_ids >> bit_shift_ir[f_pos]) & np.uint64(decomp_mask_ir)
        feat_score *= ((comp_feat >> bit_shift_ir[f_pos]) & np.uint64(decomp_mask_ir)) / float(multiplier_ir)

        decomp.append((int(feat_i), feat_score))

    #print("Image: %d\t%u\t%u\t%u" % (idx, comp_init, comp_ids, comp_feat))
    #print("Top Feature: %d\t%f" % (decomp[0][0], decomp[0][1]))
    #print("Other Features: " + str(decomp[1:]))

    return decomp


parser = argparse.ArgumentParser()
parser.add_argument('feat_init', type=str)
parser.add_argument('feat_ids', type=str)
parser.add_argument('feat_ratios', type=str)
parser.add_argument('--output', type=str, default=None)
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=10)
parser.add_argument('--bit_shift_t', type=int, default=48)
parser.add_argument('--multiplier_t', type=int, default=32)
parser.add_argument('--decomp_mask_t', type=int, default=32)
parser.add_argument('--n_feat_per_int', type=int, default=4)
parser.add_argument('--bit_shift_ir', type=int, default=16)
parser.add_argument('--decomp_mask_ir', type=int, default=16)
parser.add_argument('--multiplier_ir', type=int, default=16)
parser.add_argument('--top', type=int, default=-1)


args = parser.parse_args()
start = args.start
end = args.end

bit_shift_ir = []
padding = 64 % args.bit_shift_ir
for i in range(int(64/args.bit_shift_ir)):
    bit_shift_ir.append(np.uint64((i * args.bit_shift_ir) + padding))

all_items = {}
with h5py.File(args.feat_init, 'r') as init:
    f_init = init['data'][start:end]
    with h5py.File(args.feat_ids, 'r') as ids:
        f_ids = ids['data'][start:end]
        with h5py.File(args.feat_ratios, 'r') as ratios:
            f_ratios = ratios['data'][start:end]
            for i in range(len(f_init)):
                d = decompress(i, f_init[i], f_ids[i], f_ratios[i],
                               np.uint64(args.bit_shift_t),
                               pow(2,args.multiplier_t),
                               np.uint64(pow(2,args.decomp_mask_t)-1),
                               args.n_feat_per_int,
                               np.uint64(bit_shift_ir),
                               pow(2,args.multiplier_ir),
                               np.uint64(pow(2,args.decomp_mask_ir)-1))
                if args.top == -1:
                    all_items[i] = d
                else:
                    all_items[i] = d[:args.top]

if (args.output is not None):
    with open(args.output, 'w') as f:
        json.dump(all_items, f)

