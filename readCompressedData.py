#!/usr/bin/env python2

import argparse
import numpy as np
import json
import h5py
import os

N_MOD = 2
VIS = 0
TXT = 1
MOD_NAMES = ['vis', 'txt']

PRECISION = [3, 4, 6, 9]
FEAT_PER_INT = [6, 4, 3, 2]
DECOMP_MASK = [pow(2, 10) - 1, pow(2, 14) - 1, pow(2, 16) - 1, pow(2, 20) - 1, pow(2, 30) - 1]
BIT_SHIFT = [[np.uint64(0), np.uint64(10), np.uint64(20),
              np.uint64(30), np.uint64(40), np.uint64(50)],
             [np.uint64(0), np.uint64(14), np.uint64(28), np.uint64(42)],
             [np.uint64(0), np.uint64(20), np.uint64(40)],
             [np.uint64(0), np.uint64(30)]]

MULTIPLIER = [np.uint64(1000), np.uint64(10000),
              np.uint64(1000000), np.uint64(1000000000)]

MAX_N_INT = np.uint64(15)

BIT_SHIFT_RATIO = [np.uint64(4), np.uint64(14), np.uint64(24),
                   np.uint64(34), np.uint64(44), np.uint64(54)]

BIT_SHIFT_RATIO_16 = [np.uint64(0), np.uint64(16), np.uint64(32),
                   np.uint64(48)]

FEAT_PER_INT_RATIO = 6
MULTIPLIER_RATIO = np.uint64(1000)
PRECISION_RATIO = 3

BIT_SHIFT_INIT_RATIO = np.uint64(54)
MULTIPLIER_INIT = np.uint64(pow(10, 16))
MASK_INIT = np.uint64(18014398509481983)

FEAT_PER_INT_RATIO_16 = 4
MULTIPLIER_RATIO_16 = np.uint64(50000)
#MULTIPLIER_RATIO_16 = np.uint64(1000)
BIT_SHIFT_INIT_RATIO_48 = np.uint64(48)
MULTIPLIER_INIT_48 = np.uint64(2 * pow(10, 14))
#MULTIPLIER_INIT_48 = np.uint64(pow(10, 10))
MASK_INIT_48 = np.uint64(281474976710655)

# .............................................................................
# decompress()
#
# Decompresses a compressed item's features in a modality
# .............................................................................
def decompress(idx, comp_init, comp_ids, comp_feat, p):
    decomp = []

    comp_init = np.uint64(comp_init)
    comp_ids = np.uint64(comp_ids)
    comp_feat = np.uint64(comp_feat)

    #feat_init_id = comp_init >> BIT_SHIFT_INIT_RATIO
    feat_init_id = comp_init >> BIT_SHIFT_INIT_RATIO_48
    #feat_init_score = (comp_init & MASK_INIT) / float(MULTIPLIER_INIT)
    feat_init_score = (comp_init & MASK_INIT_48) / float(MULTIPLIER_INIT_48)
    feat_score = feat_init_score
    decomp.append((int(feat_init_id), feat_init_score))
    for f_pos in range(FEAT_PER_INT_RATIO_16):
    #for f_pos in range(FEAT_PER_INT_RATIO):
        feat_i = (comp_ids >> BIT_SHIFT_RATIO_16[f_pos]) & np.uint64(DECOMP_MASK[p])
        #feat_i = (comp_ids >> BIT_SHIFT_RATIO[f_pos]) & np.uint64(DECOMP_MASK[p])
        feat_score *=\
            ((comp_feat >> BIT_SHIFT_RATIO_16[f_pos]) & np.uint64(DECOMP_MASK[p])) /\
            float(MULTIPLIER_RATIO_16)
            #((comp_feat >> BIT_SHIFT_RATIO[f_pos]) & np.uint64(DECOMP_MASK[p])) /\
            #float(MULTIPLIER_RATIO)

        decomp.append((int(feat_i), feat_score))

    print("Image: %d\t%u\t%u\t%u" % (idx, comp_init, comp_ids, comp_feat))
    print("Top Feature: %d\t%f" % (decomp[0][0], decomp[0][1]))
    print("Other Features: " + str(decomp[1:]))

    return decomp


parser = argparse.ArgumentParser()
parser.add_argument('feat_init', type=str)
parser.add_argument('feat_ids', type=str)
parser.add_argument('feat_ratios', type=str)
parser.add_argument('--output', type=str, default=None)
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=10)

args = parser.parse_args()
start = args.start
end = args.end

all_items = {}
with h5py.File(args.feat_init, 'r') as init:
    f_init = init['data'][start:end]
    with h5py.File(args.feat_ids, 'r') as ids:
        f_ids = ids['data'][start:end]
        with h5py.File(args.feat_ratios, 'r') as ratios:
            f_ratios = ratios['data'][start:end]
            for i in range(len(f_init)):
                d = decompress(i, f_init[i], f_ids[i], f_ratios[i], 2)
                all_items[i] = d

if (args.output is not None):
    with open(args.output, 'w') as f:
        json.dump(all_items, f)

