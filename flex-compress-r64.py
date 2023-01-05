#!/usr/bin/env python2

'''
===============================================================================
    +++ flex-compress-r64.py +++

    Author: Omar S. Khan, IT University of Copenhagen, 2020

    This is a modified script of compress-iota-i64.py focusing on flexibility
    and ease of use.

    Default values create a 16-bit representation:
        Top Feature:        16-bit/48-bit
        Next Feature Ids:   16-bit/16-bit/16-bit/16-bit
        Feature Ratios:     16-bit/16-bit/16-bit/16-bit

    Original description:

    Compresses the visual and text semantic features of a collection into the
    efficient iota-I64 representation used by Blackthorn.

    Requires the following conditions:
    1) The features for each modality are stored in a IxF matrix where each row
       corresponds to an item and each column to a feature.
    2) Each feature matrix is accompanied with an array of length I with item
       IDs in integer format.
    3) Both feature matrices and both item ID arrays are stored in HDF5 format.
===============================================================================
'''
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#  IMPORTS/VARIABLES
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

import argparse
import h5py
from math import sqrt
from math import floor
import multiprocessing as mp
import numpy as np
import os
from scipy import stats
from sys import stdout
import time
import platform

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#  FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# .............................................................................
# timestamp()
#
# Formatted timestamp for console output.
# .............................................................................
def timestamp():
    return "[" + str(time.strftime("%d %b, %H:%M:%S")) + "]"


# .............................................................................
# compress_ratio()
#
# Data compression thread.
# .............................................................................
def compress_ratio(p_id,
                   n_processes,
                   iota,
                   out_file_prefix,
                   feat_path,
                   feat_h5_name,
                   out_dir,
                   tfidf,
                   idf_path,
                   threshold,
                   mean_path,
                   std_path,
                   normalize,
                   bit_shift_t,
                   multiplier_t,
                   n_feat_per_int,
                   bit_shift_ir,
                   multiplier_ir):
    out_sig = "{iota=%s} <%s>" % (iota, p_id)

    print("%s %s Compression thread started." % (timestamp(), out_sig))

    n_feat_max = iota*n_feat_per_int + 1

    # Load the ID and feature files
    print("%s %s Loading features!" % (timestamp(), out_sig))
    feat_f = h5py.File(feat_path, 'r')
    feat = feat_f[feat_h5_name]
    n_feat = len(feat[0, :])
    print("%s %s Loaded features!" % (timestamp(), out_sig))

    # If using IDF, load the IDF values for the features
    if tfidf:
        idf_f = h5py.File(idf_path, 'r')
        idf = idf_f["data"]

    # If thresholding, load the means and stds
    if threshold:
        mean_f = h5py.File(mean_path, 'r')
        std_f = h5py.File(std_path, 'r')

        mean = mean_f["data"]
        std = std_f["data"]

        thresholds = np.array(mean) + np.array(std)

    # Prepare the containers for the compressed features
    comp_ids = []
    comp_feat = []
    comp_init = []

    # Calculate the range that will be processed by the thread
    n_all = len(feat)

    i_start = int(p_id * (n_all / n_processes))
    if p_id == n_processes - 1:
        i_end = n_all
    else:
        i_end = int((p_id + 1) * (n_all / n_processes))

    # Establish the number of items to be processed and init the counter
    n = i_end - i_start
    item_ctr = 0

    # .................................................................
    # Go over all items in the modality
    print("i_start is: %d" % i_start)
    print("i_end is: %d" % i_end)
    for i in range(i_start, i_end):
        feat_vec = feat[i, :]
        # Sort the features - they will be in ASCENDING order
        if tfidf:
            tf_idf_feat = np.multiply(feat_vec, idf)
            feat_top = [np.uint64(x) for x in np.argsort(tf_idf_feat)]
        elif threshold:
            thresh_feat = np.empty(shape=(n_feat,))

            for f in range(n_feat):
                if feat_vec[f] >= thresholds[f]:
                    thresh_feat[f] = feat_vec[f]
                else:
                    thresh_feat[f] = 0.0

            feat_top = [np.uint64(x) for x in np.argsort(thresh_feat)]
        else:
            feat_top = [np.uint64(x) for x in np.argsort(feat_vec)]

        # Prepare the containers for the compressed features
        comp_i = [np.uint64(0) for x in range(iota)]
        comp_f = [np.uint64(0) for x in range(iota)]

        # Normalize the features that will be compressed
        if normalize:
            l2norm = 0.0

            for f in range(n_feat_max):
                l2norm += pow(feat_vec[feat_top[-f - 1]], 2)
            try:
                l2norm = 1 / sqrt(l2norm)
            except ZeroDivisionError:
                l2norm = 1.0

            for f in range(n_feat_max):
                feat_vec[feat_top[-f - 1]] *= l2norm

        # Compress the initial feature
        comp_if = np.uint64(feat_top[-1]) << bit_shift_t
        comp_if |= np.uint64(round(feat_vec[feat_top[-1]] * multiplier_t))

        n_encoded = 0
        for f in range(1, n_feat_max):
            f_int = floor((f - 1) / n_feat_per_int)
            f_pos = (f - 1) % n_feat_per_int

            f_ratio = np.uint64(round((feat_vec[feat_top[-f - 1]] / feat_vec[feat_top[-(f - 1) - 1]]) * multiplier_ir, 3))

            comp_i[f_int] |= feat_top[-f - 1] << bit_shift_ir[f_pos]
            comp_f[f_int] |= f_ratio << bit_shift_ir[f_pos]

        for int_i in range(iota):
            comp_ids.append(comp_i[int_i])
            comp_feat.append(comp_f[int_i])
        comp_init.append(comp_if)

        # Output the progress to stdout
        item_ctr += 1

        if item_ctr % 1000 == 0:
            print("%s %s (%s) %s/%s items compressed." % (timestamp(),
                                                          out_sig,
                                                          out_file_prefix,
                                                          item_ctr,
                                                          n))
    print("about to close all before write")
    feat_f.close()
    print("have closed feat_f")
    if tfidf:
        idf_f.close()
    if threshold:
        mean_f.close()
        std_f.close()

    # .................................................................
    print("%s %s (%s) Compression finished, writing..." % (timestamp(),
                                                           out_sig,
                                                           out_file_prefix))
    # Write the data to a temporary HDF5 file
    ids_path = os.path.join(out_dir, "%s_ids.h5.tmp%s" % (out_file_prefix, p_id))
    ratios_path = os.path.join(out_dir, "%s_ratios.h5.tmp%s" % (out_file_prefix, p_id))
    top_path = os.path.join(out_dir, "%s_top.h5.tmp%s" % (out_file_prefix, p_id))
    print("ids path is %s." % ids_path)
    print("feat path is %s." % ratios_path)
    print("top_path is %s" % top_path)

    with h5py.File(ids_path, 'w') as f:
        f.create_dataset('data', data=comp_ids, dtype=np.uint64)

    with h5py.File(ratios_path, 'w') as f:
        f.create_dataset('data', data=comp_feat, dtype=np.uint64)

    with h5py.File(top_path, 'w') as f:
        f.create_dataset('data', data=comp_init, dtype=np.uint64)

    # .....................................................................
    print("%s %s Data written." % (timestamp(), out_sig))


# .............................................................................
# compress_control()
#
# Controls the compression of the data, spawning the processes that perform
# the compression and collecting the results
# .............................................................................
def compress_control(iota,
                     out_file_prefix,
                     out_dir,
                     feat_path,
                     feat_h5_name,
                     n_processes,
                     tfidf,
                     idf_path,
                     threshold,
                     mean_path,
                     std_path,
                     normalize,
                     bit_shift_t,
                     multiplier_t,
                     n_feat_per_int,
                     bit_shift_ir,
                     multiplier_ir):
    out_sig = "{iota=%s}" % (iota)

    print("%s %s +++ COMPRESSION STARTED +++" % (timestamp(), out_sig))
    time_start = time.time()

    if platform.system() == 'Darwin':
        mp.set_start_method('fork')

    # Start the workers that perform the compression
    if __name__ == '__main__':
        # Create the shared memory structures
        processes = [None for x in range(n_processes)]

        print("%s %s Starting %d processes" % (timestamp(), out_sig, n_processes))
        for p_id in range(n_processes):
            args = (p_id,
                    n_processes,
                    iota,
                    out_file_prefix,
                    feat_path,
                    feat_h5_name,
                    out_dir,
                    tfidf,
                    idf_path,
                    threshold,
                    mean_path,
                    std_path,
                    normalize,
                    bit_shift_t,
                    multiplier_t,
                    n_feat_per_int,
                    bit_shift_ir,
                    multiplier_ir)
            processes[p_id] = mp.Process(target=compress_ratio, args=args)
            processes[p_id].start()

        print("%s %s Waiting for processes" % (timestamp(), out_sig))
        for p_id in range(n_processes):
            processes[p_id].join()

    print("%s %s All threads finished compressing, writing..." % (timestamp(),
                                                                  out_sig))

    # Determine the total no. of data items and initialize the merged
    # containers
    n_feat = 0
    n_items = 0

    for p in range(n_processes):
        ratios_path = os.path.join(out_dir, "%s_ratios.h5.tmp%s" % (out_file_prefix, p))
        top_path = os.path.join(out_dir, "%s_top.h5.tmp%s" % (out_file_prefix, p))

        with h5py.File(ratios_path, 'r') as f:
            n_feat += len(f['data'])

        with h5py.File(top_path, 'r') as f:
            n_items += len(f['data'])

    # We only need to get length from ratios to determine the length of ids since they use the same format
    comp_ids = np.empty((n_feat,), dtype=np.uint64)
    comp_feat = np.empty((n_feat,), dtype=np.uint64)
    comp_top = np.empty((n_items,), dtype=np.uint64)

    # Merge the tmp files into one and write the merged file
    feat_i = 0
    item_i = 0

    for p in range(n_processes):
        ids_path = os.path.join(out_dir, "%s_ids.h5.tmp%s" % (out_file_prefix, p))
        ratios_path = os.path.join(out_dir, "%s_ratios.h5.tmp%s" % (out_file_prefix, p))
        top_path = os.path.join(out_dir, "%s_top.h5.tmp%s" % (out_file_prefix, p))

        with h5py.File(ids_path, 'r') as f:
            comp_ids[feat_i: feat_i+len(f['data'])] = f['data']
        os.remove(ids_path)

        with h5py.File(ratios_path, 'r') as f:
            comp_feat[feat_i: feat_i+len(f['data'])] = f['data']
            feat_i += len(f['data'])
        os.remove(ratios_path)

        with h5py.File(top_path, 'r') as f:
            comp_top[item_i: item_i+len(f['data'])] = f['data']
            item_i += len(f['data'])
        os.remove(top_path)

    ids_path = os.path.join(out_dir, "%s_ids.h5" % (out_file_prefix))
    ratios_path = os.path.join(out_dir, "%s_ratios.h5" % (out_file_prefix))
    top_path = os.path.join(out_dir, "%s_top.h5" % (out_file_prefix))

    with h5py.File(ids_path, 'w') as f:
        f.create_dataset('data', data=comp_ids, dtype=np.uint64)

    with h5py.File(ratios_path, 'w') as f:
        f.create_dataset('data', data=comp_feat, dtype=np.uint64)

    with h5py.File(top_path, 'w') as f:
        f.create_dataset('data', data=comp_top, dtype=np.uint64)

    print("%s %s +++ COMPRESSION COMPLETE +++" % (timestamp(), out_sig))
    print("%s %s Elapsed time: %s seconds" % (timestamp(),
                                              out_sig,
                                              time.time() - time_start))

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#  TESTING METHODS
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# .............................................................................
# print_feat()
#
# Prints a feature vector in a sparse format on the standard output.
# .............................................................................
def print_feat(feat, p):
    stdout.write("[ ")
    for i in range(len(feat)):
        if feat[i] > 1 / float(MULTIPLIER[p]):
            stdout.write("%s:%s " % (i, feat[i]))
    print(" ]")


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#  SCRIPT BODY
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
IOTA_HELP = "The number of integers used for each item and modality (iota)."
OUT_DIR_HELP = "The output directory which will contain the compressed data."
FEAT_HELP = "The path to the HDF5 file with the features."
OUT_FILE_PREFIX_HELP = "The identifier for the features."
N_PROCESSES_HELP = "The number of processes to be spawned for the \
                    compression. Default: 1."
FEAT_DNAME_HELP = "The HDF5 name of the dataset containing the \
                       features. Default: 'data'."
TF_IDF_HELP = "Set if it is desired to weight the features using TF-IDF. \
               If set, the '--idf_path' parameter is required. Mutually \
               exclusive with --th."
IDF_PATH_HELP = "The path to the IDF values of individual features."
THRESHOLD_HELP = "Set if you want to threshold the features to only consider \
                  feature values that are above mean plus standard deviation. \
                  If set, the '--mean_path' and '--std_path' are required. \
                  Mutually exclusive with --tfidf."
MEAN_PATH_HELP = "The path to the feature means."
STD_PATH_HELP = "The path to the feature standard deviations."
NORMALIZE_HELP = "Set if you want to L2-normalize the compressed features."
BIT_SHIFT_T_HELP = "Set the bit shift value for the top feature."
MULTIPLIER_T_HELP = "Multiplier for precision of top feature ratio. Ensure that it does not surpass BIT_SHIFT_T in bits."
N_FEAT_PER_INT_HELP = "Set the number of features to add in the 64-bit representation. Make sure it corresponds with BIT_SHIFT_IR."
BIT_SHIFT_IR_HELP = "Set the bit shift value for feature ids and ratios. Make sure it corresponds with N_FEAT_PER_INT."
MULTIPLIER_IR_HELP = "Multiplier for precision of feature ratios. Ensure that it does not surpass BIT_SHIFT_IR in bits."

parser = argparse.ArgumentParser()
parser.add_argument('iota', type=int, help=IOTA_HELP)
parser.add_argument('out_dir', help=OUT_DIR_HELP)
parser.add_argument('feat_path', help=FEAT_HELP)
parser.add_argument('out_file_prefix', help=OUT_FILE_PREFIX_HELP)

parser.add_argument('--p', type=int, help=N_PROCESSES_HELP, default=1)
parser.add_argument('--feat_hdf5_name', help=FEAT_DNAME_HELP,
                    default='data')
parser.add_argument('--tfidf', action='store_true', default=False,
                    help=TF_IDF_HELP)
parser.add_argument('--idf_path', help=IDF_PATH_HELP)
parser.add_argument('--th', action='store_true', default=False,
                    help=THRESHOLD_HELP)
parser.add_argument('--mean_path', help=MEAN_PATH_HELP)
parser.add_argument('--std_path', help=STD_PATH_HELP)
parser.add_argument('--norm', action='store_true', default=False,
                    help=NORMALIZE_HELP)
parser.add_argument('--bit_shift_t', type=int, default=48, help=BIT_SHIFT_T_HELP)
parser.add_argument('--multiplier_t', type=int, default=32, help=MULTIPLIER_T_HELP)
parser.add_argument('--n_feat_per_int', type=int, default=4, help=N_FEAT_PER_INT_HELP)
parser.add_argument('--bit_shift_ir', type=int, default=16, help=BIT_SHIFT_IR_HELP)
parser.add_argument('--multiplier_ir', type=int, default=16, help=MULTIPLIER_IR_HELP)

args = parser.parse_args()

# Check if the output directory is valid
if not os.path.isdir(args.out_dir):
    try:
        os.makedirs(args.out_dir)
    except:
        print("%s ERROR: The provided output directory " % timestamp() +
              "(%s) is not a valid directory!" % args.out_dir)
        exit()


# Check if the visual feature file exists
if not os.path.exists(args.feat_path):
    print("%s ERROR: The feature file " % timestamp() +
          "(%s) does not exist!" % args.feat_path)
    exit()

# Check if the IDF path is specified in case TF-IDF is used
if args.tfidf and not os.path.exists(args.idf_path):
    print("%s ERROR: The path to the file with IDF " % timestamp() +
          "(%s) does not exist!" % args.idf_path)
    exit()

# Check if the mean and std paths are specified when thresholding is used
if args.th and not os.path.exists(args.mean_path):
    print("%s ERROR: The path to the file with feature means " % timestamp() +
          "(%s) does not exist!" % args.mean_path)
    exit()
if args.th and not os.path.exists(args.std_path):
    print("%s ERROR: The path to the file with feature STDs " % timestamp() +
          "(%s) does not exist!" % args.std_path)
    exit()

# If both TF-IDF and thresholding is requested, ask the user to choose one.
if args.tfidf and args.th:
    print("%s ERROR: TF-IDF (--tfidf) and thresholding (--th)" % timestamp() +
          " are mutually exclusive, only one of them can be set!")
    exit()


bit_shift_ir = []
padding = 64 % args.bit_shift_ir
for i in range(int(64/args.bit_shift_ir)):
    bit_shift_ir.append(np.uint64((i * args.bit_shift_ir) + padding))

compress_control(args.iota,
                 args.out_file_prefix,
                 args.out_dir,
                 args.feat_path,
                 args.feat_hdf5_name,
                 args.p,
                 args.tfidf,
                 args.idf_path,
                 args.th,
                 args.mean_path,
                 args.std_path,
                 args.norm,
                 np.uint64(args.bit_shift_t),
                 pow(2,args.multiplier_t),
                 args.n_feat_per_int,
                 bit_shift_ir,
                 pow(2,args.multiplier_ir))
