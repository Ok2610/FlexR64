import numpy as np

# DEFAULT (TOP 5)
BIT_SHIFT_T = np.uint64(48)
MULTIPLIER_T = np.uint64(pow(2,32))
DECOMP_MASK = np.uint64(pow(2,32)-1)
FEAT_PER_INT = 4
BIT_SHIFT_IR = [np.uint64(0), np.uint64(16), np.uint64(32), np.uint64(48)]
DECOMP_MASK_IR = np.uint64(pow(2,16)-1)
MULTIPLIER = np.uint64(pow(2,16))

# TOP 7
BIT_SHIFT_T = np.uint64(54)
MULTIPLIER_T = np.uint64(pow(2,32))
DECOMP_MASK = np.uint64(pow(2,32)-1)
FEAT_PER_INT = 6
BIT_SHIFT_IR = [np.uint64(4), np.uint64(14), np.uint64(24), np.uint64(34), np.uint64(44), np.uint64(54)]
DECOMP_MASK_IR = pow(2,10)-1
MULTIPLIER = np.uint64(pow(2,10))

# TOP 8
BIT_SHIFT_T = np.uint64(55)
MULTIPLIER_T = np.uint64(pow(2,32))
DECOMP_MASK = np.uint64(pow(2,32)-1)
FEAT_PER_INT = 7
BIT_SHIFT_IR = [np.uint64(1), np.uint64(10), np.uint64(19), np.uint64(28), np.uint64(37), np.uint64(46), np.uint64(55)]
DECOMP_MASK_IR = pow(2,9)-1
MULTIPLIER = np.uint64(pow(2,9))



