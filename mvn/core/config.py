from yacs.config import CfgNode as CN

_C = CN()

_C.BACKBONE = CN()
_C.BACKBONE.DOWNSAMPLE = 4

_C.KEYPOINT = CN()
_C.KEYPOINT.HEATMAP_SIZE = (96, 96)
_C.KEYPOINT.NFEATS = 256


_C.EPIPOLAR = CN()
_C.EPIPOLAR.SAMPLESIZE = 64
_C.EPIPOLAR.SOFTMAX_ENABLED = True
_C.EPIPOLAR.SOFTMAXSCALE = 1 / _C.EPIPOLAR.SAMPLESIZE**.5
_C.EPIPOLAR.ATTENTION = 'avg' # max
# 'cos', 'dot', 'prior'
_C.EPIPOLAR.SIMILARITY = 'dot'
_C.EPIPOLAR.REPROJECT_LOSS_WEIGHT = 0.
_C.EPIPOLAR.BOTTLENECK = 1
# can parameterize 'z', 'theta', 'phi', 'g'
_C.EPIPOLAR.PARAMETERIZED = ()
_C.EPIPOLAR.ZRESIDUAL = False
_C.EPIPOLAR.PRIOR = False
_C.EPIPOLAR.PRIORMUL = False
# find corrspondence based on 'feature' or 'rgb'
_C.EPIPOLAR.FIND_CORR = 'feature'
_C.EPIPOLAR.OTHER_GRAD = ('other1', 'other2')
_C.EPIPOLAR.POOLING = False
_C.EPIPOLAR.SOFTMAX_ENABLED = True
_C.EPIPOLAR.USE_CORRECT_NORMALIZE = False

_C.DATASETS = CN()
_C.DATASETS.IMAGE_SIZE = (384, 384)
# image resize ratio
_C.DATASETS.IMAGE_RESIZE = 1.
# upscale to original image size for prediction
_C.DATASETS.PREDICT_RESIZE = 1.
# INCLUDE CAMERAS, if empty, use all
_C.DATASETS.CAMERAS = ()

_C.VIS = CN()
_C.VIS.EPIPOLAR_LINE = False

