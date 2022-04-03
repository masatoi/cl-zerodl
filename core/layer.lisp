(uiop:define-package #:cl-zerodl/core/layer
    (:use #:cl)
  (:nicknames :layer)
  (:use-reexport #:cl-zerodl/core/layer/base
                 #:cl-zerodl/core/layer/affine
                 #:cl-zerodl/core/layer/relu
                 #:cl-zerodl/core/layer/sigmoid
                 #:cl-zerodl/core/layer/softmax
                 #:cl-zerodl/core/layer/conv2d
                 #:cl-zerodl/core/layer/batch-normalization
                 #:cl-zerodl/core/layer/dropout))

(in-package #:cl-zerodl/core/layer)
