(uiop:define-package #:cl-zerodl/main
    (:use #:cl
          #:mgl-mat)
  (:nicknames :cl-zerodl)
  (:use-reexport #:cl-zerodl/core/utils
                 #:cl-zerodl/core/layer
                 #:cl-zerodl/core/optimizer
                 #:cl-zerodl/core/initializer
                 #:cl-zerodl/core/network))

(in-package #:cl-zerodl/main)

;;; settings -------------

(setf *default-mat-ctype* :float
      *cuda-enabled*      t
      *print-mat*         t
      *print-length*      100
      *print-level*       10)
