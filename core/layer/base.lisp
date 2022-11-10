(defpackage #:cl-zerodl/core/layer/base
  (:use :cl
        :mgl-mat)
  (:nicknames :zerodl.layer)
  (:import-from #:cl-zerodl/core/utils
                #:define-class)
  (:export #:layer
           #:input-dimensions
           #:output-dimensions
           #:forward-out
           #:backward-out
           #:updatable-layer
           #:updatable-parameters
           #:gradients
           #:forward
           #:backward
           #:*batch-size*))

(in-package #:cl-zerodl/core/layer/base)

(defvar *batch-size* 100)

(define-class layer ()
  (input-dimensions :initform (list *batch-size* 1) :type list)
  (output-dimensions :initform (list *batch-size* 1) :type list)
  (forward-out :initform (make-mat (list *batch-size* 1)) :type mat)
  backward-out)

(define-class updatable-layer (layer)
  (updatable-parameters :type list)
  (gradients :type list))

(defgeneric forward (layer &rest inputs))
(defgeneric backward (layer dout))
