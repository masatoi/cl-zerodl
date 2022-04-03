(defpackage #:cl-zerodl/core/layer/base
  (:use :cl)
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
           #:backward))

(in-package #:cl-zerodl/core/layer/base)

(define-class layer ()
  input-dimensions
  output-dimensions
  forward-out
  backward-out)

(define-class updatable-layer (layer)
  updatable-parameters
  gradients)

(defgeneric forward (layer &rest inputs))
(defgeneric backward (layer dout))
