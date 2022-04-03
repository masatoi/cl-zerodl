(defpackage #:cl-zerodl/core/layer/dropout
  (:use #:cl
        #:mgl-mat
        #:cl-zerodl/core/layer/base)
  (:nicknames :zerodl.layer.dropout)
  (:import-from #:cl-zerodl/core/utils
                #:define-class)
  (:export #:dropout-layer
           #:make-dropout-layer))

(in-package #:cl-zerodl/core/layer/dropout)

;;; Dropout layer

(define-class dropout-layer (layer)
  mask threshold in-train?)

(defun make-dropout-layer (input-dimensions &key (dropout-rate 0.5))
  (make-instance 'dropout-layer
                 :input-dimensions  input-dimensions
                 :output-dimensions input-dimensions
                 :forward-out  (make-mat input-dimensions)
                 :backward-out (make-mat input-dimensions)
                 :mask         (make-mat input-dimensions :initial-element 0.0)
                 :threshold    (make-mat input-dimensions :initial-element dropout-rate)))

(defmethod forward ((layer dropout-layer) &rest inputs)
  (let ((x (car inputs))
        (mask (mask layer))
        (threshold (threshold layer))
        (out (forward-out layer)))

    ;; set mask
    (uniform-random! mask)
    (.<! threshold mask)
    ;; set output
    (geem! 1.0 mask x 0.0 out)))

(defmethod backward ((layer dropout-layer) dout)
  (geem! 1.0 dout (mask layer) 0.0 (backward-out layer)))
