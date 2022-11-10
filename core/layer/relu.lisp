(defpackage #:cl-zerodl/core/layer/relu
  (:use #:cl
        #:mgl-mat
        #:cl-zerodl/core/layer/base)
  (:nicknames :zerodl.layer.relu)
  (:import-from #:cl-zerodl/core/utils
                #:define-class)
  (:export #:relu-layer
           #:zero
           #:mask
           #:make-relu-layer))

(in-package #:cl-zerodl/core/layer/relu)

;; 5.5 Activation function layer
;; 5.5.1 Relu

(define-class relu-layer (layer)
  (zero :initform (make-mat '(1 1)) :type mat)
  (mask :initform (make-mat '(1 1)) :type mat))

(defun make-relu-layer (input-dimension)
  (check-type input-dimension alexandria:positive-integer)
  (let ((input-dimensions (list *batch-size* input-dimension)))
    (make-instance 'relu-layer
                   :input-dimensions  input-dimensions
                   :output-dimensions input-dimensions
                   :forward-out  (make-mat input-dimensions)
                   :backward-out (make-mat input-dimensions)
                   :zero         (make-mat input-dimensions :initial-element 0.0)
                   :mask         (make-mat input-dimensions :initial-element 0.0))))

(defmethod forward ((layer relu-layer) &rest inputs)
  (let ((zero (zero layer))
        (mask (mask layer))
        (out  (forward-out layer)))
    ;; set mask
    (copy! (car inputs) mask)
    (.<! zero mask)
    ;; set output
    (copy! (car inputs) out)
    (.max! 0.0 out)))

(defmethod backward ((layer relu-layer) dout)
  (geem! 1.0 dout (mask layer) 0.0 (backward-out layer)))
