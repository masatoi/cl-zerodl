(defpackage #:cl-zerodl/core/layer/sigmoid
  (:use #:cl
        #:mgl-mat
        #:cl-zerodl/core/layer/base)
  (:nicknames :zerodl.layer.sigmoid)
  (:import-from #:cl-zerodl/core/utils
                #:define-class)
  (:export #:sigmoid-layer
           #:make-sigmoid-layer))

(in-package #:cl-zerodl/core/layer/sigmoid)

;; 5.5.2 Sigmoid

(define-class sigmoid-layer (layer))

(defun make-sigmoid-layer (input-dimension)
  (let ((input-dimensions (list *batch-size* input-dimension)))
    (make-instance 'sigmoid-layer
                   :input-dimensions  input-dimensions
                   :output-dimensions input-dimensions
                   :forward-out  (make-mat input-dimensions)
                   :backward-out (make-mat input-dimensions))))

(defmethod forward ((layer sigmoid-layer) &rest inputs)
  (let ((out (forward-out layer)))
    (copy! (car inputs) out)
    (.logistic! out)))

(defmethod backward ((layer sigmoid-layer) dout)
  (let ((y (forward-out layer))
        (out (backward-out layer)))
    (copy! y out)
    (.+! -1.0 out)             ; (-1 + y)
    (geem! -1.0 y out 0.0 out) ; -y * (-1 + y)
    (.*! dout out)))           ; dout * -y * (-1 + y)
