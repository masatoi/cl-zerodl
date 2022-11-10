(defpackage #:cl-zerodl/core/layer/affine
  (:use #:cl
        #:mgl-mat
        #:cl-zerodl/core/layer/base)
  (:nicknames :zerodl.layer.affine)
  (:import-from #:cl-zerodl/core/utils
                #:define-class)
  (:export #:affine-layer
           #:x
           #:weight
           #:bias
           #:make-affine-layer))

(in-package #:cl-zerodl/core/layer/affine)

;; 5.6 Affine

(define-class affine-layer (updatable-layer)
  (x :initform (make-mat '(1 1)) :type mat)
  (weight :initform (make-mat '(1 1)) :type mat)
  (bias :initform (make-mat '(1)) :type mat))

;; x: (batch-size, in-size)
;; y: (batch-size, out-size)
;; W: (in-size,    out-size)
;; b: (out-size)

(defun make-affine-layer (input-dimension output-dimension)
  (check-type input-dimension alexandria:positive-integer)
  (check-type output-dimension alexandria:positive-integer)
  (let* ((weight-dimensions (list input-dimension output-dimension))
         (bias-dimension output-dimension)
         (input-dimensions (list *batch-size* input-dimension))
         (output-dimensions (list *batch-size* output-dimension))
         (layer (make-instance
                 'affine-layer
                 :input-dimensions input-dimensions
                 :output-dimensions output-dimensions
                 :forward-out  (make-mat output-dimensions)
                 :backward-out (list (make-mat input-dimensions)  ; dx
                                     (make-mat weight-dimensions) ; dW
                                     (make-mat bias-dimension))   ; db
                 :x      (make-mat input-dimensions)
                 :weight (make-mat weight-dimensions)
                 :bias   (make-mat bias-dimension))))
    (setf (updatable-parameters layer) (list (weight layer) (bias layer))
          (gradients layer) (cdr (backward-out layer)))
    layer))

(defmethod forward ((layer affine-layer) &rest inputs)
  (let* ((x (car inputs))
         (W (weight layer))
         (b (bias layer))
         (out (forward-out layer)))
    (copy! x (x layer))
    (fill! 1.0 out)
    (scale-columns! b out)
    (gemm! 1.0 x W 1.0 out)))

(defmethod backward ((layer affine-layer) dout)
  (destructuring-bind (dx dW db) (backward-out layer)
    (gemm! 1.0 dout (weight layer) 0.0 dx :transpose-b? t) ; dx
    (gemm! 1.0 (x layer) dout 0.0 dW :transpose-a? t)      ; dW
    (sum! dout db :axis 0)                                 ; db
    (backward-out layer)))
