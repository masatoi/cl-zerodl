(defpackage #:cl-zerodl/core/layer/softmax
  (:use #:cl
        #:mgl-mat
        #:cl-zerodl/core/layer/base)
  (:nicknames :zerodl.layer.softmax)
  (:import-from #:cl-zerodl/core/utils
                #:define-class
                #:average!)
  (:export #:softmax/loss-layer
           #:y
           #:target
           #:batch-size-tmp
           #:make-softmax/loss-layer))

(in-package #:cl-zerodl/core/layer/softmax)

(defun softmax! (a result batch-size-tmp &key (avoid-overflow-p t))
  ;; In order to avoid overflow, subtract average value for each column.
  (when avoid-overflow-p
    (average! a batch-size-tmp :axis 1)
    (fill! 1.0 result)
    (scale-rows! batch-size-tmp result)
    (axpy! -1.0 result a)) ; a - average(a)
  (.exp! a)
  (sum! a batch-size-tmp :axis 1)
  (fill! 1.0 result)
  (scale-rows! batch-size-tmp result)
  (.inv! result)
  (.*! a result))

;;; cross-entropy

(defun cross-entropy! (y target tmp batch-size-tmp &key (delta 1e-7))
  (let ((batch-size (mat-dimension target 0)))
    (copy! y tmp)
    (.+! delta tmp)
    (.log! tmp)
    (.*! target tmp)
    (sum! tmp batch-size-tmp :axis 1)
    (/ (asum batch-size-tmp) batch-size)))

;;; 5.6.3 Softmax-with-loss

(define-class softmax/loss-layer (layer)
  y target batch-size-tmp)

(defun make-softmax/loss-layer (input-dimension)
  (check-type input-dimension alexandria:positive-integer)
  (let ((input-dimensions (list *batch-size* input-dimension))
        (output-dimensions (list *batch-size* 1)))
    (make-instance 'softmax/loss-layer
                   :input-dimensions  input-dimensions
                   :output-dimensions output-dimensions
                   :backward-out (make-mat input-dimensions)
                   :y            (make-mat input-dimensions)
                   :target       (make-mat input-dimensions)
                   :batch-size-tmp (make-mat (car input-dimensions)))))

(defmethod forward ((layer softmax/loss-layer) &rest inputs)
  (destructuring-bind (x target) inputs
    (let ((tmp (target layer)) ; use (target layer) as tmp
          (y (y layer))
          (batch-size-tmp (batch-size-tmp layer)))
      (copy! x tmp)
      (softmax! tmp y batch-size-tmp)
      (let ((out (cross-entropy! y target tmp batch-size-tmp)))
        (copy! target (target layer))
        (setf (forward-out layer) out)
        out))))

(defmethod backward ((layer softmax/loss-layer) dout)
  (let* ((target (target layer))
         (y      (y layer))
         (out    (backward-out layer))
         (batch-size (mat-dimension target 0)))
    (copy! y out)
    (axpy! -1.0 target out)
    (scal! (/ 1.0 batch-size) out)))
