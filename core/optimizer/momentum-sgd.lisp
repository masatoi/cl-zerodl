(defpackage #:cl-zerodl/core/optimizer/momentum-sgd
  (:use #:cl
        #:mgl-mat
        #:cl-zerodl/core/optimizer/base
        #:cl-zerodl/core/network)
  (:nicknames :zerodl.optimizer.momentum-sgd)
  (:import-from #:cl-zerodl/core/utils
                #:define-class)
  (:import-from #:cl-zerodl/core/optimizer/sgd
                #:sgd)
  (:export #:momentum-sgd
           #:make-momentum-sgd))

(in-package #:cl-zerodl/core/optimizer/momentum-sgd)

;; Momentum SGD
(define-class momentum-sgd (sgd)
  velocities decay-rate)

(defun make-momentum-sgd (learning-rate decay-rate network)
  (let ((opt (make-instance 'momentum-sgd
                            :learning-rate learning-rate
                            :velocities (make-hash-table :test 'eq)
                            :decay-rate decay-rate)))
    (do-updatable-layer (layer network)
      (dolist (param (updatable-parameters layer))
        (setf (gethash param (velocities opt))
              (make-mat (mat-dimensions param) :initial-element 0.0))))
    opt))

(defmethod update! ((optimizer momentum-sgd) parameter gradient)
  (let ((velocity (gethash parameter (velocities optimizer))))
    (scal! (decay-rate optimizer) velocity)
    (axpy! (- (learning-rate optimizer)) gradient velocity)
    (axpy! 1.0 velocity parameter)))
