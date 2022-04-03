(defpackage #:cl-zerodl/core/optimizer/sgd
  (:use #:cl
        #:mgl-mat
        #:cl-zerodl/core/optimizer/base)
  (:nicknames :zerodl.optimizer.sgd)
  (:import-from #:cl-zerodl/core/utils
                #:define-class)
  (:export #:sgd
           #:make-sgd))

(in-package #:cl-zerodl/core/optimizer/sgd)

(define-class sgd (optimizer)
  (learning-rate :initform 0.1 :type single-float))

(defun make-sgd (learning-rate)
  (make-instance 'sgd :learning-rate learning-rate))

(defmethod update! ((optimizer sgd) parameter gradient)
  (axpy! (- (learning-rate optimizer)) gradient parameter))
