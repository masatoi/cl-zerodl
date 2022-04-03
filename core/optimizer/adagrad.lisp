(defpackage #:cl-zerodl/core/optimizer/adagrad
  (:use #:cl
        #:mgl-mat
        #:cl-zerodl/core/optimizer/base
        #:cl-zerodl/core/network)
  (:nicknames :zerodl.optimizer.adagrad)
  (:import-from #:cl-zerodl/core/utils
                #:define-class)
  (:import-from #:cl-zerodl/core/optimizer/sgd
                #:sgd)
  (:export #:adagrad
           #:make-adagrad))

(in-package #:cl-zerodl/core/optimizer/adagrad)

;; Adagrad
(define-class adagrad (sgd)
  velocities tmps
  ;; Tiny number to add to the denominator to avoid division by zero
  (epsilon :initform 1.0e-6 :type single-float))

(defun make-adagrad (learning-rate network &key (epsilon 1.0e-6))
  (let ((opt (make-instance 'adagrad
                            :learning-rate learning-rate
                            :velocities (make-hash-table :test 'eq)
                            :tmps (make-hash-table :test 'eq)
                            :epsilon epsilon)))
    (do-updatable-layer (layer network)
      (dolist (param (updatable-parameters layer))
        (setf (gethash param (velocities opt))
              (make-mat (mat-dimensions param) :initial-element 0.0)
              (gethash param (tmps opt))
              (make-mat (mat-dimensions param) :initial-element 0.0))))
    opt))

(defmethod update! ((optimizer adagrad) parameter gradient)
  (let ((velocity (gethash parameter (velocities optimizer)))
        (tmp (gethash parameter (tmps optimizer))))
    (geem! 1.0 gradient gradient 1.0 velocity)
    (copy! velocity tmp)
    (.+! (epsilon optimizer) tmp)
    (.sqrt! tmp)
    (.inv! tmp)
    (.*! gradient tmp)
    (axpy! (- (learning-rate optimizer)) tmp parameter)))
