(defpackage #:cl-zerodl/core/optimizer/aggmo
  (:use #:cl
        #:mgl-mat
        #:cl-zerodl/core/layer/base
        #:cl-zerodl/core/optimizer/base
        #:cl-zerodl/core/network)
  (:nicknames :zerodl.optimizer.aggmo)
  (:import-from #:cl-zerodl/core/utils
                #:define-class)
  (:import-from #:cl-zerodl/core/optimizer/sgd
                #:sgd
                #:learning-rate)
  (:export #:aggmo
           #:make-aggmo))

(in-package #:cl-zerodl/core/optimizer/aggmo)

;; AggMo (Aggregated Momentum)
;; https://arxiv.org/pdf/1804.00325.pdf

(define-class aggmo (sgd)
  velocity-hash-list decay-rate-list)

(defun make-aggmo (learning-rate decay-rate-list network)
  (let ((opt (make-instance 'aggmo
                            :learning-rate learning-rate
                            :velocity-hash-list (loop repeat (length decay-rate-list)
                                                      collect (make-hash-table :test 'eq))
                            :decay-rate-list decay-rate-list)))
    (do-updatable-layer (layer network)
      (dolist (param (updatable-parameters layer))
        (loop for decay-rate in decay-rate-list
              for velocity-hash in (velocity-hash-list opt)
              do (setf (gethash param velocity-hash)
                       (make-mat (mat-dimensions param) :initial-element 0.0)))))
    opt))

(defmethod update! ((optimizer aggmo) parameter gradient)
  (let ((v-list (mapcar (lambda (hash) (gethash parameter hash))
                        (velocity-hash-list optimizer)))
        (K (length (decay-rate-list optimizer))))
    (loop for v in v-list
          for decay in (decay-rate-list optimizer)
          do (scal! decay v)
             (axpy! (- (learning-rate optimizer)) gradient v))
    (dolist (v v-list)
      (axpy! (/ 1.0 K) v parameter))))
