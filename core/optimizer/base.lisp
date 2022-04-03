(defpackage #:cl-zerodl/core/optimizer/base
  (:use #:cl)
  (:nicknames :zerodl.optimizer)
  (:import-from #:cl-zerodl/core/utils
                #:define-class)
  (:export #:optimizer
           #:update!))

(in-package #:cl-zerodl/core/optimizer/base)

;;; Optimizer

(define-class optimizer ())

(defgeneric update! (optimizer parameter gradient))
