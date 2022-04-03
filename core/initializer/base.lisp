(defpackage #:cl-zerodl/core/initializer/base
  (:use #:cl)
  (:nicknames :zerodl.initializer)
  (:import-from #:cl-zerodl/core/utils
                #:define-class)
  (:export #:initializer
           #:initialize!))

(in-package #:cl-zerodl/core/initializer/base)

;;; Initializer
(define-class initializer ())

(defgeneric initialize! (initializer parameter))
