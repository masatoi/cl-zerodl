(defpackage #:cl-zerodl/core/initializer/gaussian-initializer
  (:use #:cl
        #:mgl-mat
        #:cl-zerodl/core/initializer/base)
  (:nicknames :zerodl.initializer.gaussian-initializer)
  (:import-from #:cl-zerodl/core/utils
                #:define-class)
  (:export #:gaussian-initializer))

(in-package #:cl-zerodl/core/initializer/gaussian-initializer)

(define-class gaussian-initializer (initializer)
  (weight-init-std :initform 0.01 :type single-float))

(defmethod initialize! ((initializer gaussian-initializer) parameter)
  (gaussian-random! parameter :stddev (weight-init-std initializer)))
