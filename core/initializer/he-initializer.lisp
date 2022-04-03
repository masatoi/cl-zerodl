(defpackage #:cl-zerodl/core/initializer/he-initializer
  (:use #:cl
        #:mgl-mat
        #:cl-zerodl/core/initializer/base)
  (:nicknames :zerodl.initializer.he-initializer)
  (:import-from #:cl-zerodl/core/utils
                #:define-class)
  (:export #:he-initializer))

(in-package #:cl-zerodl/core/initializer/he-initializer)

(define-class he-initializer (initializer))

(defmethod initialize! ((initializer he-initializer) parameter)
  (gaussian-random! parameter :stddev (sqrt (/ 2.0 (mat-dimension parameter 0)))))
