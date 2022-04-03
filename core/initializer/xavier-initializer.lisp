(defpackage #:cl-zerodl/core/initializer/xavier-initializer
  (:use #:cl
        #:mgl-mat
        #:cl-zerodl/core/initializer/base)
  (:nicknames :zerodl.initializer.xavier-initializer)
  (:import-from #:cl-zerodl/core/utils
                #:define-class)
  (:export #:xavier-initializer))

(in-package #:cl-zerodl/core/initializer/xavier-initializer)

(define-class xavier-initializer (initializer))

(defmethod initialize! ((initializer xavier-initializer) parameter)
  (gaussian-random! parameter :stddev (sqrt (/ 2.0 (+ (mat-dimension parameter 0)
                                                      (mat-dimension parameter 1))))))
