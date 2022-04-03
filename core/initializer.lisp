(uiop:define-package #:cl-zerodl/core/initializer
    (:use #:cl)
  (:nicknames :initializer)
  (:use-reexport #:cl-zerodl/core/initializer/base
                 #:cl-zerodl/core/initializer/gaussian-initializer
                 #:cl-zerodl/core/initializer/xavier-initializer
                 #:cl-zerodl/core/initializer/he-initializer))

(in-package #:cl-zerodl/core/initializer)
