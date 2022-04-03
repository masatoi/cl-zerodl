(uiop:define-package #:cl-zerodl/core/optimizer
    (:use #:cl)
  (:nicknames :optimizer)
  (:use-reexport #:cl-zerodl/core/optimizer/base
                 #:cl-zerodl/core/optimizer/sgd
                 #:cl-zerodl/core/optimizer/momentum-sgd
                 #:cl-zerodl/core/optimizer/adagrad
                 #:cl-zerodl/core/optimizer/aggmo))

(in-package #:cl-zerodl/core/optimizer)
