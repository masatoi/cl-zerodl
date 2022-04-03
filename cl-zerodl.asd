(defsystem cl-zerodl
  :class :package-inferred-system
  :author "Satoshi Imai"
  :version "0.3"
  :license "MIT"
  :depends-on ("mgl-mat"
               "cl-libsvm-format"
               "alexandria"
               "cl-zerodl/main")
  :description "Common Lisp implementation of 'deep-learning-from-scratch'"
  :in-order-to ((test-op (test-op "cl-zerodl/tests"))))
