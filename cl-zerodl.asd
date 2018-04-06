#|
  This file is a part of cl-zerodl project.
|#

(in-package :cl-user)
(defpackage cl-zerodl-asd
  (:use :cl :asdf))
(in-package :cl-zerodl-asd)

(defsystem cl-zerodl
  :version "0.1"
  :author "Satoshi Imai"
  :license "MIT"
  :depends-on (:clgplot :mgl-mat :metabang-bind :cl-libsvm-format)
  :components ((:module "src"
                :components
                ((:file "cl-zerodl")
                 ;; (:file "matrix")
                 ;; (:file "2-perceptron")
                 ;; (:file "3-neural-networks" :depends-on ("matrix"))
                 )))
  :description ""
  :long-description
  #.(with-open-file (stream (merge-pathnames
                             #p"README.org"
                             (or *load-pathname* *compile-file-pathname*))
                            :if-does-not-exist nil
                            :direction :input)
      (when stream
        (let ((seq (make-array (file-length stream)
                               :element-type 'character
                               :fill-pointer t)))
          (setf (fill-pointer seq) (read-sequence seq stream))
          seq)))
  :in-order-to ((test-op (test-op cl-zerodl-test))))
