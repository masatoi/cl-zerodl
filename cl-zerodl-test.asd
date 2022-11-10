#|
  This file is a part of cl-zerodl project.
|#

(in-package :cl-user)
(defpackage cl-zerodl-test-asd
  (:use :cl :asdf))
(in-package :cl-zerodl-test-asd)

(defsystem cl-zerodl-test
  :author ""
  :license ""
  :depends-on (:cl-zerodl
               :prove)
  :components ((:module "tests"
                :components
                ((:test-file "cl-zerodl"))))
  :description "Test system for cl-zerodl"

  :defsystem-depends-on (:prove-asdf)
  :perform (test-op :after (op c)
                    (funcall (intern #.(string :run-test-system) :prove-asdf) c)
                    (asdf:clear-system c)))
