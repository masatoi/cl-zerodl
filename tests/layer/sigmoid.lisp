(defpackage #:cl-zerodl/tests/layer/sigmoid
  (:use #:cl
        #:rove
        #:cl-zerodl/core/layer/base
        #:cl-zerodl/core/layer/sigmoid)
  (:import-from #:mgl-mat
                #:mat
                #:make-mat
                #:mat-to-array
                #:mat-dimensions))

(in-package #:cl-zerodl/tests/layer/sigmoid)

(deftest forward-backward
  (let* ((*batch-size* 1)
         (in-dim 3)
         (out-dim 3)
         (layer (make-sigmoid-layer in-dim)))
    (testing "forward"
      (let* ((input (make-mat (list *batch-size* in-dim)
                              :initial-contents '((3.0 -1.0 0.0))))
             (result (forward layer input)))
        (ok (typep result 'mat))
        (ok (equal (mat-dimensions result) (list *batch-size* in-dim)))
        (ok (equalp (mat-to-array result) #2A((0.95257413 0.26894143 0.5))))))

    (testing "backward"
      (let* ((dout (make-mat (list *batch-size* out-dim)
                             :initial-contents '((3.0 -1.0 0.0))))
             ;; sigmoid-layer is a layer with no parameters, so it has only the gradient of the input
             (dx (backward layer dout)))
        (ok (typep dx 'mat))
        (ok (equal (mat-dimensions dx) (list *batch-size* in-dim)))
        (ok (equalp (mat-to-array dx)
                    #2A((0.13552997 -0.19661194 0.0))))))))

#+(or)
;; plot standard sigmoid function with forward function
(let* ((*batch-size* 1)
       (x (loop for i from -10.0 to 10.0 by 0.01 collect i))
       (layer (make-sigmoid-layer (length x)))
       (out (forward layer (make-mat (list 1 (length x)) :initial-contents (list x))))
       (out-arr (mat-to-array out))
       (result (loop for i from 0 below (length x) collect (aref out-arr 0 i))))
  (clgp:plot result :x-seq x))
