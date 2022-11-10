(defpackage #:cl-zerodl/tests/layer/relu
  (:use #:cl
        #:rove
        #:cl-zerodl/core/layer/base
        #:cl-zerodl/core/layer/relu)
  (:import-from #:mgl-mat
                #:mat
                #:make-mat
                #:mat-to-array
                #:mat-dimensions))

(in-package #:cl-zerodl/tests/layer/relu)

(deftest forward-backward
  (let* ((*batch-size* 1)
         (in-dim 3)
         (out-dim 3)
         (layer (make-relu-layer in-dim)))
    (testing "forward"
      (let* ((input (make-mat (list *batch-size* in-dim)
                              :initial-contents '((3.0 -1.0 0.0))))
             (result (forward layer input)))
        (ok (typep result 'mat))
        (ok (equal (mat-dimensions result) (list *batch-size* in-dim)))
        (ok (equalp (mat-to-array result) #2A((3.0 0.0 0.0))))))

    (testing "backward"
      (let* ((dout (make-mat (list *batch-size* out-dim)
                             :initial-element 2.0))
             ;; relu-layer is a layer with no parameters, so it has only the gradient of the input
             (dx (backward layer dout)))
        (ok (typep dx 'mat))
        (ok (equal (mat-dimensions dx) (list *batch-size* in-dim)))
        (ok (equalp (mat-to-array dx)
                    #2A((2.0 0.0 0.0))))))))

#+(or)
;; plot standard relu function with forward function
(let* ((*batch-size* 1)
       (x (loop for i from -10.0 to 10.0 by 0.01 collect i))
       (layer (make-relu-layer (length x)))
       (out (forward layer (make-mat (list 1 (length x)) :initial-contents (list x))))
       (out-arr (mat-to-array out))
       (result (loop for i from 0 below (length x) collect (aref out-arr 0 i))))
  (clgp:plot result :x-seq x :y-range '(-0.5 11)))
