(defpackage #:cl-zerodl/tests/layer/affine
  (:use #:cl
        #:rove
        #:cl-zerodl/core/layer/base
        #:cl-zerodl/core/layer/affine)
  (:import-from #:mgl-mat
                #:mat
                #:make-mat
                #:mat-to-array
                #:mat-dimensions))

(in-package #:cl-zerodl/tests/layer/affine)

(deftest forward-backward
  (let* ((*batch-size* 2)
         (in-dim 4)
         (out-dim 3)
         (layer (make-affine-layer in-dim out-dim)))
    (setf (weight layer) (make-mat (list in-dim out-dim)
                                   :initial-contents '((1 2 3)
                                                       (4 5 6)
                                                       (7 8 9)
                                                       (10 11 12)))
          (bias layer) (make-mat out-dim :initial-contents '(1 2 3)))
    (testing "forward"
      (let* ((input (make-mat (list *batch-size* in-dim)
                              :initial-contents '((10 20 30 40)
                                                  (50 60 70 80))))
             (result (forward layer input)))
        (ok (typep result 'mat))
        (ok (equal (mat-dimensions result) (list *batch-size* out-dim)))
        (ok (equalp (mat-to-array result)
                    #2A((701.0 802.0 903.0) (1581.0 1842.0 2103.0))))))

    (testing "backward"
      (let* ((dout (make-mat (list *batch-size* out-dim) :initial-contents '((1 2 3)
                                                                             (1 2 3))))
             (result (backward layer dout)))
        (ok (typep result 'list))
        (ok (= (length result) 3))
        (destructuring-bind (dx dW db) result
          (testing "dx: differential of input"
            (ok (typep dx 'mat))
            (ok (equal (mat-dimensions dx) (list *batch-size* in-dim)))
            (ok (equalp (mat-to-array dx)
                        #2A((14.0 32.0 50.0 68.0) (14.0 32.0 50.0 68.0)))))

          (testing "dW: differential of weight"
            (ok (typep dW 'mat))
            (ok (equal (mat-dimensions dW) (list in-dim out-dim)))
            (ok (equalp (mat-to-array dW)
                        #2A((60.0 120.0 180.0) (80.0 160.0 240.0)
                            (100.0 200.0 300.0) (120.0 240.0 360.0)))))

          (testing "db: differential of bias"
            (ok (typep db 'mat))
            (ok (equal (mat-dimensions db) (list out-dim)))
            (ok (equalp (mat-to-array db) #(2.0 4.0 6.0)))))))))
