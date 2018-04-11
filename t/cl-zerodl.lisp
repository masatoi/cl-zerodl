(in-package :cl-user)
(defpackage cl-zerodl-test
  (:use :cl
        :cl-zerodl
        :prove))
(in-package :cl-zerodl-test)

;; NOTE: To run this test file, execute `(asdf:test-system :cl-zerodl)' in your Lisp.

(plan nil)

;;; ReLU

(defparameter mask (make-mat '(1 1) :initial-element 0.0))
(defparameter zero (make-mat '(1 1) :initial-element 0.0))

(defparameter relu-layer1 (make-relu-layer '(1 3)))
(defparameter relu-input (make-mat '(1 3) :initial-contents '((3.0 -1.0 0.0))))
(forward relu-layer1 relu-input)

;; #<MAT 3x1 ABF #2A((3.0) (0.0) (0.0))>

(defparameter drelu (make-mat '(1 3) :initial-element 2.0))
(backward relu-layer1 drelu)

;; #<MAT 3x1 AB #2A((2.0) (0.0) (0.0))>

;;; Sigmoid

(defparameter sigmoid-layer1 (make-sigmoid-layer '(1 3)))
(defparameter sigmoid-input (make-mat '(1 3) :initial-contents '((3.0 -1.0 0.0))))
(forward sigmoid-layer1 sigmoid-input)

(defparameter dsigmoid (make-mat '(1 3) :initial-element 2.0))
(backward sigmoid-layer1 dsigmoid)

;;; Affine

(defparameter affine-layer1 (make-affine-layer '(2 4) '(2 3)))

(setf (weight affine-layer1)
      (make-mat '(4 3) :initial-contents '((1 2 3)
                                           (4 5 6)
                                           (7 8 9)
                                           (10 11 12))))

(setf (bias affine-layer1) (make-mat 3 :initial-contents '(1 2 3)))

(defparameter x-affine (make-mat '(2 4) :initial-contents '((10 20 30 40)
                                                            (50 60 70 80))))

(print (forward affine-layer1 x-affine))

;; #<MAT 2x3 AF #2A((701.0 802.0 903.0) (1581.0 1842.0 2103.0))> 

(defparameter dout-affine (make-mat '(2 3) :initial-contents '((1 2 3)
                                                               (1 2 3))))
(print (backward affine-layer1 dout-affine))

;; (#<MAT 2x4 F #2A((14.0 32.0 50.0 68.0) (14.0 32.0 50.0 68.0))>
;;  #<MAT 4x3 AF #2A((60.0 120.0 180.0)
;;                   (80.0 160.0 240.0)
;;                   (100.0 200.0 300.0)
;;                   (120.0 240.0 360.0))>
;;  #<MAT 3 F #(2.0 4.0 6.0)>)

;;; softmax!

(defparameter a (make-mat '(2 3) :initial-contents '((0.3 2.9 4.0)
                                                     (1010 1000 990))))
(defparameter result (make-mat '(2 3)))
(defparameter batch-size-tmp (make-mat 2))

(softmax! a result batch-size-tmp)

;; #<MAT 2x3 BF #2A((0.018211272 0.24519181 0.7365969)
;;                  (0.99995464 4.5397872e-5 2.0610602e-9))>

;;; cross-entropy!

(defparameter y (make-mat '(2 3) :initial-contents '((1.1 1.2 1.3)
                                                     (3.1 5.1 0.1))))

(defparameter target0 (make-mat '(2 3) :initial-contents '((1 0 0)
                                                           (0 1 0))))

(defparameter tmp (make-mat '(2 3)))
(defparameter batch-size-tmp (make-mat 2))
(defparameter size-1-tmp (make-mat '(1 1)))

(cross-entropy! y target0 tmp batch-size-tmp) ; (/ (+ (log (+ 1.1 1e-7)) (log (+ 5.1 1e-7))) 2)

;;; Softmax-with-loss

(defparameter softmax/loss-layer1 (make-softmax/loss-layer '(2 3)))

(defparameter x-softmax/loss
  (make-mat '(2 3) :initial-contents '((0.3  2.9 4.0)
                                       (1010 1000 990))))

(defparameter target (make-mat '(2 3) :initial-contents '((1 0 0)
                                                          (0 1 0))))

(forward softmax/loss-layer1 x-softmax/loss target)
;; => -7.0017767

(backward softmax/loss-layer1 1.0)

;; #<MAT 2x3 AF #2A((-0.49089438 0.122595906 0.36829844)
;;                  (0.49997732 -0.4999773 1.0305301e-9))>

;;; Neural Network

(defparameter network1
  (make-network '((affine  :in 3 :out 4)
                  (relu    :in 4)
                  (affine  :in 4 :out 2)
                  (softmax :in 2))
                2))

(defparameter x1
  (make-mat '(2 3) :initial-contents '((1.1 1.2 1.3)
                                       (10.1 10.2 10.3))))

(defparameter target1 (make-mat '(2 2) :initial-contents '((1 0)
                                                           (0 1))))

(predict network1 x1)

(loss network1 x1 target1)

;; Calculate gradient
(print (set-gradient! network1 x1 target1))



(finalize)
