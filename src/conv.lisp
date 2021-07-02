;;; -*- coding:utf-8; mode:lisp -*-

(in-package :zerodl)

(ql:quickload :alexandria)

;; conv2d

(defparameter x (make-mat '(2 4 4)
                          :ctype :float
                          :initial-contents '(((1 2 3 0)
                                               (0 1 2 3)
                                               (3 0 1 2)
                                               (2 3 0 1))
                                              ((1 2 3 0)
                                               (0 1 2 3)
                                               (3 0 1 2)
                                               (2 3 0 1)))))

(defparameter w (make-mat '(3 3)
                          :ctype :float
                          :initial-contents '((2 0 1)
                                              (0 1 2)
                                              (1 0 2))))

(defparameter y (make-mat '(2 4 4)
                          :ctype :float
                          :initial-element 0))

(defparameter conv2d1 (make-conv2d-layer '(2 4 4) 3 1))

(copy! w (filter conv2d1))

(defparameter input (reshape x '(2 16)))

(forward conv2d1 input)
(forward-out conv2d1)

(defparameter dout (make-mat '(2 16)
                             :ctype :float
                             :initial-element 1.0))

(backward conv2d1 dout)

(convolve! x w y :start '(0 0) :stride '(1 1) :anchor '(1 1) :batched t)

(print y)

;; #<MAT 2x4x4 AB #3A(((7.0 12.0 10.0 2.0)
;;                     (4.0 15.0 16.0 10.0)
;;                     (10.0 6.0 15.0 6.0)
;;                     (8.0 10.0 4.0 3.0))
;;                    ((7.0 12.0 10.0 2.0)
;;                     (4.0 15.0 16.0 10.0)
;;                     (10.0 6.0 15.0 6.0)
;;                     (8.0 10.0 4.0 3.0)))> 

;; 

(defparameter xd (make-mat '(2 3 4)))
(defparameter wd (make-mat '(2 2)))
(defparameter yd (make-mat '(2 3 2) :initial-element 1.0))

(derive-convolve! x xd w wd yd :start '(0 0) :stride '(1 2) :anchor '(1 0) :batched t)


;; max-pooling

(defparameter x (make-mat '(2 4 4)
                          :ctype :float
                          :initial-contents '(((1 2 1 0)
                                               (0 1 2 3)
                                               (3 0 1 2)
                                               (2 4 0 1))
                                              ((1 2 1 0)
                                               (0 1 2 3)
                                               (3 0 1 2)
                                               (2 4 0 1)))))

(defparameter y (make-mat '(2 2 2)
                          :ctype :float
                          :initial-element 0))

(max-pool! x y :start '(0 0) :stride '(2 2) :anchor '(0 0) :batched t :pool-dimensions '(2 2))

(print y)

;; #<MAT 2x2x2 AB #3A(((2.0 3.0) (4.0 2.0)) ((2.0 3.0) (4.0 2.0)))> 

;; anchor位置を変えてみる

(defparameter y
  (make-mat '(2 3 2)))

(convolve! x w y :start '(0 0) :stride '(1 2) :anchor '(0 0) :batched t)

(print y)

;; #<MAT 2x3x2 AB #3A(((15.0 21.0) (27.0 33.0) (8.0 10.0))
;;                    ((-15.0 -21.0) (-27.0 -33.0) (-8.0 -10.0)))> 

;;; max-pool!

(max-pool! x y :start '(0 0) :stride '(1 2) :anchor '(1 0) :batched t
               :pool-dimensions '(2 2))

;;; ydが後のレイヤーから入ってくるbackwardの入力
;;; xd,wdには代入ではなく加算代入される

(defparameter xd (make-mat '(2 3 4)))
(defparameter wd (make-mat '(2 2)))
(defparameter yd (make-mat '(2 3 2) :initial-element 1.0))

(derive-convolve! x xd w wd yd :start '(0 0) :stride '(1 2) :anchor '(1 0) :batched t)
(print xd)
;; #<MAT 2x3x4 AB #3A(((8.0 24.0 8.0 38.0)
;;                     (8.0 52.0 2.0 76.0)
;;                     (-84.0 140.0 -102.0 170.0))
;;                    ((-10.0 -16.0 -10.0 -30.0)
;;                     (-10.0 -44.0 -4.0 -68.0)
;;                     (78.0 -130.0 96.0 -160.0)))> 

(print wd)
;; #<MAT 2x2 AB #2A((888.0 1080.0) (1736.0 1964.0))> 


(max-pool! x y :start '(0 0) :stride '(1 2) :anchor '(1 0) :batched t
               :pool-dimensions '(2 2))

(ql:quickload :alexandria)
(replace! yd (append (alexandria:iota 6 :start 1)
                     (alexandria:iota 6 :start -1 :step -1)))

(derive-max-pool! x xd y yd :start '(0 0) :stride '(1 2) :anchor '(1 0) :batched t :pool-dimensions '(2 2))
(print xd)
(print yd)

;;; 3次元データでできるか？

(defparameter x (make-mat '(2 2 3 3)))
(defparameter w (make-mat '(2 2 2)))
(defparameter y (make-mat '(2 2 2 2) :initial-element 1.0))

(convolve! x w y :start '(1 1) :stride '(1 1) :anchor '(1 1) :batched t)

;;; 逆伝搬の具体例
;; https://qiita.com/bukei_student/items/a3d1bcd429f99942ace4
;; convolve!はyの値を累積していく(yの初期化が必要)
;; derive-convolve!はxdとwdを累積していく
(defparameter x (make-mat '(9 9)
                          :ctype :float
                          :initial-element 1.0))

(defparameter w (make-mat '(3 3)
                          :ctype :float
                          :initial-contents '((1 2 3)
                                              (4 5 6)
                                              (7 8 9))))

(defparameter y (make-mat '(3 3)
                          :ctype :float
                          :initial-element 0))

(convolve! x w y :start '(0 0) :stride '(3 3) :anchor '(0 0))

(defparameter xd (make-mat '(9 9)
                          :ctype :float
                          :initial-element 0))

(defparameter wd (make-mat '(3 3)
                          :ctype :float
                          :initial-element 0))

(defparameter yd (make-mat '(3 3)
                          :ctype :float
                          :initial-element 1))

(derive-convolve! x xd w wd yd :start '(0 0) :stride '(3 3) :anchor '(0 0))

;;; max-pool

(defparameter x2 (make-mat '(2 4 4)
                          :ctype :float
                          :initial-contents '(((1 2 1 0)
                                               (0 1 2 3)
                                               (3 0 1 2)
                                               (2 4 0 1))
                                              ((1 2 1 0)
                                               (0 1 2 3)
                                               (3 0 1 2)
                                               (2 4 0 1)))))
(defparameter y2 (make-mat '(2 2 2)
                          :ctype :float
                          :initial-element 0))

;; 一般的にstrideはpool-dimensionsサイズと合わせる
(max-pool! x2 y2 :start '(0 0) :stride '(2 2) :anchor '(0 0) :batched t :pool-dimensions '(2 2))

y2

;; #<MAT 2x2x2 AB #3A(((2.0 3.0)
;;                     (4.0 2.0))
;;                    ((2.0 3.0)
;;                     (4.0 2.0)))>

(defparameter x2d (make-mat '(2 4 4)
                          :ctype :float
                          :initial-element 0))
(defparameter dout (make-mat '(2 2 2)
                            :ctype :float
                          :initial-element 1))

(derive-max-pool! x2 x2d y2 dout :start '(0 0) :stride '(2 2) :anchor '(0 0) :batched t :pool-dimensions '(2 2))
