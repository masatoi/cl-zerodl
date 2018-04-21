;;; -*- coding:utf-8; mode:lisp -*-

(in-package :cl-zerodl)

;;; load dataset =========================================================

;; Download CIFAR-10 binary version and extract to directory.
;; http://www.cs.toronto.edu/~kriz/cifar.html
(defparameter dir #P"/home/wiz/datasets/cifar-10-batches-bin/")

(defparameter dim 3072)
(defparameter n-class 10)

(defparameter x
  (make-array '(50000 3072) :element-type 'single-float))

(defparameter y
  (make-array '(50000 10) :element-type 'single-float))

(defparameter x.t
  (make-array '(10000 3072) :element-type 'single-float))

(defparameter y.t
  (make-array '(10000 10) :element-type 'single-float))

(defun load-cifar (file datamatrix target n)
  (with-open-file (s file :element-type '(unsigned-byte 8))
    (loop for i from (* n 10000) below (* (1+ n) 10000) do
      (setf (aref target i (read-byte s)) 1.0)
      (loop for j from 0 below 3072 do
        (setf (aref datamatrix i j) (coerce (read-byte s) 'single-float))))
    'done))

(loop for i from 0 to 4 do
  (load-cifar (merge-pathnames (format nil "data_batch_~A.bin" (1+ i)) dir) x y i))

(load-cifar (merge-pathnames "test_batch.bin" dir) x.t y.t 0)

;;; Normalize

(defparameter x.ave (make-array 3072 :element-type 'single-float))

(loop for i from 0 below 50000 do
  (loop for j from 0 below 3072 do
    (incf (aref x.ave j) (aref x i j))))

(loop for j from 0 below 3072 do
  (setf (aref x.ave j) (/ (aref x.ave j) 50000)))

(defun square (x)
  (* x x))

(defparameter x.std (make-array 3072 :element-type 'single-float))

(loop for i from 0 below 50000 do
  (loop for j from 0 below 3072 do
    (incf (aref x.std j) (square (- (aref x i j) (aref x.ave j))))))

(loop for j from 0 below 3072 do
  (setf (aref x.std j) (/ (aref x.std j) 50000)))

(loop for j from 0 below 3072 do
  (setf (aref x.std j) (sqrt (aref x.std j))))

(defparameter x.norm
  (make-array '(50000 3072) :element-type 'single-float))

(defparameter x.t.norm
  (make-array '(10000 3072) :element-type 'single-float))

(loop for i from 0 below 50000 do
  (loop for j from 0 below 3072 do
    (setf (aref x.norm i j)
          (/ (- (aref x i j) (aref x.ave j))
             (aref x.std j)))))

(loop for i from 0 below 10000 do
  (loop for j from 0 below 3072 do
    (setf (aref x.t.norm i j)
          (/ (- (aref x.t i j) (aref x.ave j))
             (aref x.std j)))))

(defparameter cifar-dataset (array-to-mat x.norm))
(defparameter cifar-target  (array-to-mat y))

(defparameter cifar-test (array-to-mat x.t.norm))
(defparameter cifar-target-test (array-to-mat y.t))

;;; Define network

(defparameter cifar-network
  (make-network '((affine  :in 3072 :out 256)
                  (relu    :in 256)
                  (affine  :in 256 :out 256)
                  (relu    :in 256)
                  (affine  :in 256  :out 10)
                  (softmax :in 10))
                :batch-size 100
                :initializer (make-instance 'he-initializer)))

;;; Momentum optimizer

(setf (optimizer cifar-network)
      (make-momentum-sgd 0.01 0.9 cifar-network))

(setf (optimizer cifar-network)
      (make-adagrad 0.01 0.9 cifar-network))

(setf (optimizer cifar-network)
      (make-aggmo 0.01 '(0.0 0.9 0.99) cifar-network))

(defparameter train-acc-list nil)
(defparameter test-acc-list nil)

(with-cuda* ()
  (loop for i from 1 to (* 500 15) do
    (let* ((batch-size (batch-size cifar-network))
           (rand (random (- 50000 batch-size))))
      (set-mini-batch! cifar-dataset rand batch-size)
      (set-mini-batch! cifar-target  rand batch-size)
      (train cifar-network cifar-dataset cifar-target)
      (when (zerop (mod i 500))
        (let ((train-acc (accuracy cifar-network cifar-dataset cifar-target))
              (test-acc  (accuracy cifar-network cifar-test cifar-target-test)))
          (format t "cycle: ~A~,15Ttrain-acc: ~A~,10Ttest-acc: ~A~%" i train-acc test-acc)
          (push train-acc train-acc-list)
          (push test-acc  test-acc-list))))))

(clgp:plots (list (reverse train-acc-list)
                  (reverse test-acc-list)
                  )
            :title-list '("train(adagrad)" "test(adagrad)"
                          
                          ;;"train(momentum)" "test(momentum)"
                          ;; "train(SGD+BN)" "test(SGD+BN)"
                          )
            :x-label "n-epoch"
            :y-label "accuracy")
