;;; -*- coding:utf-8; mode:lisp -*-

(in-package :cl-zerodl)

(multiple-value-bind (datamat target)
    (read-data "/home/wiz/datasets/mnist.scale" 784 10 :most-min-class 0)
  (defparameter mnist-dataset datamat)
  (defparameter mnist-target target))

(multiple-value-bind (datamat target)
    (read-data "/home/wiz/datasets/mnist.scale.t" 784 10 :most-min-class 0)
  (defparameter mnist-dataset-test datamat)
  (defparameter mnist-target-test target))

(defparameter mnist-network
  (make-network '((affine  :in 784 :out 512)
                  (batch-norm :in 512)
                  (relu    :in 512)
                  (affine  :in 512 :out 512)
                  (batch-norm :in 512)
                  (relu    :in 512)
                  (affine  :in 512 :out 10)
                  (softmax :in 10))
                :batch-size 100
                :initializer (make-instance 'he-initializer)))

(defparameter mnist-network
  (make-network (vector
                 (make-affine-layer '(10 784) '(10 512))
                 (make-batch-normalization-layer '(10 512))
                 (make-relu-layer '(10 512))
                 (make-affine-layer '(10 512) '(10 512))
                 (make-batch-normalization-layer '(10 512))
                 (make-relu-layer '(10 512))
                 (make-affine-layer '(10 512) '(10 10))
                 (make-softmax/loss-layer '(10 10)))
                :batch-size 10
                :initializer (make-instance 'he-initializer)))

(defparameter mnist-network
  (make-network (vector
                 (make-affine-layer '(100 784) '(100 512))
                 (make-batch-normalization-layer '(100 512))
                 (make-relu-layer '(100 512))
                 (make-affine-layer '(100 512) '(100 512))
                 (make-batch-normalization-layer '(100 512))
                 (make-relu-layer '(100 512))
                 (make-affine-layer '(100 512) '(100 10))
                 (make-softmax/loss-layer '(100 10)))
                :batch-size 100
                :initializer (make-instance 'he-initializer)))

(defparameter mnist-network
  (make-network (vector
                 (make-conv2d-layer '(100 28 28) 3 3)
                 (make-relu-layer '(100 100))
                 (make-max-pool-layer '(100 10 10) '(100 5 5) '(2 2))
                 (make-affine-layer '(100 25) '(100 25))
                 (make-batch-normalization-layer '(100 25))
                 (make-relu-layer '(100 25))
                 (make-affine-layer '(100 25) '(100 25))
                 (make-batch-normalization-layer '(100 25))
                 (make-relu-layer '(100 25))
                 (make-affine-layer '(100 25) '(100 10))
                 (make-softmax/loss-layer '(100 10)))
                :batch-size 100
                :initializer (make-instance 'he-initializer)))

(defparameter cnn-net
  (make-network '((affine  :in 784 :out 512)
                  (batch-norm :in 512)
                  (relu    :in 512)
                  (affine  :in 512 :out 512)
                  (batch-norm :in 512)
                  (relu    :in 512)
                  (affine  :in 512 :out 512)
                  (batch-norm :in 512)
                  (relu    :in 512)
                  (affine  :in 512 :out 512)
                  (batch-norm :in 512)
                  (relu    :in 512)
                  (affine  :in 512 :out 512)
                  (batch-norm :in 512)
                  (relu    :in 512)
                  (affine  :in 512 :out 512)
                  (batch-norm :in 512)
                  (relu    :in 512)
                  (affine  :in 512 :out 10)
                  (softmax :in 10))
                :batch-size 100
                :initializer (make-instance 'he-initializer)))

;;; Momentum optimizer

(setf (optimizer mnist-network)
      (make-momentum-sgd 0.01 0.9 mnist-network))

(setf (optimizer mnist-network)
      (make-momentum-sgd 0.01 0.9 mnist-network))

(setf (optimizer mnist-network)
      (make-adagrad 0.01 0.9 mnist-network))

(setf (optimizer mnist-network)
      (make-aggmo 0.01 '(0.0 0.9 0.99) mnist-network))

(setf (optimizer mnist-network)
      (make-aggmo 0.05 '(0.0 0.9 0.99) mnist-network))

(time
 (loop repeat 10000 do
   (let* ((batch-size (batch-size mnist-network))
          (rand (random (- 60000 batch-size))))
     (set-mini-batch! mnist-dataset rand batch-size)
     (set-mini-batch! mnist-target  rand batch-size)
     (train mnist-network mnist-dataset mnist-target))))

(loop repeat (* 600 15) do
  (let* ((batch-size (batch-size mnist-network))
         (rand (random (- 60000 batch-size))))
    (set-mini-batch! mnist-dataset rand batch-size)
    (set-mini-batch! mnist-target  rand batch-size)
    (train mnist-network mnist-dataset mnist-target)))

;; CPU

;; CPU (hidden-size = 50)

;; Evaluation took:
;;   6.252 seconds of real time
;;   24.940000 seconds of total run time (16.044000 user, 8.896000 system)
;;   [ Run times consist of 0.020 seconds GC time, and 24.920 seconds non-GC time. ]
;;   398.91% CPU
;;   21,206,428,661 processor cycles
;;   370,380,864 bytes consed

;; CPU (hidden-size = 256)

;; Evaluation took:
;;   13.120 seconds of real time
;;   52.396000 seconds of total run time (35.036000 user, 17.360000 system)
;;   [ Run times consist of 0.020 seconds GC time, and 52.376 seconds non-GC time. ]
;;   399.36% CPU
;;   44,502,821,057 processor cycles
;;   371,635,088 bytes consed

;; CPU (hidden-size = 512)

;; Evaluation took:
;;   27.189 seconds of real time
;;   108.0000000 seconds of total run time (75.776000 user, 32.224000 system)
;;   [ Run times consist of 0.024 seconds GC time, and 107.976 seconds non-GC time. ]
;;   397.22% CPU
;;   92,232,020,174 processor cycles
;;   373,157,568 bytes consed

;; GPU

(with-cuda* ()
  (time
   (loop repeat 10000 do
     (let* ((batch-size (batch-size mnist-network))
            (rand (random (- 60000 batch-size))))
       (set-mini-batch! mnist-dataset rand batch-size)
       (set-mini-batch! mnist-target  rand batch-size)
       (train mnist-network mnist-dataset mnist-target)))))

(time
   (loop repeat 10000 do
     (let* ((batch-size (batch-size mnist-network))
            (rand (random (- 60000 batch-size))))
       (set-mini-batch! mnist-dataset rand batch-size)
       (set-mini-batch! mnist-target  rand batch-size)
       (train mnist-network mnist-dataset mnist-target))))

;; GPU (hidden-size = 50)

;; Evaluation took:
;;   4.882 seconds of real time
;;   4.884000 seconds of total run time (4.504000 user, 0.380000 system)
;;   [ Run times consist of 0.004 seconds GC time, and 4.880 seconds non-GC time. ]
;;   100.04% CPU
;;   16,561,635,611 processor cycles
;;   335,076,320 bytes consed

;; GPU (hidden-size = 256)

;; Evaluation took:
;;   6.709 seconds of real time
;;   6.712000 seconds of total run time (5.660000 user, 1.052000 system)
;;   100.04% CPU
;;   22,759,082,791 processor cycles
;;   323,884,528 bytes consed

;; GPU (hidden-size = 512)

;; Evaluation took:
;;   8.552 seconds of real time
;;   8.556000 seconds of total run time (6.932000 user, 1.624000 system)
;;   [ Run times consist of 0.004 seconds GC time, and 8.552 seconds non-GC time. ]
;;   100.05% CPU
;;   29,011,606,780 processor cycles
;;   323,874,336 bytes consed

(set-mini-batch! mnist-dataset 0 100)
(set-mini-batch! mnist-target 0 100)

(print (predict-class mnist-network mnist-dataset))

;; #(5 0 4 1 9 2 1 3 1 4 3 5 3 6 1 7 2 8 6 9 4 0 9 1 1 2 4 3 2 7 3 8 6 9 0 5 6 0 7
;;   6 1 8 7 9 3 9 8 5 9 3 3 0 7 4 9 8 0 9 4 1 4 4 6 0 4 5 6 1 0 0 1 7 1 6 3 0 2 1
;;   1 7 0 0 2 6 7 8 3 9 0 4 6 7 4 6 8 0 7 8 3 1)

;; (time
;;  (loop repeat 10000 do
;;    (predict-class mnist-network mnist-dataset)))

;; Evaluation took:
;;   2.603 seconds of real time
;;   10.228000 seconds of total run time (6.408000 user, 3.820000 system)
;;   392.93% CPU
;;   8,827,408,084 processor cycles
;;   140,452,704 bytes consed

(print (max-position-column (mat-to-array mnist-target)))

;; #(5 0 4 1 9 2 1 3 1 4 3 5 3 6 1 7 2 8 6 9 4 0 9 1 1 2 4 3 2 7 3 8 6 9 0 5 6 0 7
;;   6 1 8 7 9 3 9 8 5 9 3 3 0 7 4 9 8 0 9 4 1 4 4 6 0 4 5 6 1 0 0 1 7 1 6 3 0 2 1
;;   1 7 9 0 2 6 7 8 3 9 0 4 6 7 4 6 8 0 7 8 3 1)

(accuracy mnist-network mnist-dataset mnist-target)
(accuracy mnist-network mnist-dataset-test mnist-target-test)

(defparameter train-acc-list nil)
(defparameter test-acc-list nil)

(defparameter train-acc-list2 nil)
(defparameter test-acc-list2 nil)


(with-cuda* ()
  (loop for i from 1 to (* 600 150) do
    (let* ((batch-size (batch-size mnist-network))
           (rand (random (- 60000 batch-size))))
      (set-mini-batch! mnist-dataset rand batch-size)
      (set-mini-batch! mnist-target  rand batch-size)
      (train mnist-network mnist-dataset mnist-target)
      (when (zerop (mod i 600))
        ;; (clgp:splot-matrix (mat-to-array (gethash (weight (aref (layers mnist-network) 0))
        ;;                                           (velocities (optimizer mnist-network)))))
        (let ((train-acc (accuracy mnist-network mnist-dataset mnist-target))
              (test-acc  (accuracy mnist-network mnist-dataset-test mnist-target-test)))
          (format t "cycle: ~A~,15Ttrain-acc: ~A~,10Ttest-acc: ~A~%" i train-acc test-acc)
          (push train-acc train-acc-list)
          (push test-acc  test-acc-list))))))

(loop for i from 1 to (* 600 100) do
  (let* ((batch-size (batch-size mnist-network))
         (rand (random (- 60000 batch-size))))
    (set-mini-batch! mnist-dataset rand batch-size)
    (set-mini-batch! mnist-target  rand batch-size)
    (train mnist-network mnist-dataset mnist-target)
    (when (zerop (mod i 600))
      (let ((train-acc (accuracy mnist-network mnist-dataset mnist-target))
            (test-acc  (accuracy mnist-network mnist-dataset-test mnist-target-test)))
        (format t "cycle: ~A~,15Ttrain-acc: ~A~,10Ttest-acc: ~A~%" i train-acc test-acc)))))

(clgp:plots (list (reverse train-acc-list)
                  (reverse test-acc-list)
                  (reverse train-acc-list2)
                  (reverse test-acc-list2)
                  (reverse train-acc-list3)
                  (reverse test-acc-list3)
                  )
            :title-list '("train(momentum)" "test(momentum)"
                          "train(momentum,lambda=0.00001)" "test(momentum,,lambda=0.00001)"
                          "train(momentum,lambda=0.0001)" "test(momentum,,lambda=0.0001)"
                          )
            :x-label "n-epoch"
            :y-label "accuracy"
            :y-range '(0.96 1.015))

;;  6.179 seconds of real time for set-gradient!      ; python: 7.0sec
;;  22.230 seconds of real time for set-mini-batch!   ; python: 0.55sec
;;  1.583 seconds of real time for  set-batch

;; (sb-profile:profile forward backward train softmax! set-gradient! predict loss)
;; (sb-profile:report)
;; (sb-profile:unprofile forward backward train softmax! set-gradient! predict loss)

;;   seconds  |     gc     |     consed     |  calls |  sec/call  |  name  
;; -------------------------------------------------------------
;;     99.143 |      6.596 | 14,728,331,056 |  1,000 |   0.099143 | SET-MINI-BATCH!
;;      2.069 |      0.000 |     32,759,808 |  4,000 |   0.000517 | FORWARD
;;      1.142 |      0.000 |         32,736 |  4,000 |   0.000285 | BACKWARD
;;      0.399 |      0.000 |              0 |  1,000 |   0.000399 | SOFTMAX!
;;      0.129 |      0.000 |         32,768 |  1,000 |   0.000129 | TRAIN
;;      0.070 |      0.000 |              0 |  1,000 |   0.000070 | CALC-GRADIENT
;; -------------------------------------------------------------
;;    102.953 |      6.596 | 14,761,156,368 | 12,000 |            | Total

;; estimated total profiling overhead: 0.01 seconds
;; overhead estimation parameters:
;;   2.4e-8s/call, 1.136e-6s total profiling, 5.68e-7s internal profiling

(defparameter mnist-network-sigmoid
  (make-network '((affine  :in 784 :out 50)
                  (sigmoid :in 50)
                  (affine  :in 50  :out 10)
                  (softmax :in 10))
                :batch-size 100))

;; (time
;;  (loop for i from 1 to 10000 do
;;    (let* ((batch-size (batch-size mnist-network-sigmoid))
;;           (rand (random (- 60000 batch-size))))
;;      (set-mini-batch! mnist-dataset rand batch-size)
;;      (set-mini-batch! mnist-target  rand batch-size)
;;      (train mnist-network-sigmoid mnist-dataset mnist-target)
;;      (when (zerop (mod i 600))
;;        (format t "cycle: ~A~,15Ttrain-acc: ~A~,10Ttest-acc: ~A~%"
;;                i
;;                (accuracy mnist-network-sigmoid mnist-dataset mnist-target)
;;                (accuracy mnist-network-sigmoid mnist-dataset-test mnist-target-test))))))
