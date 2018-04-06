;;; -*- coding:utf-8; mode:lisp -*-

(in-package :cl-user)

(defpackage cl-zerodl
  (:use :cl :mgl-mat :metabang.bind)
  (:nicknames :zerodl))

(in-package :cl-zerodl)

;;; settings -------------

(setf *default-mat-ctype* :float
      *cuda-enabled*      t
      *print-mat*         t
      *print-length*      100
      *print-level*       10)

;;; utils ----------------

(defmacro define-class (class-name superclass-list &body body)
  "Simplified definition of classes which similar to definition of structure.
 [Example]
  (define-class agent (superclass1 superclass2)
    currency
    position-list
    (position-upper-bound 1)
    log
    money-management-rule)
=> #<STANDARD-CLASS AGENT>"
  `(defclass ,class-name (,@superclass-list)
     ,(mapcar (lambda (slot)
                (let* ((slot-symbol (if (listp slot) (car slot) slot))
                       (slot-name (symbol-name slot-symbol))
                       (slot-initval (if (listp slot) (cadr slot) nil)))
                  (list slot-symbol
                        :accessor (intern slot-name)
                        :initarg (intern slot-name :keyword)
                        :initform slot-initval)))
       body)))

;;; ----------------------

;; 5.4 単純なレイヤーの実装

(define-class layer ()
  input-dimensions output-dimensions
  forward-out backward-out)

(defgeneric forward (layer &rest inputs))
(defgeneric backward (layer dout))

;; 5.4.1

(define-class multiple-layer (layer)
  x y)

(defun make-multiple-layer (input-dimensions)
  (make-instance 'multiple-layer
                 :input-dimensions  input-dimensions
                 :output-dimensions input-dimensions
                 :forward-out (make-mat input-dimensions)
                 :backward-out (list (make-mat input-dimensions)  ; dx
                                     (make-mat input-dimensions)) ; dy
                 :x (make-mat input-dimensions)
                 :y (make-mat input-dimensions)))

(defmethod forward ((layer multiple-layer) &rest inputs)
  (bind ((out (forward-out layer))
         ((x y) inputs))
    (copy! x (x layer))
    (copy! y (y layer))
    ;; geem! is elementwise matrix multiplication
    (geem! 1.0 x y 0.0 out)))

(defparameter mul-layer1 (make-multiple-layer '(2 3)))
(defparameter x (make-mat '(2 3) :initial-contents '((1 2 3)
                                                     (4 5 6))))
(defparameter y (make-mat '(2 3) :initial-contents '((10 20 30)
                                                     (40 50 60))))
(forward mul-layer1 x y)

;; #<MULTIPLE-LAYER {1009DD6143}>
;;   [standard-object]

;; Slots with :INSTANCE allocation:
;;   INPUT-DIMENSIONS               = 3
;;   OUTPUT-DIMENSIONS              = 3
;;   FORWARD-OUT                    = #<MAT 3 AB #(10.0 40.0 90.0)>
;;   BACKWARD-OUT                   = (#<MAT 3 A #(0.0 0.0 0.0)> #<MAT 3 A #(0.0 0.0 0.0)>)
;;   X                              = #<MAT 3 AF #(1.0 2.0 3.0)>
;;   Y                              = #<MAT 3 AF #(10.0 20.0 30.0)>

(defmethod backward ((layer multiple-layer) dout)
  (let* ((out (backward-out layer))
         (dx (car  out))
         (dy (cadr out)))
    (geem! 1.0 dout (y layer) 0.0 dx)
    (geem! 1.0 dout (x layer) 0.0 dy)
    out))

(defparameter dout (make-mat '(2 3) :initial-element 1.0))
(backward mul-layer1 dout)

;; #<MULTIPLE-LAYER {1009DD6143}>
;;   [standard-object]

;; Slots with :INSTANCE allocation:
;;   INPUT-DIMENSIONS               = 3
;;   OUTPUT-DIMENSIONS              = 3
;;   FORWARD-OUT                    = #<MAT 3 AB #(10.0 40.0 90.0)>
;;   BACKWARD-OUT                   = (#<MAT 3 AB #(10.0 20.0 30.0)> #<MAT 3 AB #(1.0 2.0 3.0)>)
;;   X                              = #<MAT 3 ABF #(1.0 2.0 3.0)>
;;   Y                              = #<MAT 3 ABF #(10.0 20.0 30.0)>

;; example of multiple-layer

(defparameter apple   (make-mat '(1 1) :initial-element 100.0))
(defparameter n-apple (make-mat '(1 1) :initial-element 2.0))
(defparameter tax     (make-mat '(1 1) :initial-element 1.1))
(defparameter mul-apple-layer (make-multiple-layer '(1 1)))
(defparameter mul-tax-layer   (make-multiple-layer '(1 1)))

;; forward example
(let* ((apple-price (forward mul-apple-layer apple n-apple))
       (price (forward mul-tax-layer apple-price tax)))
  (print price))

;; #<MAT 1x1 AB #2A((220.0))> 

;; backward example
(defparameter dprice (make-mat '(1 1) :initial-element 1.0))

(bind (((dapple-price dtax) (backward mul-tax-layer dprice))
       ((dapple dn-apple)   (backward mul-apple-layer dapple-price)))
  (print (list dapple dn-apple dtax)))

;; (#<MAT 1x1 B #2A((2.2))> #<MAT 1x1 B #2A((110.0))> #<MAT 1x1 B #2A((200.0))>)

;; add layer

(define-class add-layer (layer))

(defun make-add-layer (input-dimensions)
  (make-instance 'add-layer
                 :input-dimensions  input-dimensions
                 :output-dimensions input-dimensions
                 :forward-out (make-mat input-dimensions)
                 :backward-out (list (make-mat input-dimensions)    ; dx
                                     (make-mat input-dimensions)))) ; dy

(defmethod forward ((layer add-layer) &rest inputs)
  (let ((out (forward-out layer)))
    (copy! (car inputs) out)
    (axpy! 1.0 (cadr inputs) out)))

(defmethod backward ((layer add-layer) dout)
  (bind ((out (backward-out layer))
         ((dx dy) out))
    (copy! dout dx)
    (copy! dout dy)
    out))

;; example of add-layer and multiple-layer

(defparameter orange (make-mat '(1 1) :initial-element 150.0))
(defparameter n-orange (make-mat '(1 1) :initial-element 3.0))
(defparameter mul-orange-layer (make-multiple-layer '(1 1)))
(defparameter add-apple-orange-layer (make-add-layer '(1 1)))

;; forward example
(let* ((apple-price  (forward mul-apple-layer apple n-apple))
       (orange-price (forward mul-orange-layer orange n-orange))
       (all-price    (forward add-apple-orange-layer apple-price orange-price))
       (price        (forward mul-tax-layer all-price tax)))
  (print price))

;; #<MAT 1x1 AB #2A((715.0))> 

;; backward example
(bind ((dprice (make-mat '(1 1) :initial-element 1.0))
       ((dall-price dtax)            (backward mul-tax-layer dprice))
       ((dapple-price dorange-price) (backward add-apple-orange-layer dall-price))
       ((dorange dnorange)           (backward mul-orange-layer dorange-price))
       ((dapple dnapple)             (backward mul-apple-layer dapple-price)))
  (print (list dnapple dapple dorange dnorange dtax)))

;; (#<MAT 1x1 AB #2A((110.0))> #<MAT 1x1 AB #2A((2.2))>
;;  #<MAT 1x1 B #2A((3.3000002))> #<MAT 1x1 B #2A((165.0))>
;;  #<MAT 1x1 AB #2A((650.0))>)

;; 5.5 Activation function layer
;; 5.5.1 Relu

(define-class relu-layer (layer)
  zero
  mask)

(defun make-relu-layer (input-dimensions)
  (make-instance 'relu-layer
                 :input-dimensions  input-dimensions
                 :output-dimensions input-dimensions
                 :forward-out  (make-mat input-dimensions)
                 :backward-out (make-mat input-dimensions)
                 :zero         (make-mat input-dimensions :initial-element 0.0)
                 :mask         (make-mat input-dimensions :initial-element 0.0)))

(defmethod forward ((layer relu-layer) &rest inputs)
  (let ((zero (zero layer))
        (mask (mask layer))
        (out  (forward-out layer)))
    ;; set mask
    (copy! (car inputs) mask)
    (.<! zero mask)
    ;; set output
    (copy! (car inputs) out)
    (.max! 0.0 out)))

(defparameter mask (make-mat '(1 1) :initial-element 0.0))
(defparameter zero (make-mat '(1 1) :initial-element 0.0))

(defparameter relu-layer1 (make-relu-layer '(1 3)))
(defparameter relu-input (make-mat '(1 3) :initial-contents '((3.0 -1.0 0.0))))
(forward relu-layer1 relu-input)

;; #<MAT 3x1 ABF #2A((3.0) (0.0) (0.0))>

(defmethod backward ((layer relu-layer) dout)
  (geem! 1.0 dout (mask layer) 0.0 (backward-out layer)))

(defparameter drelu (make-mat '(1 3) :initial-element 2.0))
(backward relu-layer1 drelu)

;; #<MAT 3x1 AB #2A((2.0) (0.0) (0.0))>

;; 5.5.2 Sigmoid

(define-class sigmoid-layer (layer))

(defun make-sigmoid-layer (input-dimensions)
  (make-instance 'sigmoid-layer
                 :input-dimensions  input-dimensions
                 :output-dimensions input-dimensions
                 :forward-out  (make-mat input-dimensions)
                 :backward-out (make-mat input-dimensions)))

(defmethod forward ((layer sigmoid-layer) &rest inputs)
  (let ((out (forward-out layer)))
    (copy! (car inputs) out)
    (.logistic! out)))

(defmethod backward ((layer sigmoid-layer) dout)
  (let ((y (forward-out layer))
        (out (backward-out layer)))
    (copy! y out)
    (.+! -1.0 out)             ; (-1 + y)
    (geem! -1.0 y out 0.0 out) ; -y * (-1 + y)
    (.*! dout out)))           ; dout * -y * (-1 + y)
  
(defparameter sigmoid-layer1 (make-sigmoid-layer '(1 3)))
(defparameter sigmoid-input (make-mat '(1 3) :initial-contents '((3.0 -1.0 0.0))))
(forward sigmoid-layer1 sigmoid-input)

(defparameter dsigmoid (make-mat '(1 3) :initial-element 2.0))
(backward sigmoid-layer1 dsigmoid)

;; 5.6 Affine

(define-class affine-layer (layer)
  x weight bias)

;; x: (batch-size, in-size)
;; y: (batch-size, out-size)
;; W: (out-size, in-size)
;; b: (out-size)

(defun make-affine-layer (input-dimensions output-dimensions)
  (let ((weight-dimensions (list (cadr input-dimensions) (cadr output-dimensions)))
        (bias-dimension (cadr output-dimensions)))
    (make-instance 'affine-layer
                   :input-dimensions  input-dimensions
                   :output-dimensions output-dimensions
                   :forward-out  (make-mat output-dimensions)
                   :backward-out (list (make-mat input-dimensions)  ; dX
                                       (make-mat weight-dimensions) ; dW
                                       (make-mat bias-dimension))   ; dB
                   :x      (make-mat input-dimensions)
                   :weight (make-mat weight-dimensions)
                   :bias   (make-mat bias-dimension))))

(defmethod forward ((layer affine-layer) &rest inputs)
  (let* ((x (car inputs))
         (W (weight layer))
         (b (bias layer))
         (out (forward-out layer)))
    (copy! x (x layer))
    (fill! 1.0 out)
    (scale-columns! b out)
    (gemm! 1.0 x W 1.0 out)))

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

(defmethod backward ((layer affine-layer) dout)
  (bind (((dx dW db) (backward-out layer)))
    (gemm! 1.0 dout (weight layer) 0.0 dx :transpose-b? t) ; dx
    (gemm! 1.0 (x layer) dout 0.0 dW :transpose-a? t)      ; dW
    (sum! dout db :axis 0)                                 ; dB
    (backward-out layer)))

;; test of gemm! with transpose
(defparameter dout-affine (make-mat '(2 3) :initial-contents '((1 2 3)
                                                               (1 2 3))))
(print (backward affine-layer1 dout-affine))

;; (#<MAT 2x4 F #2A((14.0 32.0 50.0 68.0) (14.0 32.0 50.0 68.0))>
;;  #<MAT 4x3 AF #2A((60.0 120.0 180.0)
;;                   (80.0 160.0 240.0)
;;                   (100.0 200.0 300.0)
;;                   (120.0 240.0 360.0))>
;;  #<MAT 3 F #(2.0 4.0 6.0)>)

(defun average! (a batch-size-tmp)
  (sum! a batch-size-tmp :axis 1)
  (scal! (/ 1.0 (mat-dimension a 1)) batch-size-tmp))

(defun softmax! (a result batch-size-tmp &key (avoid-overflow-p t))
  ;; In order to avoid overflow, subtract average value for each column.
  (when avoid-overflow-p
    (average! a batch-size-tmp)
    (fill! 1.0 result)
    (scale-rows! batch-size-tmp result)
    (axpy! -1.0 result a)) ; a - average(a)
  (.exp! a)
  (sum! a batch-size-tmp :axis 1)
  (fill! 1.0 result)
  (scale-rows! batch-size-tmp result)
  (.inv! result)
  (.*! a result))

(defparameter a (make-mat '(2 3) :initial-contents '((0.3 2.9 4.0)
                                                     (1010 1000 990))))
(defparameter result (make-mat '(2 3)))
(defparameter batch-size-tmp (make-mat 2))

(softmax! a result batch-size-tmp)

;; #<MAT 2x3 BF #2A((0.018211272 0.24519181 0.7365969)
;;                  (0.99995464 4.5397872e-5 2.0610602e-9))>

;;; cross-entropy

(defun cross-entropy! (y target tmp batch-size-tmp &key (delta 1e-7))
  (let ((batch-size (mat-dimension target 0)))
    (copy! y tmp)
    (.+! delta tmp)
    (.log! tmp)
    (geem! 1.0 target tmp 0.0 tmp)
    (sum! tmp batch-size-tmp :axis 1)
    (/ (asum batch-size-tmp) batch-size)))

(defparameter y (make-mat '(2 3) :initial-contents '((1.1 1.2 1.3)
                                                     (3.1 5.1 0.1))))

(defparameter target0 (make-mat '(2 3) :initial-contents '((1 0 0)
                                                           (0 1 0))))

(defparameter tmp (make-mat '(2 3)))
(defparameter batch-size-tmp (make-mat 2))
(defparameter size-1-tmp (make-mat '(1 1)))

(cross-entropy! y target0 tmp batch-size-tmp) ; (/ (+ (log (+ 1.1 1e-7)) (log (+ 5.1 1e-7))) 2)

;;; 5.6.3 Softmax-with-loss

(define-class softmax/loss-layer (layer)
  loss y target batch-size-tmp)

(defun make-softmax/loss-layer (input-dimensions)
  (make-instance 'softmax/loss-layer
                 :input-dimensions  input-dimensions
                 :output-dimensions 1
                 :backward-out (make-mat input-dimensions)
                 :y            (make-mat input-dimensions)
                 :target       (make-mat input-dimensions)
                 :batch-size-tmp (make-mat (car input-dimensions))))

(defparameter softmax/loss-layer1 (make-softmax/loss-layer '(2 3)))

(defmethod forward ((layer softmax/loss-layer) &rest inputs)
  (bind (((x target) inputs)
         (tmp (target layer)) ; use (target layer) as tmp
         (y (y layer))
         (batch-size-tmp (batch-size-tmp layer)))
    (copy! x tmp)
    (softmax! tmp y batch-size-tmp)
    (let ((out (cross-entropy! y target tmp batch-size-tmp)))
      (copy! target (target layer))
      (setf (forward-out layer) out)
      out)))

(defparameter x-softmax/loss
  (make-mat '(2 3) :initial-contents '((0.3  2.9 4.0)
                                       (1010 1000 990))))

(defparameter target (make-mat '(2 3) :initial-contents '((1 0 0)
                                                          (0 1 0))))

(forward softmax/loss-layer1 x-softmax/loss target)
;; => -7.0017767

(defmethod backward ((layer softmax/loss-layer) dout)
  (let* ((target (target layer))
         (y      (y layer))
         (out    (backward-out layer))
         (batch-size (mat-dimension target 0)))
    (copy! y out)
    (axpy! -1.0 target out)
    (scal! (/ 1.0 batch-size) out)))

(backward softmax/loss-layer1 1.0)

;; #<MAT 2x3 AF #2A((-0.49089438 0.122595906 0.36829844)
;;                  (0.49997732 -0.4999773 1.0305301e-9))>

(define-class network ()
  layers last-layer batch-size)

(defun make-network (input-size hidden-size output-size batch-size
                     &key (weight-init-std 0.01))
  (let* ((network
           (make-instance
            'network
            :layers (vector
                     (make-affine-layer (list batch-size input-size)
                                        (list batch-size hidden-size))
                     (make-relu-layer   (list batch-size hidden-size))
                     (make-affine-layer (list batch-size hidden-size )
                                        (list batch-size output-size)))
            :last-layer (make-softmax/loss-layer (list batch-size output-size))
            :batch-size batch-size))
         (W1 (weight (svref (layers network) 0)))
         (W2 (weight (svref (layers network) 2))))
    (gaussian-random! W1)
    (scal! weight-init-std W1)
    (gaussian-random! W2)
    (scal! weight-init-std W2)
    network))

(defparameter network1 (make-network 3 4 2 2))

(defun predict (network x)
  (loop for layer across (layers network) do
    (setf x (forward layer x)))
  x)

(defparameter x1
  (make-mat '(2 3) :initial-contents '((1.1 1.2 1.3)
                                       (10.1 10.2 10.3))))

(defparameter target1 (make-mat '(2 2) :initial-contents '((1 0)
                                                           (0 1))))

(predict network1 x1)

(defun network-loss (network x target)
  (let ((y (predict network x)))
    (forward (last-layer network) y target)))

(network-loss network1 x1 target1)

;; ミニバッチに対する正答率
;; (defmethod accuracy ((network network) )

(defmethod set-gradient! ((network network) x target)
  (let ((layers (layers network))
        dout)
    ;; forward
    (network-loss network x target)
    ;; backward
    (setf dout (backward (last-layer network) 1.0))
    (loop for i from (1- (length layers)) downto 0 do
      (let ((layer (svref layers i)))
        (setf dout (backward layer (if (listp dout) (car dout) dout)))))))

(print (set-gradient! network1 x1 target1))

;;; read data

(defmacro do-index-value-list ((index value list) &body body)
  (let ((iter (gensym))
        (inner-list (gensym)))
    `(labels ((,iter (,inner-list)
                     (when ,inner-list
                       (let ((,index (car ,inner-list))
                             (,value (cadr ,inner-list)))
                         ,@body)
                       (,iter (cddr ,inner-list)))))
       (,iter ,list))))

(defun read-data (data-path data-dimension n-class &key (most-min-class 1))
  (let* ((data-list (svmformat:parse-file data-path))
         (len (length data-list))
         (target     (make-array (list len n-class)
                                 :element-type 'single-float
                                 :initial-element 0.0))
         (datamatrix (make-array (list len data-dimension)
                                 :element-type 'single-float
                                 :initial-element 0.0)))
    (loop for i fixnum from 0
          for datum in data-list
          do (setf (aref target i (- (car datum) most-min-class)) 1.0)
             (do-index-value-list (j v (cdr datum))
               (setf (aref datamatrix i (- j most-min-class)) v)))
    (values (array-to-mat datamatrix) (array-to-mat target))))

(multiple-value-bind (datamat target)
    (read-data "/home/wiz/datasets/mnist.scale" 784 10 :most-min-class 0)
  (defparameter mnist-dataset datamat)
  (defparameter mnist-target target))

(multiple-value-bind (datamat target)
    (read-data "/home/wiz/datasets/mnist.scale.t" 784 10 :most-min-class 0)
  (defparameter mnist-dataset-test datamat)
  (defparameter mnist-target-test target))

(defun set-mini-batch! (dataset start-row-index batch-size)
  (let ((dim (mat-dimension dataset 1)))
    (reshape-and-displace! dataset
                           (list batch-size dim)
                           (* start-row-index dim))))

(defun reset-shape! (dataset)
  (let* ((dim (mat-dimension dataset 1))
         (len (/ (mat-max-size dataset) dim)))
    (reshape-and-displace! dataset (list len dim) 0)))

;;;;;;;;;;;;;;;;;;;;

(defun train (network x target &key (learning-rate 0.1))
  (set-gradient! network x target)

  (bind ((layer (aref (layers network) 0))
         ((dx dW dB) (backward-out layer)))
    (declare (ignore dx))
    (axpy! (- learning-rate) dW (weight layer))
    (axpy! (- learning-rate) dB (bias layer)))

  (bind ((layer (aref (layers network) 2))
         ((dx dW dB) (backward-out layer)))
    (declare (ignore dx))
    (axpy! (- learning-rate) dW (weight layer))
    (axpy! (- learning-rate) dB (bias layer))))

(defparameter mnist-network (make-network 784 50 10 100))

(time
 (loop repeat 10000 do
   (let* ((batch-size (batch-size mnist-network))
          (rand (random (- 60000 batch-size))))
     (set-mini-batch! mnist-dataset rand batch-size)
     (set-mini-batch! mnist-target  rand batch-size)
     (train mnist-network mnist-dataset mnist-target))))

;; CPU

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

;; GPU

;; (with-cuda* ()
;;   (time
;;    (loop repeat 10000 do
;;      (let* ((batch-size (batch-size mnist-network))
;;             (rand (random (- 60000 batch-size))))
;;        (set-mini-batch! mnist-dataset rand batch-size)
;;        (set-mini-batch! mnist-target  rand batch-size)
;;        (train mnist-network mnist-dataset mnist-target)))))

;; Evaluation took:
;;   4.882 seconds of real time
;;   4.884000 seconds of total run time (4.504000 user, 0.380000 system)
;;   [ Run times consist of 0.004 seconds GC time, and 4.880 seconds non-GC time. ]
;;   100.04% CPU
;;   16,561,635,611 processor cycles
;;   335,076,320 bytes consed

(defun max-position-column (arr)
  (declare (optimize (speed 3) (space 0) (safety 0) (debug 0))
           (type (array single-float) arr))
  (let ((max-arr (make-array (array-dimension arr 0)
                             :element-type 'single-float
                             :initial-element most-negative-single-float))
        (pos-arr (make-array (array-dimension arr 0)
                             :element-type 'fixnum
                             :initial-element 0)))
    (loop for i fixnum from 0 below (array-dimension arr 0) do
      (loop for j fixnum from 0 below (array-dimension arr 1) do
        (when (> (aref arr i j) (aref max-arr i))
          (setf (aref max-arr i) (aref arr i j)
                (aref pos-arr i) j))))
    pos-arr))

(defun predict-class (network x)
  (max-position-column (mat-to-array (predict network x))))

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

(defun accuracy (network dataset target)
  (let* ((batch-size (batch-size network))
         (dim (mat-dimension dataset 1))
         (len (/ (mat-max-size dataset) dim))
         (cnt 0))
    (loop for n from 0 to (- len batch-size) by batch-size do
      (set-mini-batch! dataset n batch-size)
      (set-mini-batch! target n batch-size)
      (incf cnt
            (loop for pred across (predict-class network dataset)
                  for tgt  across (max-position-column (mat-to-array target))
                  count (= pred tgt))))
    (* (/ cnt len) 1.0)))

(accuracy mnist-network mnist-dataset mnist-target)
(accuracy mnist-network mnist-dataset-test mnist-target-test)

(defparameter mnist-network (make-network 784 256 10 100))
(defparameter train-acc-list nil)
(defparameter test-acc-list nil)

(loop for i from 1 to 100000 do
  (let* ((batch-size (batch-size mnist-network))
         (rand (random (- 60000 batch-size))))
    (set-mini-batch! mnist-dataset rand batch-size)
    (set-mini-batch! mnist-target  rand batch-size)
    (train mnist-network mnist-dataset mnist-target)
    (when (zerop (mod i 600))
      (let ((train-acc (accuracy mnist-network mnist-dataset mnist-target))
            (test-acc  (accuracy mnist-network mnist-dataset-test mnist-target-test)))
        (format t "cycle: ~A~,15Ttrain-acc: ~A~,10Ttest-acc: ~A~%" i train-acc test-acc)
        (push train-acc train-acc-list)
        (push test-acc  test-acc-list)))))

(clgp:plots (list (reverse train-acc-list)
                  (reverse test-acc-list))
            :main "MNIST, 256->Relu->256"
            :title-list '("train" "test")
            :x-label "n-epoch"
            :y-label "accuracy"
            :y-range '(0.9 1.015))

;;  6.179 seconds of real time for set-gradient!      ; python: 7.0sec
;;  22.230 seconds of real time for set-mini-batch!   ; python: 0.55sec
;;  1.583 seconds of real time for  set-batch

;; (sb-profile:profile forward backward train softmax! set-gradient! predict network-loss)
;; (sb-profile:report)
;; (sb-profile:unprofile forward backward train softmax! set-gradient! predict network-loss)

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


(defun make-network-sigmoid (input-size hidden-size output-size batch-size &key (weight-init-std 0.01))
  (let* ((network
           (make-instance
            'network
            :layers (vector
                     (make-affine-layer (list batch-size input-size)   (list batch-size hidden-size))
                     (make-sigmoid-layer (list batch-size hidden-size))
                     (make-affine-layer (list batch-size hidden-size ) (list batch-size output-size)))
            :last-layer (make-softmax/loss-layer (list batch-size output-size))
            :batch-size batch-size))
         (W1 (weight (svref (layers network) 0)))
         (W2 (weight (svref (layers network) 2))))
    (gaussian-random! W1)
    (scal! weight-init-std W1)
    (gaussian-random! W2)
    (scal! weight-init-std W2)
    network))

(defparameter mnist-network-sigmoid (make-network-sigmoid 784 50 10 100))

(time
 (loop for i from 1 to 10000 do
   (let* ((batch-size (batch-size mnist-network-sigmoid))
          (rand (random (- 60000 batch-size))))
     (set-mini-batch! mnist-dataset rand batch-size)
     (set-mini-batch! mnist-target  rand batch-size)
     (train mnist-network-sigmoid mnist-dataset mnist-target)
     (when (zerop (mod i 600))
       (format t "cycle: ~A~,15Ttrain-acc: ~A~,10Ttest-acc: ~A~%"
               i
               (accuracy mnist-network-sigmoid mnist-dataset mnist-target)
               (accuracy mnist-network-sigmoid mnist-dataset-test mnist-target-test))))))
