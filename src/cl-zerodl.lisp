(in-package :cl-user)

(defpackage cl-zerodl
  (:use :cl :mgl-mat :metabang.bind)
  (:nicknames :zerodl))

(in-package :cl-zerodl)

;;; settings -------------

(setf *default-mat-ctype* :float)
(setf *cuda-enabled* t)

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
                 :input-dimensions input-dimensions
                 :output-dimensions input-dimensions
                 :forward-out (make-mat input-dimensions)
                 :backward-out (list (make-mat input-dimensions) (make-mat input-dimensions))
                 :x (make-mat input-dimensions)
                 :y (make-mat input-dimensions)))

(defmethod forward ((layer multiple-layer) &rest inputs)
  (bind ((out (forward-out layer))
         ((x y) inputs))
    (copy! x (x layer))
    (copy! y (y layer))
    ;; geem! is elementwise matrix multiplication
    (geem! 1.0 x y 0.0 out)))

(defparameter mul-layer1 (make-multiple-layer 3))
(defparameter x (make-mat 3 :initial-contents '(1.0 2.0 3.0)))
(defparameter y (make-mat 3 :initial-contents '(10.0 20.0 30.0)))
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

(defparameter dout (make-mat 3 :initial-element 1.0))
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
                 :input-dimensions input-dimensions
                 :output-dimensions input-dimensions
                 :forward-out (make-mat input-dimensions)
                 :backward-out (list (make-mat input-dimensions) (make-mat input-dimensions))))

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

(defparameter relu-layer1 (make-relu-layer '(3 1)))
(defparameter relu-input (make-mat '(3 1) :initial-contents '((3.0) (-1.0) (0.0))))
(forward relu-layer1 relu-input)

;; #<MAT 3x1 ABF #2A((3.0) (0.0) (0.0))>

(defmethod backward ((layer relu-layer) dout)
  (geem! 1.0 dout (mask layer) 0.0 (backward-out layer)))

(defparameter drelu (make-mat '(3 1) :initial-element 2.0))
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
  
(defparameter sigmoid-layer1 (make-sigmoid-layer '(3 1)))
(defparameter sigmoid-input (make-mat '(3 1) :initial-contents '((3.0) (-1.0) (0.0))))
(forward sigmoid-layer1 sigmoid-input)

(defparameter dsigmoid (make-mat '(3 1) :initial-element 2.0))
(backward sigmoid-layer1 dsigmoid)

;; 5.6 Affine

(define-class affine-layer (layer)
  x weight bias stacked-bias)

;; x: (in-size,  batch-size)
;; y: (out-size, batch-size)
;; W: (out-size, in-size)
;; b: (out-size)

(defun make-affine-layer (input-dimensions output-dimensions)
  (let ((weight-dimensions (list (car output-dimensions) (car input-dimensions)))
        (bias-dimensions (list (car output-dimensions) 1)))
    (make-instance 'affine-layer
                   :input-dimensions  input-dimensions
                   :output-dimensions output-dimensions
                   :forward-out  (make-mat output-dimensions)
                   :backward-out (list (make-mat input-dimensions)  ; dX
                                       (make-mat weight-dimensions) ; dW
                                       (make-mat bias-dimensions))  ; dB
                   :x (make-mat input-dimensions)
                   :weight (make-mat weight-dimensions)
                   :bias   (make-mat bias-dimensions)
                   :stacked-bias (make-mat output-dimensions))))

(defparameter Wx (make-mat '(3 2) :initial-contents '((1 4) (2 5) (3 6))))
(defparameter result (make-mat '(3 1)))

;; sum by rows
(sum! Wx result :axis 1)
result

;; partial copy

(defparameter result2 (make-mat '(3 2)))
(stack! 1 (list result result) result2)

(defparameter affine-layer1 (make-affine-layer '(4 2) '(3 2)))

(defmethod forward ((layer affine-layer) &rest inputs)
  (let* ((x (car inputs))
         (W (weight layer))
         (b (bias layer))
         (bs (stacked-bias layer))
         (out (forward-out layer)))
    (copy! x (x layer))
    (fill! 1.0 bs)
    (scale-rows! b bs)
    (gemm! 1.0 W x 0.0 out)
    (axpy! 1.0 bs out)))

(setf (weight affine-layer1)
      (make-mat '(3 4) :initial-contents '((1 4 7 10) (2 5 8 11) (3 6 9 12))))
(setf (bias affine-layer1) (make-mat '(3 1) :initial-contents '((1) (2) (3))))
(defparameter x-affine (make-mat '(4 2) :initial-contents '((10 50) (20 60) (30 70) (40 80))))

(time (print (forward affine-layer1 x-affine)))

;; #<MAT 3x2 F #2A((701.0 1581.0) (802.0 1842.0) (903.0 2103.0))> 

(defmethod backward ((layer affine-layer) dout)
  (bind (((dx dW db) (backward-out layer)))
    (gemm! 1.0 (weight layer) dout 0.0 dx :transpose-a? t) ; dx
    (gemm! 1.0 dout (x layer) 0.0 dW :transpose-b? t) ; dW
    (sum! dout db :axis 1)
    (backward-out layer)))

;; test of gemm! with transpose
(defparameter dout-affine (make-mat '(3 2) :initial-contents '((1 1) (2 2) (3 3))))
(defparameter result (make-mat '(3 3)))
(gemm! 1.0 dout-affine (weight affine-layer1) 0.0 result :transpose-b? t)
result

(print (backward affine-layer1 dout-affine))

;; (#<MAT 4x2 AF #2A((14.0 14.0) (32.0 32.0) (50.0 50.0) (68.0 68.0))>
;;  #<MAT 3x4 AF #2A((60.0 80.0 100.0 120.0)
;;                   (120.0 160.0 200.0 240.0)
;;                   (180.0 240.0 300.0 360.0))>
;;  #<MAT 3x1 AF #2A((2.0) (4.0) (6.0))>)

(defun average! (a batch-size-tmp)
  (sum! a batch-size-tmp :axis 0)
  (scal! (/ 1.0 (mat-dimension a 0)) batch-size-tmp))

(defun softmax! (a result batch-size-tmp)
  ;; In order to avoid overflow, subtract average value for each column.
  (average! a batch-size-tmp)
  (fill! 1.0 result)
  (geerv! -1.0 result batch-size-tmp 1.0 a) ; a - average(a)

  (.exp! a)
  (sum! a batch-size-tmp :axis 0)
  (scale-columns! batch-size-tmp result)
  (.inv! result)
  (geem! 1.0 a result 0.0 result))

(defparameter a (make-mat '(3 2) :initial-contents '((0.3  1010)
                                                     (2.9  1000)
                                                     (4.0   990))))
(defparameter result (make-mat '(3 2)))
(defparameter batch-size-tmp (make-mat '(1 2)))

(softmax! a result batch-size-tmp)

;; #<MAT 3x2 B #2A((0.018211272 0.99995464)
;;                 (0.24519181 4.5397872e-5)
;;                 (0.7365969 2.0610602e-9))>


;;; cross-entropy

(defun cross-entropy! (y target tmp batch-size-tmp size-1-tmp)
  (let ((delta 1e-7)
        (batch-size (mat-dimension target 1)))
    (copy! y tmp)
    (.+! delta tmp)
    (.log! tmp)
    (geem! 1.0 target tmp 0.0 tmp)
    (sum! tmp batch-size-tmp :axis 0)
    (sum! batch-size-tmp size-1-tmp :axis 1)
    (/ (mat-as-scalar size-1-tmp) batch-size)))

(defparameter y (make-mat '(3 2) :initial-contents '((1.1 3.1)
                                                     (1.2 5.1)
                                                     (1.3 0.1))))

(defparameter target (make-mat '(3 2) :initial-contents '((1 0)
                                                          (0 1)
                                                          (0 0))))

(defparameter tmp (make-mat '(3 2)))
(defparameter batch-size-tmp (make-mat '(1 2)))
(defparameter size-1-tmp (make-mat '(1 1)))

(cross-entropy! y target tmp batch-size-tmp size-1-tmp) ; (/ (+ (log (+ 1.1 1e-7)) (log (+ 5.1 1e-7))) 2)

;;; 5.6.3 Softmax-with-loss

(define-class softmax/loss-layer (layer)
  loss y target batch-size-tmp size-1-tmp)

(defun make-softmax/loss-layer (input-dimensions)
  (make-instance 'softmax/loss-layer
                 :input-dimensions  input-dimensions
                 :output-dimensions 1
                 :backward-out (make-mat input-dimensions)
                 :y (make-mat input-dimensions)
                 :target (make-mat input-dimensions)
                 :batch-size-tmp (make-mat (list 1 (cadr input-dimensions)))
                 :size-1-tmp (make-mat '(1 1))))

(defparameter softmax/loss-layer1 (make-softmax/loss-layer '(3 2)))

(defmethod forward ((layer softmax/loss-layer) &rest inputs)
  (bind (((x target) inputs)
         (tmp (target layer)) ; use (target layer) as tmp
         (y (y layer))
         (batch-size-tmp (batch-size-tmp layer))
         (size-1-tmp (size-1-tmp layer)))
    (copy! x tmp)
    (softmax! tmp y batch-size-tmp)
    (let ((out (cross-entropy! y target tmp batch-size-tmp size-1-tmp)))
      (copy! target (target layer))
      (setf (forward-out layer) out)
      out)))

(defparameter x-softmax/loss
  (make-mat '(3 2) :initial-contents '((0.3  1010)
                                       (2.9  1000)
                                       (4.0   990))))
(defparameter target (make-mat '(3 2) :initial-contents '((1 0)
                                                          (0 1)
                                                          (0 0))))

(forward softmax/loss-layer1 x-softmax/loss target)
;; => -7.0017767

(defmethod backward ((layer softmax/loss-layer) dout)
  (let* ((target (target layer))
         (y      (y layer))
         (out    (backward-out layer))
         (batch-size (mat-dimension target 1)))
    (copy! y out)
    (axpy! -1.0 target out)
    (scal! (/ 1.0 batch-size) out)))

(backward softmax/loss-layer1 1.0)
;; #<MAT 3x2 AF #2A((-0.49089438 0.49997732)
;;                  (0.122595906 -0.4999773)
;;                  (0.36829844 1.0305301e-9))>
