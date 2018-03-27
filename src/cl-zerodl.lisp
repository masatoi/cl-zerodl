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
         (batch-size (mat-dimension x 1))
         (W (weight layer))
         (b (bias layer))
         (bs (stacked-bias layer))
         (out (forward-out layer)))
    (copy! x (x layer))
    (stack! 1 (loop repeat batch-size collect b) bs)
    (gemm! 1.0 W x 0.0 out)
    (axpy! 1.0 bs out)))

(setf (weight affine-layer1)
      (make-mat '(3 4) :initial-contents '((1 4 7 10) (2 5 8 11) (3 6 9 12))))
(setf (bias affine-layer1) (make-mat '(3 1) :initial-contents '((1) (2) (3))))
(defparameter x-affine (make-mat '(4 2) :initial-contents '((10 50) (20 60) (30 70) (40 80))))

(print (forward affine-layer1 x-affine))

;; #<MAT 3x2 AF #2A((131.0 181.0) (172.0 242.0) (213.0 303.0))> 

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
