;;; -*- coding:utf-8; mode:lisp -*-

(in-package :cl-zerodl)

;; 5.4 単純なレイヤーの実装

(define-class layer ()
  input-dimensions output-dimensions
  forward-out backward-out)

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
