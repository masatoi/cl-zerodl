;;; -*- coding:utf-8; mode:lisp -*-

(in-package :cl-user)

(defpackage cl-zerodl
  (:use :cl :mgl-mat :metabang.bind :alexandria)
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

(define-class layer ()
  input-dimensions output-dimensions
  forward-out backward-out)

(define-class updatable-layer (layer)
  updatable-parameters
  gradients)

(defgeneric forward (layer &rest inputs))
(defgeneric backward (layer dout))

;; 5.5 Activation function layer
;; 5.5.1 Relu

(define-class relu-layer (layer)
  zero mask)

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

(defmethod backward ((layer relu-layer) dout)
  (geem! 1.0 dout (mask layer) 0.0 (backward-out layer)))

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

;; 5.6 Affine

(define-class affine-layer (updatable-layer)
  x weight bias)

;; x: (batch-size, in-size)
;; y: (batch-size, out-size)
;; W: (in-size,    out-size)
;; b: (out-size)

(defun make-affine-layer (input-dimensions output-dimensions)
  (let* ((weight-dimensions (list (cadr input-dimensions) (cadr output-dimensions)))
         (bias-dimension (cadr output-dimensions))
         (layer (make-instance 'affine-layer
                               :input-dimensions  input-dimensions
                               :output-dimensions output-dimensions
                               :forward-out  (make-mat output-dimensions)
                               :backward-out (list (make-mat input-dimensions)  ; dX
                                                   (make-mat weight-dimensions) ; dW
                                                   (make-mat bias-dimension))   ; db
                               :x      (make-mat input-dimensions)
                               :weight (make-mat weight-dimensions)
                               :bias   (make-mat bias-dimension))))
    (setf (updatable-parameters layer) (list (weight layer) (bias layer))
          (gradients layer)            (cdr (backward-out layer)))
    layer))

(defmethod forward ((layer affine-layer) &rest inputs)
  (let* ((x (car inputs))
         (W (weight layer))
         (b (bias layer))
         (out (forward-out layer)))
    (copy! x (x layer))
    (fill! 1.0 out)
    (scale-columns! b out)
    (gemm! 1.0 x W 1.0 out)))

(defmethod backward ((layer affine-layer) dout)
  (bind (((dx dW db) (backward-out layer)))
    (gemm! 1.0 dout (weight layer) 0.0 dx :transpose-b? t) ; dx
    (gemm! 1.0 (x layer) dout 0.0 dW :transpose-a? t)      ; dW
    (sum! dout db :axis 0)                                 ; db
    (backward-out layer)))

(defun average! (a batch-size-tmp &key (axis 0))
  (sum! a batch-size-tmp :axis axis)
  (scal! (/ 1.0 (mat-dimension a axis)) batch-size-tmp))

(defun softmax! (a result batch-size-tmp &key (avoid-overflow-p t))
  ;; In order to avoid overflow, subtract average value for each column.
  (when avoid-overflow-p
    (average! a batch-size-tmp :axis 1)
    (fill! 1.0 result)
    (scale-rows! batch-size-tmp result)
    (axpy! -1.0 result a)) ; a - average(a)
  (.exp! a)
  (sum! a batch-size-tmp :axis 1)
  (fill! 1.0 result)
  (scale-rows! batch-size-tmp result)
  (.inv! result)
  (.*! a result))

;;; cross-entropy

(defun cross-entropy! (y target tmp batch-size-tmp &key (delta 1e-7))
  (let ((batch-size (mat-dimension target 0)))
    (copy! y tmp)
    (.+! delta tmp)
    (.log! tmp)
    (.*! target tmp)
    (sum! tmp batch-size-tmp :axis 1)
    (/ (asum batch-size-tmp) batch-size)))

;;; 5.6.3 Softmax-with-loss

(define-class softmax/loss-layer (layer)
  y target batch-size-tmp)

(defun make-softmax/loss-layer (input-dimensions)
  (make-instance 'softmax/loss-layer
                 :input-dimensions  input-dimensions
                 :output-dimensions 1
                 :backward-out (make-mat input-dimensions)
                 :y            (make-mat input-dimensions)
                 :target       (make-mat input-dimensions)
                 :batch-size-tmp (make-mat (car input-dimensions))))

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

(defmethod backward ((layer softmax/loss-layer) dout)
  (let* ((target (target layer))
         (y      (y layer))
         (out    (backward-out layer))
         (batch-size (mat-dimension target 0)))
    (copy! y out)
    (axpy! -1.0 target out)
    (scal! (/ 1.0 batch-size) out)))

;;; 7.2 conv2d layer

(defun reshape-flatten! (mat)
  (let ((dims (mat-dimensions mat)))
    (reshape! mat (list (car dims) (reduce #'* (cdr dims))))))

(define-class conv2d-layer (updatable-layer)
  X filter anchor stride)

;; X: (batch-size, in1-size, in2-size)
;; Y: (batch-size, out1-size, out2-size)
;; W: (filter-x, filter-y)

(defun make-conv2d-layer (input-dimensions filter-size stride-size)
  (assert (and (listp input-dimensions) (= (length input-dimensions) 3)))
  (check-type filter-size positive-integer)
  (assert (oddp filter-size))
  (check-type stride-size positive-integer)
  (assert (and (< stride-size (second input-dimensions))
               (< stride-size (third input-dimensions))))

  (let ((anchor-size (/ (1- filter-size) 2)))
    (flet ((out-dim (in-dim)
             (1+ (/ (+ in-dim (* anchor-size 2) (- filter-size)) stride-size))))
      (let* ((out1 (out-dim (second input-dimensions)))
             (out2 (out-dim (third input-dimensions)))
             (layer (make-instance 'conv2d-layer
                                   :input-dimensions  input-dimensions
                                   :output-dimensions (list (first input-dimensions) out1 out2)
                                   :forward-out (make-mat (list (first input-dimensions) (* out1 out2)))
                                   :backward-out (list (make-mat (list (first input-dimensions) ; dX
                                                                       (* (second input-dimensions)
                                                                          (third input-dimensions))))
                                                       (make-mat (list filter-size filter-size))) ; dW
                                   :X      (make-mat input-dimensions)
                                   :filter (make-mat (list filter-size filter-size))
                                   :anchor (list anchor-size anchor-size)
                                   :stride (list stride-size stride-size))))
        (setf (updatable-parameters layer) (list (filter layer))
              (gradients layer)            (cdr (backward-out layer)))
        layer))))

(defmethod forward ((layer conv2d-layer) &rest inputs)
  (let* ((X (car inputs))
         (W (filter layer))
         (Y (forward-out layer)))

    (reshape! X (input-dimensions layer))
    (copy! X (X layer))
    (reshape! Y (output-dimensions layer))
    
    (fill! 0.0 Y)
    (convolve! X W Y :start '(0 0) :stride (stride layer) :anchor (anchor layer) :batched t)

    (reshape-flatten! X)
    (reshape-flatten! Y)))

(defmethod backward ((layer conv2d-layer) dout)
  (bind (((dX dW) (backward-out layer))
         (X (X layer))
         (W (filter layer)))

    (reshape! dX (input-dimensions layer))
    (reshape! dout (output-dimensions layer))

    (fill! 0.0 dX)
    (fill! 0.0 dW)

    (derive-convolve! X dX W dW dout
                      :start '(0 0) :stride (stride layer) :anchor (anchor layer) :batched t)

    (reshape-flatten! dX)
    (reshape-flatten! dout)
    
    (backward-out layer)))

;;; 7.3 max-pool layer

(define-class max-pool-layer (layer)
  X pool-dimensions)

(defun make-max-pool-layer (input-dimensions output-dimensions pool-dimensions)
  (assert (and (listp input-dimensions) (= (length input-dimensions) 3)))
  (assert (and (listp output-dimensions) (= (length output-dimensions) 3)))
  (assert (and (listp pool-dimensions) (= (length pool-dimensions) 2)))

  (make-instance 'max-pool-layer
                 :input-dimensions  input-dimensions
                 :output-dimensions output-dimensions
                 :forward-out (make-mat output-dimensions)
                 :backward-out (list (make-mat input-dimensions))
                 :X      (make-mat input-dimensions)
                 :pool-dimensions pool-dimensions))

(defmethod forward ((layer max-pool-layer) &rest inputs)
  (let* ((X (car inputs))
         (Y (forward-out layer)))

    (reshape! X (input-dimensions layer))
    (copy! X (X layer))
    (reshape! Y (output-dimensions layer))
    
    (fill! 0.0 Y)
    (max-pool! X Y :start '(0 0)
                   :stride (pool-dimensions layer)
                   :anchor '(0 0)
                   :batched t
                   :pool-dimensions (pool-dimensions layer))

    (reshape-flatten! X)
    (reshape-flatten! Y)))

(defmethod backward ((layer max-pool-layer) dout)
  (bind (((dX) (backward-out layer))
         (X (X layer))
         (Y (forward-out layer)))

    (reshape! dX (input-dimensions layer))
    (reshape! dout (output-dimensions layer))
    (reshape! Y (output-dimensions layer))

    (fill! 0.0 dX)

    (derive-max-pool! X dX Y dout :start '(0 0)
                                  :stride (pool-dimensions layer)
                                  :anchor '(0 0)
                                  :batched t
                                  :pool-dimensions (pool-dimensions layer))
    
    (reshape-flatten! dX)
    (reshape-flatten! dout)
    (reshape-flatten! Y)

    (backward-out layer)))

;;; Network operations

;;; Network

(define-class network ()
  layers batch-size initializer optimizer)

(defun spec->layer (spec batch-size)
  (let ((input-dimensions  (list batch-size (getf (cdr spec) :in)))
        (output-dimensions (list batch-size (getf (cdr spec) :out))))
    (ecase (car spec)
      (affine  (make-affine-layer  input-dimensions output-dimensions))
      ;; (conv2d (make-conv2d-layer input-dimensions ))
      (batch-norm  (make-batch-normalization-layer input-dimensions))
      (dropout  (make-dropout-layer input-dimensions))
      (relu    (make-relu-layer    input-dimensions))
      (sigmoid (make-sigmoid-layer input-dimensions))
      (softmax (make-softmax/loss-layer input-dimensions)))))

(defmacro do-layer ((layer network type) &body body)
  `(loop for ,layer across (layers ,network) do
    (when (eq (type-of ,layer) (quote ,type))
      ,@body)))

(defmacro do-updatable-layer ((layer network) &body body)
  `(loop for ,layer across (layers ,network) do
    (when (slot-exists-p ,layer 'updatable-parameters)
      ,@body)))

(defun update-network! (network)
  (do-updatable-layer (layer network)
    (mapc (lambda (param grad)
            (update! (optimizer network) param grad))
          (updatable-parameters layer)
          (gradients layer))))

(defun initialize-network! (network)
  (do-layer (layer network affine-layer)
    (initialize! (initializer network) (weight layer)))
  (do-layer (layer network conv2d-layer)
    (initialize! (initializer network) (filter layer))))

(defun make-network (layer-specs
                     &key (batch-size 100)
                       (initializer (make-instance 'gaussian-initializer))
                       (optimizer   (make-instance 'sgd)))
  (let ((network (make-instance
                  'network
                  :layers (map 'vector
                               (lambda (spec) (spec->layer spec batch-size))
                               layer-specs)
                  :batch-size  batch-size
                  :initializer initializer
                  :optimizer   optimizer)))
    (initialize-network! network)
    network))

(defun make-network (layers
                     &key (batch-size 100)
                       (initializer (make-instance 'gaussian-initializer))
                       (optimizer   (make-instance 'sgd)))
  (let ((network (make-instance
                  'network
                  :layers layers
                  :batch-size  batch-size
                  :initializer initializer
                  :optimizer   optimizer)))
    (initialize-network! network)
    network))

(defun last-layer (network)
  (aref (layers network) (1- (length (layers network)))))

(defun predict (network x)
  (let* ((layers (layers network))
         (len (length layers)))
    (loop for i from 0 below (1- len) do
      (setf x (forward (aref layers i) x)))
    x))

(defun loss (network x target)
  (let ((y (predict network x)))
    (forward (last-layer network) y target)))

;; Calculate gradient

(defmethod set-gradient! ((network network) x target)
  (let ((layers (layers network))
        dout)
    ;; forward
    (loss network x target)
    ;; backward
    (setf dout (backward (last-layer network) 1.0))
    (loop for i from (- (length layers) 2) downto 0 do
      (let ((layer (svref layers i)))
        (setf dout (backward layer (if (listp dout) (car dout) dout)))))
    ;; ;; weight-decay
    ;; (do-layer (layer network affine-layer)
    ;;   (let ((dW (cadr (backward-out layer))))
    ;;     (axpy! 0.00001 (weight layer) dW))
    ;;   )
    ))

(defun weight-decay-network! (network regularization-rate)
  (do-layer (layer network affine-layer)
    (axpy! regularization-rate (weight layer) (weight layer))))

;;; Optimizer

(define-class optimizer ())

(define-class sgd (optimizer)
  (learning-rate 0.1))

(defmethod update! ((optimizer sgd) parameter gradient)
  (axpy! (- (learning-rate optimizer)) gradient parameter))

;; Momentum SGD
(define-class momentum-sgd (sgd)
  velocities decay-rate)

(defun make-momentum-sgd (learning-rate decay-rate network)
  (let ((opt (make-instance 'momentum-sgd
                            :learning-rate learning-rate
                            :velocities (make-hash-table :test 'eq)
                            :decay-rate decay-rate)))
    (do-updatable-layer (layer network)
      (dolist (param (updatable-parameters layer))
        (setf (gethash param (velocities opt))
              (make-mat (mat-dimensions param) :initial-element 0.0))))
    opt))

(defmethod update! ((optimizer momentum-sgd) parameter gradient)
  (let ((velocity (gethash parameter (velocities optimizer))))
    (scal! (decay-rate optimizer) velocity)
    (axpy! (- (learning-rate optimizer)) gradient velocity)
    (axpy! 1.0 velocity parameter)))

;; AggMo (Aggregated Momentum)
;; https://arxiv.org/pdf/1804.00325.pdf

(define-class aggmo (sgd)
  velocity-hash-list decay-rate-list)

(defun make-aggmo (learning-rate decay-rate-list network)
  (let ((opt (make-instance 'aggmo
                            :learning-rate learning-rate
                            :velocity-hash-list (loop repeat (length decay-rate-list)
                                                      collect (make-hash-table :test 'eq))
                            :decay-rate-list decay-rate-list)))
    (do-updatable-layer (layer network)
      (dolist (param (updatable-parameters layer))
        (loop for decay-rate in decay-rate-list
              for velocity-hash in (velocity-hash-list opt)
              do (setf (gethash param velocity-hash)
                       (make-mat (mat-dimensions param) :initial-element 0.0)))))
    opt))

(defmethod update! ((optimizer aggmo) parameter gradient)
  (let ((v-list (mapcar (lambda (hash) (gethash parameter hash))
                        (velocity-hash-list optimizer)))
        (K (length (decay-rate-list optimizer))))
    (loop for v in v-list
          for decay in (decay-rate-list optimizer)
          do (scal! decay v)
             (axpy! (- (learning-rate optimizer)) gradient v))
    (dolist (v v-list)
      (axpy! (/ 1.0 K) v parameter))))

;; Adagrad
(define-class adagrad (sgd)
  velocities decay-rate tmps epsilon)

(defun make-adagrad (learning-rate decay-rate network &key (epsilon 1.0e-6))
  (let ((opt (make-instance 'adagrad
                            :learning-rate learning-rate
                            :velocities (make-hash-table :test 'eq)
                            :decay-rate decay-rate
                            :tmps (make-hash-table :test 'eq)
                            :epsilon epsilon)))
    (do-updatable-layer (layer network)
      (dolist (param (updatable-parameters layer))
        (setf (gethash param (velocities opt))
              (make-mat (mat-dimensions param) :initial-element 0.0)
              (gethash param (tmps opt))
              (make-mat (mat-dimensions param) :initial-element 0.0))))
    opt))

(defmethod update! ((optimizer adagrad) parameter gradient)
  (let ((velocity (gethash parameter (velocities optimizer)))
        (tmp (gethash parameter (tmps optimizer))))
    (geem! 1.0 gradient gradient 1.0 velocity)
    (copy! velocity tmp)
    (.+! (epsilon optimizer) tmp)
    (.sqrt! tmp)
    (.inv! tmp)
    (.*! gradient tmp)
    (axpy! (- (learning-rate optimizer)) tmp parameter)))

;;; Initializer
(define-class initializer ())

(define-class gaussian-initializer (initializer)
  (weight-init-std 0.01))

(defmethod initialize! ((initializer gaussian-initializer) parameter)
  (gaussian-random! parameter :stddev (weight-init-std initializer)))

(define-class xavier-initializer (initializer))

(defmethod initialize! ((initializer xavier-initializer) parameter)
  (gaussian-random! parameter :stddev (sqrt (/ 2.0 (+ (mat-dimension parameter 0)
                                                      (mat-dimension parameter 1))))))

(define-class he-initializer (initializer))

(defmethod initialize! ((initializer he-initializer) parameter)
  (gaussian-random! parameter :stddev (sqrt (/ 2.0 (mat-dimension parameter 0)))))

;;; Read data

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

;;; Set/Reset mini-batch

(defun set-mini-batch! (dataset start-row-index batch-size)
  (let ((dim (mat-dimension dataset 1)))
    (reshape-and-displace! dataset
                           (list batch-size dim)
                           (* start-row-index dim))))

(defun reset-shape! (dataset)
  (let* ((dim (mat-dimension dataset 1))
         (len (/ (mat-max-size dataset) dim)))
    (reshape-and-displace! dataset (list len dim) 0)))

;;; Training

(defun train (network x target)
  (set-gradient! network x target)
  (update-network! network))

;;; Predict class, Accuracy for dataset

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

;;; Batch Normalization

(define-class batch-normalization-layer (updatable-layer)
  epsilon beta gamma var sqrtvar ivar x^ xmu tmp)

(defun make-batch-normalization-layer (input-dimensions &key (epsilon 1.0e-6))
  (let* ((dim (cadr input-dimensions))
         (layer (make-instance 'batch-normalization-layer
                               :input-dimensions  input-dimensions
                               :output-dimensions input-dimensions
                               :forward-out  (make-mat input-dimensions)
                               :backward-out (list (make-mat input-dimensions) ; dX
                                                   (make-mat dim)              ; dβ
                                                   (make-mat dim))             ; dγ
                               :epsilon epsilon
                               :beta    (make-mat dim :initial-element 0.0)
                               :gamma   (make-mat dim :initial-element 1.0)
                               :var     (make-mat dim)
                               :sqrtvar (make-mat dim)
                               :ivar    (make-mat dim)
                               :x^      (make-mat input-dimensions)
                               :xmu     (make-mat input-dimensions)
                               :tmp     (make-mat input-dimensions))))
    (setf (updatable-parameters layer) (list (beta layer) (gamma layer))
          (gradients layer)            (cdr (backward-out layer)))
    layer))

(defmethod forward ((layer batch-normalization-layer) &rest inputs)
  (let ((x       (car inputs))
        (epsilon (epsilon layer))
        (beta    (beta    layer))
        (gamma   (gamma   layer))
        (var     (var     layer))
        (sqrtvar (sqrtvar layer))
        (ivar    (ivar    layer))
        (x^      (x^      layer))
        (xmu     (xmu     layer))
        (tmp     (tmp     layer))
        (out     (forward-out layer)))
    (average! x (ivar layer)) ; use ivar as tmp
    ;; calc xmu
    (fill! 1.0 xmu)
    (scale-columns! ivar xmu)
    (axpy! -1.0 x xmu)
    (scal! -1.0 xmu)
    ;; calc var
    (copy! xmu x^) ; use x^ as tmp
    (.square! x^)
    (average! x^ var)
    ;; calc sqrtvar
    (copy! var sqrtvar)
    (.+! epsilon sqrtvar)
    (.sqrt! sqrtvar)
    ;; calc ivar
    (copy! sqrtvar ivar)
    (.inv! ivar)
    ;; calc x^
    (fill! 1.0 x^)
    (scale-columns! ivar x^)
    (.*! xmu x^)
    ;; calc output
    (fill! 1.0 tmp)
    (scale-columns! gamma tmp)
    (.*! x^ tmp)
    (fill! 1.0 out)
    (scale-columns! beta out)
    (axpy! 1.0 tmp out)))

(defmethod backward ((layer batch-normalization-layer) dout)
  (bind (((dx dbeta dgamma) (backward-out layer))
         (epsilon (epsilon layer))
         (gamma   (gamma   layer))
         (var     (var     layer))
         (sqrtvar (sqrtvar layer))
         (ivar    (ivar    layer))
         (x^      (x^      layer))
         (xmu     (xmu     layer))
         (tmp     (tmp     layer)))
    ;; calc dx^ -> tmp
    (fill! 1.0 tmp)
    (scale-columns! gamma tmp)
    (.*! dout tmp)
    ;; calc dxmu1 -> dx
    (fill! 1.0 dx)
    (scale-columns! ivar dx)
    (.*! tmp dx)
    ;; calc divar -> dbeta
    (.*! xmu tmp)
    (sum! tmp dbeta :axis 0)
    ;; calc dsqrtvar -> dbeta
    (copy! sqrtvar dgamma)
    (.square! dgamma)
    (.inv! dgamma)
    (geem! -1.0 dbeta dgamma 0.0 dbeta)
    ;; calc dvar -> dbeta
    (copy! var dgamma)
    (.+! epsilon dgamma)
    (.sqrt! dgamma)
    (.inv! dgamma)
    (geem! 0.5 dbeta dgamma 0.0 dbeta)
    ;; calc dsq -> tmp
    (fill! 1.0 tmp)
    (scale-columns! dbeta tmp)
    (scal! (/ 1.0 (mat-dimension tmp 0)) tmp)
    ;; calc dxmu2 -> tmp
    (geem! 2.0 xmu tmp 0.0 tmp)
    ;; calc dx1 -> dx
    (axpy! 1.0 tmp dx)
    ;; calc -dmu -> dbeta
    (sum! dx dbeta :axis 0)
    ;; calc dx2 -> tmp
    (fill! 1.0 tmp)
    (scale-columns! dbeta tmp)
    (scal! (/ -1.0 (mat-dimension tmp 0)) tmp)
    ;; calc dx
    (axpy! 1.0 tmp dx)
    ;; calc dbeta
    (sum! dout dbeta :axis 0)
    ;; calc dgamma
    (geem! 1.0 dout x^ 0.0 tmp)
    (sum! tmp dgamma :axis 0)
    dx))

;;; Dropout layer

(define-class dropout-layer (layer)
  mask threshold in-train?)

(defun make-dropout-layer (input-dimensions &key (dropout-rate 0.5))
  (make-instance 'dropout-layer
                 :input-dimensions  input-dimensions
                 :output-dimensions input-dimensions
                 :forward-out  (make-mat input-dimensions)
                 :backward-out (make-mat input-dimensions)
                 :mask         (make-mat input-dimensions :initial-element 0.0)
                 :threshold    (make-mat input-dimensions :initial-element dropout-rate)))

(defmethod forward ((layer dropout-layer) &rest inputs)
  (let ((x (car inputs))
        (mask (mask layer))
        (threshold (threshold layer))
        (out (forward-out layer)))

    ;; set mask
    (uniform-random! mask)
    (.<! threshold mask)
    ;; set output
    (geem! 1.0 mask x 0.0 out)))

(defmethod backward ((layer dropout-layer) dout)
  (geem! 1.0 dout (mask layer) 0.0 (backward-out layer)))
