(defpackage #:cl-zerodl/core/network
  (:use #:cl
        #:mgl-mat
        #:cl-zerodl/core/layer)
  (:nicknames :zerodl.network)
  (:import-from #:cl-zerodl/core/utils
                #:define-class)
  (:import-from #:cl-zerodl/core/initializer
                #:initializer
                #:he-initializer
                #:initialize!)
  (:import-from #:cl-zerodl/core/optimizer/base
                #:optimizer
                #:update!)
  (:import-from #:cl-zerodl/core/optimizer/sgd
                #:sgd)
  (:import-from #:alexandria
                #:positive-integer)
  (:export #:network
           ;; Network slots
           #:layers
           #:batch-size
           #:initializer
           #:optimizer
           ;; Constructor
           #:make-network
           ;; Utilities
           #:do-layer
           #:do-updatable-layer
           #:last-layer
           ;; Learning interfaces
           #:predict
           #:set-mini-batch!
           #:train
           #:accuracy
           ))

(in-package #:cl-zerodl/core/network)

;;; Network

(define-class network ()
  (layers :initform #()
          :type vector)
  (batch-size :initform 100
              :type positive-integer)
  (initializer :initform (make-instance 'he-initializer)
               :type initializer)
  (optimizer :initform (make-instance 'sgd)
             :type optimizer))

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

(defun make-network (layers
                     &key (batch-size 100)
                       (initializer (make-instance 'he-initializer))
                       (optimizer (make-instance 'sgd)))
  (assert (every (lambda (layer) (typep layer 'layer)) layers))
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
