(defpackage #:cl-zerodl/core/layer/conv2d
  (:use #:cl
        #:mgl-mat
        #:cl-zerodl/core/layer/base)
  (:nicknames :zerodl.layer.conv2d)
  (:import-from #:cl-zerodl/core/utils
                #:define-class)
  (:import-from #:alexandria
                #:positive-integer)
  (:export #:conv2d-layer
           #:filter
           #:anchor
           #:stride
           #:make-conv2d-layer
           #:max-pool-layer
           #:pool-dimensions
           #:make-max-pool-layer))

(in-package #:cl-zerodl/core/layer/conv2d)

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
  (assert (and (listp input-dimensions) (= (length input-dimensions) 2)))
  (check-type filter-size positive-integer)
  (assert (oddp filter-size))
  (check-type stride-size positive-integer)
  (assert (and (< stride-size (first input-dimensions))
               (< stride-size (second input-dimensions))))

  (let ((input-dimensions (cons *batch-size* input-dimensions))
        (anchor-size (/ (1- filter-size) 2)))
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
  (destructuring-bind (dX dW) (backward-out layer)
    (let ((X (X layer))
          (W (filter layer)))

      (reshape! dX (input-dimensions layer))
      (reshape! dout (output-dimensions layer))

      (fill! 0.0 dX)
      (fill! 0.0 dW)

      (derive-convolve! X dX W dW dout
                        :start '(0 0) :stride (stride layer) :anchor (anchor layer) :batched t)

      (reshape-flatten! dX)
      (reshape-flatten! dout)
      
      (backward-out layer))))

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
  (destructuring-bind (dX) (backward-out layer)
    (let ((X (X layer))
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

      (backward-out layer))))
