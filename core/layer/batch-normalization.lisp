(defpackage #:cl-zerodl/core/layer/batch-normalization
  (:use #:cl
        #:mgl-mat
        #:cl-zerodl/core/layer/base)
  (:nicknames :zerodl.layer.batchnorm)
  (:import-from #:cl-zerodl/core/utils
                #:define-class
                #:average!)
  (:export #:batch-normalization-layer
           #:make-batch-normalization-layer))

(in-package #:cl-zerodl/core/layer/batch-normalization)

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
  (destructuring-bind (dx dbeta dgamma) (backward-out layer)
    (let ((epsilon (epsilon layer))
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
      dx)))
