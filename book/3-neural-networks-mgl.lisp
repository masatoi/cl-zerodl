(defparameter ma (make-mat '(2 2) :initial-contents '((1 -2) (-3 4))))
(defparameter mb (make-mat '(2 2) :initial-contents '((5 6) (7 8))))
(defparameter mc (make-mat '(2 2) :initial-element 0.0))

;; mc = 1.0 * ma * mb + 0.0 * mc
(gemm! 1.0 ma mb 0.0 mc) ; => #<MAT 2x2 AF #2A((19.0 22.0) (43.0 50.0))>
mc                       ; => #<MAT 2x2 AF #2A((19.0 22.0) (43.0 50.0))>

;;; When use CUDA
(setf *print-length* 10
      *print-level* 10)

(defparameter ma (make-mat '(10000 10000)))
(defparameter mb (make-mat '(10000 10000)))
(defparameter mc (make-mat '(10000 10000)))

(uniform-random! ma)
(uniform-random! mb)

(time (gemm! 1.0 ma mb 0.0 mc))

;; Evaluation took:
;;   6.539 seconds of real time
;;   26.092000 seconds of total run time (25.744000 user, 0.348000 system)
;;   399.02% CPU
;;   22,180,377,236 processor cycles
;;   0 bytes consed

(with-cuda* ()
  (time (gemm! 1.0 ma mb 0.0 mc)))

;; Evaluation took:
;;   0.427 seconds of real time
;;   0.424000 seconds of total run time (0.424000 user, 0.000000 system)
;;   99.30% CPU
;;   1,447,343,752 processor cycles
;;   0 bytes consed

(defparameter va (make-mat '(3 1) :initial-contents '((1) (2) (3))))
(defparameter vb (make-mat '(3 1) :initial-contents '((10) (20) (30))))

(defparameter x (make-mat '(3 1) :initial-contents '((1) (-2) (3))))
(defparameter r (make-mat '(3 1) :initial-element 0))

(.<! r x)

;; vb = 1.0 * va + vb
(axpy! 1.0 va vb) ; => #<MAT 3x1 ABF #2A((11.0) (22.0) (33.0))>
vb                ; => #<MAT 3x1 ABF #2A((11.0) (22.0) (33.0))>

;; Sigmoid
(defun sigmoid! (v)
  (.logistic! v))

;; ReLU
(defun relu! (v)
  (.max! 0.0 v))

;; 3.4 3層ニューラルネットワークの実装

(defparameter x  (make-mat '(2 1) :initial-contents '((1.0) (0.5))))
(defparameter W1 (make-mat '(3 2) :initial-contents '((0.1 0.2)
                                                      (0.3 0.4)
                                                      (0.5 0.6))))
(defparameter b1 (make-mat '(3 1) :initial-contents '((0.1) (0.2) (0.3))))
(defparameter z1 (make-mat '(3 1) :initial-element 0.0))

;; calc z1
(gemm! 1.0 W1 x 0.0 z1)
(axpy! 1.0 b1 z1) ; => #<MAT 3x1 AF #2A((0.3) (0.7) (1.1))>
(sigmoid! z1)     ; => #<MAT 3x1 ABF #2A((0.5744425) (0.66818774) (0.7502601))>

(defparameter W2 (make-mat '(2 3) :initial-contents '((0.1 0.2 0.3)
                                                      (0.4 0.5 0.6))))
(defparameter b2 (make-mat '(2 1) :initial-contents '((0.1) (0.2))))
(defparameter z2 (make-mat '(2 1) :initial-element 0.0))

(gemm! 1.0 W2 z1 0.0 z2)
(axpy! 1.0 b2 z2)
(sigmoid! z2)

(defparameter W3 (make-mat '(2 2) :initial-contents '((0.1 0.2)
                                                      (0.3 0.4))))
(defparameter b3 (make-mat '(2 1) :initial-contents '((0.1) (0.2))))
(defparameter z3 (make-mat '(2 1) :initial-element 0.0))

(gemm! 1.0 W3 z2 0.0 z3)
(axpy! 1.0 b3 z3) ; => #<MAT 2x1 AF #2A((0.3168271) (0.6962791))>

(time
 (loop repeat 1000 do
   (gemm! 1.0 W1 x 0.0 z1)
   (axpy! 1.0 b1 z1)
   (sigmoid! z1)
   (gemm! 1.0 W2 z1 0.0 z2)
   (axpy! 1.0 b2 z2)
   (sigmoid! z2)
   (gemm! 1.0 W3 z2 0.0 z3)
   (axpy! 1.0 b3 z3)))

;; Evaluation took:
;;   0.000 seconds of real time
;;   0.000000 seconds of total run time (0.000000 user, 0.000000 system)
;;   100.00% CPU
;;   691,968 processor cycles
;;   0 bytes consed
