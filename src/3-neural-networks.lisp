;;; 3 ニューラルネットワーク

;; 3.2.2 ステップ関数の実装

(defun step-function (x)
  (if (> x 0) 1 0))

;; 3.2.3 ステップ関数のグラフ

;; $ ros install masatoi/clgplot
(ql:quickload :clgplot)

(let* ((x (clgp:seq -5.0 5.0 0.1))
       (y (mapcar #'step-function x)))
  (clgp:plot y :x-seq x :y-range '(-0.1 1.1)))

;; 3.2.4 シグモイド関数の実装

(defun sigmoid (x)
  (/ 1 (+ 1 (exp (- x)))))

(mapcar #'sigmoid '(-1.0 1.0 2.0)) ; => (0.26894143 0.7310586 0.880797)

(let* ((x (clgp:seq -5.0 5.0 0.1))
       (y (mapcar #'sigmoid x)))
  (clgp:plot y :x-seq x :y-range '(-0.1 1.1)))

;; 3.2.5 シグモイド関数とステップ関数の比較

(let* ((x (clgp:seq -5.0 5.0 0.1))
       (y1 (mapcar #'step-function x))
       (y2 (mapcar #'sigmoid x)))
  (clgp:plots (list y1 y2) :x-seqs (list x x) :y-range '(-0.1 1.1)))

;; 3.2.7 ReLU関数

(defun relu (x)
  (max x 0))

(let* ((x (clgp:seq -5.0 5.0 0.1))
       (y (mapcar #'relu x)))
  (clgp:plot y :x-seq x :y-range '(-1 5)))

;; 3.3.2 行列の内積

(defparameter ma (mat '((1 2) (3 4))))
(array-dimensions ma) ; => (2 2)

(defparameter mb (mat '((5 6) (7 8))))
(array-dimensions mb) ; => (2 2)

(m* ma mb) ; => #2A((19 22) (43 50))

(defparameter ma (mat '((1 2 3) (4 5 6))))
(defparameter mb (mat '((1 2) (3 4) (5 6))))

(m* ma mb) ; => #2A((22 28) (49 64))

(defparameter ma (mat '((1 2) (3 4) (5 6))))
(defparameter mb (vec 7 8))

(m* ma mb) ; => #2A((23) (53) (83))

;; 3.4 3層ニューラルネットワークの実装

(defparameter x  (mat '((1.0) (0.5))))
(defparameter W1 (mat '((0.1 0.2) (0.3 0.4) (0.5 0.6))))
(defparameter b1 (vec 0.1 0.2 0.3))
(defparameter a1 (m+ (m* W1 x) b1))     ; => #2A((0.3) (0.7) (1.1))
(defparameter z1 (mapmat #'sigmoid a1)) ; => #2A((0.5744425) (0.66818774) (0.7502601))

(defparameter W2 (mat '((0.1 0.2 0.3) (0.4 0.5 0.6))))
(defparameter b2 (vec 0.1 0.2))

(array-dimensions z1) ; => (3 1)
(array-dimensions W2) ; => (2 3)
(array-dimensions b2) ; => (2 1)

(defparameter a2 (m+ (m* W2 z1) b2))
(defparameter z2 (mapmat #'sigmoid a2))

(defparameter W3 (mat '((0.1 0.2) (0.3 0.4))))
(defparameter b3 (vec 0.1 0.2))

(defparameter a3 (m+ (m* W3 z2) b3))

(defun identity-function (x) x)

(defparameter z3 (mapmat #'identity-function a3))

;; 3.4.3 実装のまとめ
(defun init-network ()
  (let ((network (make-hash-table)))
    (setf (gethash 'W1 network) (mat '((0.1 0.2) (0.3 0.4) (0.5 0.6)))
          (gethash 'b1 network) (vec 0.1 0.2 0.3)
          (gethash 'W2 network) (mat '((0.1 0.2 0.3) (0.4 0.5 0.6)))
          (gethash 'b2 network) (vec 0.1 0.2)
          (gethash 'W3 network) (mat '((0.1 0.2) (0.3 0.4)))
          (gethash 'b3 network) (vec 0.1 0.2))
    network))

(defun forward (network x)
  (destructuring-bind (W1 b1 W2 b2 W3 b3)
      (list (gethash 'W1 network) (gethash 'b1 network)
            (gethash 'W2 network) (gethash 'b2 network)
            (gethash 'W3 network) (gethash 'b3 network))
    (let* ((a1 (m+ (m* W1 x) b1))
           (z1 (mapmat #'sigmoid a1))
           (a2 (m+ (m* W2 z1) b2))
           (z2 (mapmat #'sigmoid a2))
           (a3 (m+ (m* W3 z2) b3))
           (y (mapmat #'identity-function a3)))
      y)))

(defparameter network (init-network))
(forward network x) ; => #2A((0.3168271) (0.6962791))

;;; 3.5 出力層の設計
