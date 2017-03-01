;;; 2.3 パーセプトロンの実装

;; 2.3.1 簡単な実装
(defun pand (x1 x2)
  (let ((w1 0.5)
        (w2 0.5)
        (theta 0.7))
    (if (<= (+ (* w1 x1) (* w2 x2)) theta)
      0 1)))

(pand 1 0) ; => 0
(pand 0 1) ; => 0
(pand 1 1) ; => 1
(pand 0 0) ; => 0

;; 2.3.2 重みとバイアスの導入

(defun elementwise-* (lst1 lst2)
  (mapcar #'* lst1 lst2))

(defun sum (lst)
  (reduce #'+ lst))

(let* ((x '(0 1))
       (w '(0.5 0.5))
       (b -0.7)
       (w*x (elementwise-* x w))) ; => 0.5
  (+ (sum w*x) b)) ; => -0.19999999

;; 2.3.3 重みとバイアスによる実装

(defun pand (x1 x2)
  (let ((x (list x1 x2))
        (w (list 0.5 0.5))
        (b -0.7))
    (if (<= (+ (sum (elementwise-* x w)) b) 0)
      0 1)))

(defun pnand (x1 x2)
  (let ((x (list x1 x2))
        (w (list -0.5 -0.5)) ; 重みとバイアスだけが AND と違う!
        (b 0.7))
    (if (<= (+ (sum (elementwise-* x w)) b) 0)
      0 1)))

(defun por (x1 x2)
  (let ((x (list x1 x2))
        (w (list 0.5 0.5)) ; 重みとバイアスだけが AND と違う!
        (b -0.2))
    (if (<= (+ (sum (elementwise-* x w)) b) 0)
      0 1)))

;; 2.5.2 XORゲートの実装

(defun pxor (x1 x2)
  (let* ((s1 (pnand x1 x2))
         (s2 (por   x1 x2))
         (y  (pand  s1 s2)))
    y))

(pxor 1 0) ; => 1
(pxor 0 1) ; => 1
(pxor 1 1) ; => 0
(pxor 0 0) ; => 0
