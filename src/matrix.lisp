(defun matrixp (matrix)
  "Test whether the argument is a matrix"
  (and (arrayp matrix)
       (= (array-rank matrix) 2)))

(defun num-rows (matrix)
  "Return the number of rows of a matrix"
  (array-dimension matrix 0))

(defun num-cols (matrix)
  "Return the number of rows of a matrix"
  (array-dimension matrix 1))

(defun square-matrix? (matrix)
  "Is the matrix a square matrix?"
  (and (matrixp matrix)
       (= (num-rows matrix) (num-cols matrix))))
       
(defun make-matrix (rows &optional (cols rows))
  "Create a matrix filled with zeros.  If only one parameter is
  specified the matrix will be square."
  (make-array (list rows cols) :initial-element 0))

(defun make-identity-matrix (size)
  "Make an identity matrix of the specified size."
  (let ((matrix (make-array (list size size) :initial-element 0)))
    (dotimes (i size matrix)
      (setf (aref matrix i i) 1))))

(defun copy-matrix (matrix)
  "Return a copy of the matrix."
  (let* ((rows (num-rows matrix))
         (cols (num-cols matrix))
         (copy (make-array (list rows cols))))
    (dotimes (row rows copy)
      (dotimes (col cols)
        (setf (aref copy row col) (aref matrix row col))))))

(defun print-matrix (matrix &optional (destination t) (control-string " ~$"))
  "Print a matrix.  The optional control string indicates how each
  entry should be printed."
  (let ((rows (num-Rows matrix))
        (cols (num-Cols matrix)))
    (dotimes (row rows)
      (format destination "~%")
      (dotimes (col cols)
        (format destination control-string (aref matrix row col))))
    (format destination "~%")))

(defun transpose-matrix (matrix)
  "Transpose a matrix"
  (let* ((rows (num-rows matrix))
         (cols (num-cols matrix))
         (transpose (make-matrix cols rows)))
    (dotimes (row rows transpose)
      (dotimes (col cols)
        (setf (aref transpose col row)
              (aref matrix row col))))))

(defun multiply-scalar-matrix (scalar matrix)
  (let* ((rows (num-rows matrix))
	 (cols (num-cols matrix))
	 (result (make-matrix rows cols)))
    (dotimes (row rows result)
      (dotimes (col cols)
	(setf (aref result row col)
	      (* scalar (aref matrix row col)))))
    result))

(defun multiply-matrix (&rest matrices)
  "Multiply matrices"
  (labels ((multiply-two (m1 m2)
             (let* ((rows1 (num-rows m1))
                    (cols1 (num-cols m1))
                    (cols2 (num-cols m2))
                    (result (make-matrix rows1 cols2)))
               (dotimes (row rows1 result)
                 (dotimes (col cols2)
                   (dotimes (i cols1)
                     (setf (aref result row col)
                           (+ (aref result row col)
                              (* (aref m1 row i)
                                 (aref m2 i col))))))))))
    (when matrices                      ; Empty arguments check
      (reduce #'multiply-two matrices))))

(defun add-matrix (&rest matrices)
  "Add matrices"
  (labels ((add-two (m1 m2)
             (let* ((rows (num-rows m1))
                    (cols (num-cols m1))
                    (result (make-matrix rows cols)))
               (dotimes (row rows result)
                 (dotimes (col cols)
                   (setf (aref result row col)
                         (+ (aref m1 row col)
                            (aref m2 row col))))))))
    (when matrices                      ; Empty arguments check
      (reduce #'add-two matrices))))

(defun subtract-matrix (&rest matrices)
  "Subtract matrices"
  (labels ((subtract-two (m1 m2)
             (let* ((rows (num-rows m1))
                    (cols (num-cols m1))
                    (result (make-matrix rows cols)))
               (dotimes (row rows result)
                 (dotimes (col cols)
                   (setf (aref result row col)
                         (- (aref m1 row col)
                            (aref m2 row col))))))))
    (when matrices                      ; Empty arguments check
      (reduce #'subtract-two matrices))))

(defun invert-matrix (matrix &optional (destructive T))
  "Find the inverse of a matrix.  By default this operation is
  destructive.  If you want to preserve the original matrix, call this
  function with an argument of NIL to destructive."
  (let ((result (if destructive matrix (copy-matrix matrix)))
        (size (num-rows matrix))
        (temp 0))
    (dotimes (i size result)
      (setf temp (aref result i i))
      (dotimes (j size)
        (setf (aref result i j)
              (if (= i j)
                  (/ (aref result i j))
                  (/ (aref result i j) temp))))
      (dotimes (j size)
        (unless (= i j)
          (setf temp (aref result j i)
                (aref result j i) 0)
          (dotimes (k size)
            (setf (aref result j k)
                  (- (aref result j k)
                     (* temp (aref result i k))))))))))

(defun exchange-rows (matrix row-i row-j)
  "Exchange row-i and row-j of a matrix"
  (let ((cols (num-cols matrix)))
    (dotimes (col cols)
      (rotatef (aref matrix row-i col) (aref matrix row-j col)))))


(defun eliminate-matrix (matrix rows cols)
  "Gaussian elimination with partial pivoting.  "
  ;; Evaluated for side effect.  A return value of :singular indicates the
  ;; matrix is singular (an error).
  (let ((max 0))
    (loop for i below rows
     do (setf max i)
     do (loop for j from (1+ i) below rows
         do (when (> (abs (aref matrix j i))
                     (abs (aref matrix max i)))
              (setf max j)))
     do (when (zerop (aref matrix max i))
          (return-from eliminate-matrix :singular)) ; error "Singular matrix"
     do (loop for k from i below cols   ; Exchange rows
         do (rotatef (aref matrix i k) (aref matrix max k)))
     do (loop for j from (1+ i) below rows
         do (loop for k from (1- cols) downto i
             do (setf (aref matrix j k)
                      (- (aref matrix j k)
                         (* (aref matrix i k)
                            (/ (aref matrix j i)
                               (aref matrix i i)))))
               )))
    matrix))

(defun substitute-matrix (matrix rows cols)
  (let ((temp 0.0)
        (x (make-array rows :initial-element 0)))
    (loop for j from (1- rows) downto 0
     do (setf temp 0.0)
     do (loop for k from (1+ j) below rows
         do (incf temp (* (aref matrix j k) (aref x k))))
     do (setf (aref x j) (/ (- (aref matrix j (1- cols)) temp) 
                            (aref matrix j j))))
    x))

(defun solve-matrix (matrix &optional (destructive T) print-soln)
  "Solve a matrix using Gaussian elimination
   Matrix must be N by N+1
   Assume solution is stored as the N+1st column of the matrix"
  (let ((rows (num-rows matrix))
        (cols  (num-cols matrix))
        (result (if destructive matrix (copy-matrix matrix))))
    (unless (= (1+ rows) cols)
      (error "Ill formed matrix"))      ; Cryptic error message
    (cond ((eq :singular (eliminate-matrix result rows cols)))
          (T (let ((soln (substitute-matrix result rows cols)))
               (when print-soln
                 (loop for i below rows
                  do (format t "~% X~A = ~A" i (aref soln i))))
               soln)))))

(defun trace-matrix (matrix)
  (if (not (square-matrix? matrix))
      (error "Ill formed matrix")
      (let ((rows (num-rows matrix)))	    
	(loop for i from 0  to (1- rows) sum (aref matrix i i)))))

(defun partial-matrix (matrix rows cols)
  (let ((mat (make-array (list rows cols))))
    (dotimes (i rows)
      (dotimes (j cols)
	(setf (aref mat i j) (aref matrix i j))))
    mat))

;;; vector utilities ==========================================================
(defun make-vector (len &key (element-type t) (initial-element 0.0d0) (initial-contents nil))
  (if initial-contents
      (let ((ini-con (mapcar (lambda (x) (cons x ())) initial-contents)))
	(make-array (list len 1) :element-type element-type :initial-contents ini-con))
      (make-array (list len 1) :element-type element-type :initial-element initial-element)))

(defun list->vector (lst)
  (make-vector (length lst) :initial-contents lst))

(defun simple-vector->arrayed-vector (simple-vector &optional (vertical? t))
  (if vertical?
      (let ((arr (make-array (list (length simple-vector) 1))))
	(loop for i from 0 to (1- (length simple-vector)) do
	     (setf (aref arr i 0) (svref simple-vector i)))
	arr)
      (let ((arr (make-array (list 1 (length simple-vector)))))
	(loop for j from 0 to (1- (length simple-vector)) do
	     (setf (aref arr 0 j) (svref simple-vector j)))
	arr)))

(defun vector-cat (v1 v2)
  (let* ((v1-len (array-dimension v1 0))
	 (v2-len (array-dimension v2 0))
	 (len (+ v1-len v2-len))
	 (v-new (make-array (list len 1))))
    (loop for i from 0 to (1- v1-len) do
      (setf (aref v-new i 0) (aref v1 i 0)))
    (loop for i from v1-len to (1- len) do
      (setf (aref v-new i 0) (aref v2 (- i v1-len) 0)))
    v-new))

(defun vector-cat2 (v1 v2)
  (let ((gv1 (if (arrayp v1) v1 (make-vector 1 :initial-element v1)))
	(gv2 (if (arrayp v2) v2 (make-vector 1 :initial-element v2))))
    (vector-cat gv1 gv2)))

(defun vector-length (vec)
  (if (numberp vec) 1
      (array-dimension vec 0)))

(defun euclidean-norm (vec)
  (sqrt
   (summation (i 0 (1- (vector-length vec)))
     (let ((elem (aref vec i 0)))
       (* elem elem)))))


;;; matrix utilities ==========================================================

(defmacro nlet (tag var-vals &body body)
  `(labels ((,tag ,(mapcar #'car var-vals) ,@body))
     (declare (optimize (speed 3))) ; for tail recursion optimization
     (,tag ,@(mapcar #'cadr var-vals))))

(defun m* (&rest args)
  (nlet itf ((prod (car args))
	     (args (cdr args)))
    (if (null args)
	prod
	(cond ((and (numberp prod) (numberp (car args)))
	       (itf (* prod (car args)) (cdr args)))
	      ((numberp prod)
	       (itf (multiply-scalar-matrix prod (car args)) (cdr args)))
	      ((numberp (car args))
	       (itf (multiply-scalar-matrix (car args) prod) (cdr args)))
	      (t (itf (multiply-matrix prod (car args)) (cdr args)))))))

(defun m+ (&rest matrices)  
  (apply #'add-matrix matrices))

(defun ssum-vec (vec-lst)
  (reduce #'m+ vec-lst :initial-value (make-vector (length vec-lst) :initial-element 0.0d0)))

(defun m- (&rest matrices)  
  (apply #'subtract-matrix matrices))

(defun m-t (mat)
  (transpose-matrix mat))

(defun umat (size)
  (let ((matrix (make-array (list size size)
			    :initial-element 0d0 :element-type 'double-float)))
    (dotimes (i size matrix)
      (setf (aref matrix i i) 1d0))))

(defun zero-mat (size)
  (make-array (list size size) :initial-element 0d0))

(defun m-1 (mat)
  (invert-matrix mat NIL))

(defun m-append-horizon (m1 m2)
  (let* ((m1-dims (array-dimensions m1))
	 (m2-dims (array-dimensions m2))
	 (product (make-array (list (car m1-dims) (+ (cadr m1-dims) (cadr m2-dims))))))
    (if (not (= (car m1-dims) (car m2-dims)))
	(print "Error: wrong matrix size.")
	(progn
	  (loop for i from 0 to (1- (car m1-dims)) do
	    (loop for j from 0 to (1- (cadr m1-dims)) do
	      (setf (aref product i j) (aref m1 i j)))
	    (loop for j from (cadr m1-dims) to (1- (+ (cadr m1-dims) (cadr m2-dims))) do
	      (setf (aref product i j) (aref m2 i (- j (cadr m1-dims))))))
	  product))))

(defun vec (&rest elements)
  (make-vector (length elements) :initial-contents elements))

(defun mat (contents-list &key (element-type t))
  (let ((row-len (length contents-list))
	(col-len (length (car contents-list))))
    (make-array (list row-len col-len) 
		:element-type element-type
		:initial-contents contents-list)))

(defun mapmat (proc matrix)
  (let ((m (make-array (array-dimensions matrix))))
    (loop for i from 0 to (1- (array-dimension matrix 0)) do
	 (loop for j from 0 to (1- (array-dimension matrix 1)) do
	      (setf (aref m i j) (funcall proc (aref matrix i j)))))
    m))

(defun mapvec (proc &rest vectors)
  (let ((v (make-array (array-dimensions (car vectors)))))
    (loop for i from 0 to (1- (array-dimension (car vectors) 0))
       do (setf (aref v i 0)
		(apply proc (mapcar (lambda (vec) (aref vec i 0)) vectors))))
    v))
