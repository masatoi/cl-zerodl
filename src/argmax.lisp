(in-package :mgl-mat)

(defun blas-isamax (n x &key (incx 1))
  (cffi:with-foreign-object (len :int)
    (setf (cffi:mem-ref len :int) n)
    (with-facets ((mat-ptr (x 'foreign-array :direction :io)))
      (let ((mat-ptr (offset-pointer mat-ptr)))
        (cffi:with-foreign-object (offset :int)
          (setf (cffi:mem-ref offset :int) incx)
          (cffi:foreign-funcall "isamax_"
                                (:pointer :int)   len
                                (:pointer :float) mat-ptr
                                (:pointer :int)   offset
                                :int ; return type
                                ))))))

(defun argmax! (mat result)
  (let ((len (mat-dimension mat 0))
        (dim (mat-dimension mat 1))
        (dis (mat-displacement mat)))
    (loop for i from 0 below len do
      (reshape-and-displace! mat dim (+ (* i dim) dis))
      (setf (aref result i) (1- (blas-isamax dim mat))))
    (reshape-and-displace! mat (list len dim) dis)
    result))
