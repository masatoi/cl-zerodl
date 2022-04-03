(defpackage #:cl-zerodl/core/utils
  (:use #:cl
        #:mgl-mat
        #:cl-libsvm-format)
  (:nicknames :zerodl.utils)
  (:export #:define-class
           #:read-data
           #:average!))

(in-package #:cl-zerodl/core/utils)

(defmacro define-class (class-name superclass-list &body body)
  "Simplified definition of classes which similar to definition of structure.
 [Example]
  (define-class agent (superclass1 superclass2)
    currency
    position-list
    (position-upper-bound :initform 1 :type single-float)
    log
    money-management-rule)
=> #<STANDARD-CLASS AGENT>"
  (alexandria:with-gensyms (class initargs)
    `(prog1
         (defclass ,class-name (,@superclass-list)
           ,(mapcar (lambda (slot)
                      (let* ((slot-symbol (if (listp slot) (car slot) slot))
                             (slot-name (symbol-name slot-symbol))
                             (slot-initval (if (listp slot)
                                               (getf (cdr slot) :initform)
                                               nil))
                             (slot-type (if (listp slot)
                                            (getf (cdr slot) :type)
                                            t)))
                        (list slot-symbol
                              :accessor (intern slot-name)
                              :initarg (intern slot-name :keyword)
                              :initform slot-initval
                              :type slot-type)))
             body))

       (defmethod initialize-instance :before ((,class ,class-name)
                                               &rest ,initargs
                                               &key ,@(mapcar (lambda (slot)
                                                                (etypecase slot
                                                                  (list (if (getf (cdr slot) :initform)
                                                                            (list (car slot)
                                                                                  (getf (cdr slot) :initform))
                                                                            (car slot)))
                                                                  (symbol slot)))
                                                              body)
                                               &allow-other-keys)
         (declare (ignorable ,initargs
                             ,@(mapcar (lambda (slot)
                                         (etypecase slot
                                           (list (car slot))
                                           (symbol slot)))
                                       body)))
         ,@(remove nil
                   (mapcar (lambda (slot)
                             (when (and (listp slot) (getf (cdr slot) :type))
                               `(check-type ,(car slot) ,(getf (cdr slot) :type))))
                           body))))))

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

;;; Calculation utilities

(defun average! (a batch-size-tmp &key (axis 0))
  (sum! a batch-size-tmp :axis axis)
  (scal! (/ 1.0 (mat-dimension a axis)) batch-size-tmp))
