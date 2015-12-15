/**
 * @file _reg_maths.h
 * @brief Library that contains small math routines
 * @author Marc Modat
 * @date 25/03/2009
 *
 *  Created by Marc Modat on 25/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */
#ifndef _REG_MATHS_H
#define _REG_MATHS_H

#include <stdio.h>
#include <math.h>
#include <iostream>
#include "nifti1_io.h"

#if defined (_OPENMP)
#include <omp.h>
#endif

typedef enum
{
   DEF_FIELD,
   DISP_FIELD,
   SPLINE_GRID,
   DEF_VEL_FIELD,
   DISP_VEL_FIELD,
   SPLINE_VEL_GRID
} NREG_TRANS_TYPE;

/* *************************************************************** */
#define reg_pow2(a) ((a)*(a))
#define reg_ceil(a) (ceil(a))
#define reg_round(a) ((a)>0.0 ?(int)((a)+0.5):(int)((a)-0.5))
#ifdef _WIN32
#define reg_floor(a) ((a)>0?(int)(a):(int)((a)-1))
#else
#define reg_floor(a) ((a)>=0?(int)(a):floor(a))
#endif
#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))
#define FMAX(a,b) (a > b ? a : b)
#define IMIN(a,b) (a < b ? a : b)
#define SQR(a) (a==0.0 ? 0.0 : a*a)
/* *************************************************************** */
#define reg_exit(val){ \
    fprintf(stderr,"[NiftyReg] Exit here. File: %s:%i\n",__FILE__, __LINE__); \
    exit(val); \
}
#define reg_print_fct_debug(name){printf("[NiftyReg DEBUG] Function: %s called\n",name);}
#define reg_print_msg_debug(name){printf("[NiftyReg DEBUG] Message: %s\n",name);}
#define reg_print_fct_warn(name){printf("[NiftyReg WARNING] Function: %s\n",name);}
#define reg_print_msg_warn(name){printf("[NiftyReg WARNING] Message: %s\n",name);}
#define reg_print_fct_error(name){fprintf(stderr,"[NiftyReg ERROR] Function: %s\n",name);}
#define reg_print_msg_error(name){fprintf(stderr,"[NiftyReg ERROR] Message: %s\n",name);}
/* *************************************************************** */
#if defined(_WIN32) && !defined(__CYGWIN__)
#include <limits>
#include <float.h>
#include <time.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef isnan
#define isnan(_X) _isnan(_X)
#endif
#ifndef strtof
#define strtof(_s, _t) (float) strtod(_s, _t)
#endif
template<class PrecisionType> inline int round(PrecisionType x)
{
   return int(x > 0.0 ? (x + 0.5) : (x - 0.5));
}
#if _MSC_VER < 1800 //test if visual studio version older than 2013
template<typename T>inline bool isinf(T value)
{
   return std::numeric_limits<T>::has_infinity && value == std::numeric_limits<T>::infinity();
}
#endif
inline int fabs(int _x)
{
   return (int)fabs((float)(_x));
}
#endif // If on windows...

/* *************************************************************** */
/* Functions calling the Eigen library                             */
/* See http://eigen.tuxfamily.org/index.php?title=Main_Page        */
/* *************************************************************** */
extern "C++"
void reg_logarithm_tensor(mat33 *in_tensor);
/* *************************************************************** */
extern "C++"
void reg_exponentiate_logged_tensor(mat33 *in_tensor);
/* *************************************************************** */
/* *************************************************************** */
extern "C++" template <class T>
void svd(T ** in, size_t m, size_t n, T * w, T ** v);
/* *************************************************************** */
/* *************************************************************** */
extern "C++" template <class T>
void reg_LUdecomposition(T *inputMatrix,
                         size_t dim,
                         size_t *index);
/* *************************************************************** */
extern "C++" template <class T>
void reg_matrixMultiply(T *mat1,
                        T *mat2,
                        int *dim1,
                        int *dim2,
                        T * &res);
/* *************************************************************** */
extern "C++" template <class T>
void reg_matrixInvertMultiply(T *mat,
                              size_t dim,
                              size_t *index,
                              T *vec);
/* *************************************************************** */
extern "C++"
void reg_heapSort(float *array_tmp, int *index_tmp, int blockNum);
/* *************************************************************** */
extern "C++" template <class T>
void reg_heapSort(T *array_tmp,int blockNum);
/* *************************************************************** */
/* *************************************************************** */
bool operator==(mat44 A,mat44 B);
/* *************************************************************** */
bool operator!=(mat44 A,mat44 B);
/* *************************************************************** */
/** @brief here
 */
mat33 reg_mat44_to_mat33(mat44 const* A);
/* *************************************************************** */
/** @brief Multipy two 4-by-4 matrices
 */
mat44 reg_mat44_mul(mat44 const* A,
                    mat44 const* B);
mat44 operator*(mat44 A,
                mat44 B);
/* *************************************************************** */
/** @brief Multipy a vector with a 4-by-4 matrix
 */
void reg_mat44_mul(mat44 const* mat,
                   float const* in,
                   float *out);
void reg_mat44_mul(mat44 const* mat,
                   double const* in,
                   double *out);
/* *************************************************************** */
/** @brief Multipy a 4-by-4 matrix with a scalar
 */
mat44 reg_mat44_mul(mat44 const* mat,
                    double scalar);
/* *************************************************************** */
/** @brief Add two 4-by-4 matrices
 */
mat44 reg_mat44_add(mat44 const* A, mat44 const* B);
mat44 operator+(mat44 A,mat44 B);
/* *************************************************************** */
/** @brief Substract two 4-by-4 matrices
 */
mat44 reg_mat44_minus(mat44 const* A, mat44 const* B);
mat44 operator-(mat44 A,mat44 B);
/* *************************************************************** */
/** @brief Set up a 3-by-3 matrix with an identity
 */
void reg_mat33_eye (mat33 *mat);
/* *************************************************************** */
/** @brief Set up a 4-by-4 matrix with an identity
 */
void reg_mat44_eye(mat44 *mat);
/* *************************************************************** */
/** @brief Compute the determinant of a 4-by-4 matrix
 */
float reg_mat44_det(mat44 const* A);
/* *************************************************************** */
/** @brief Compute the inverse of a 4-by-4 matrix
 */
mat44 reg_mat44_inv(mat44 const* A);
/* *************************************************************** */
float reg_mat44_norm_inf(mat44 const* mat);
/* *************************************************************** */
/** @brief Compute the square root of a 4-by-4 matrix
 */
mat44 reg_mat44_sqrt(mat44 const* mat);
/* *************************************************************** */
/** @brief Compute the exp of a 4-by-4 matrix
 */
mat44 reg_mat44_expm(mat44 const* mat);
/* *************************************************************** */
/** @brief Compute the log of a 4-by-4 matrix
 */
mat44 reg_mat44_logm(mat44 const* mat);
/* *************************************************************** */
/** @brief Compute the average of two matrices using a log-euclidean
 * framework
 */
mat44 reg_mat44_avg2(mat44 const* A, mat44 const* b);
/* *************************************************************** */
/** @brief Display a mat44 matrix
 */
void reg_mat44_disp(mat44 *mat,
                    char * title);
/* *************************************************************** */
/** @brief Display a mat33 matrix
 */
void reg_mat33_disp(mat33 *mat,
                    char * title);
/* *************************************************************** */
#endif // _REG_MATHS_H
