#ifndef _REG_MATHS_CPP
#define _REG_MATHS_CPP

#define USE_EIGEN

#include "_reg_maths.h"
// Eigen headers are in there because of nvcc preprocessing step
#include "Eigen/Core"
#include "Eigen/SVD"
#include "Eigen/unsupported/MatrixFunctions"

#define mat(i,j,dim) mat[i*dim+j]


/* *************************************************************** */
/* *************************************************************** */
void reg_logarithm_tensor(mat33 *in_tensor)
{
   int sm, sn;
   Eigen::Matrix3d tensor, sing;

   // Convert to Eigen format
   for(sm=0; sm<3; sm++)
      for(sn=0; sn<3; sn++)
         tensor(sm,sn)=static_cast<double>(in_tensor->m[sm][sn]);

   // Decompose the input tensor
   Eigen::JacobiSVD<Eigen::Matrix3d> svd(tensor,Eigen::ComputeThinV|Eigen::ComputeThinU);

   // Set a matrix containing the eigen values
   sing.setZero();
   sing(0,0)=svd.singularValues()(0);
   sing(1,1)=svd.singularValues()(1);
   sing(2,2)=svd.singularValues()(2);

   if(sing(0,0)<=0)
      sing(0,0)=std::numeric_limits<double>::epsilon();
   if(sing(1,1)<=0)
      sing(1,1)=std::numeric_limits<double>::epsilon();
   if(sing(2,2)<=0)
      sing(2,2)=std::numeric_limits<double>::epsilon();

   // Compute Rt log(E) R
   tensor = svd.matrixU() * sing.log() * svd.matrixU().transpose();

   // Convert the result to mat33 format
   for(sm=0; sm<3; sm++)
      for(sn=0; sn<3; sn++)
         in_tensor->m[sm][sn]=static_cast<float>(tensor(sm,sn));
}
/* *************************************************************** */
void reg_exponentiate_logged_tensor(mat33 *in_tensor)
{
   int sm, sn;
   Eigen::Matrix3d tensor;

   // Convert to Eigen format
   for(sm=0; sm<3; sm++)
      for(sn=0; sn<3; sn++)
         tensor(sm,sn)=static_cast<double>(in_tensor->m[sm][sn]);

   // Compute Rt exp(E) R
   tensor = tensor.exp();

   // Convert the result to mat33 format
   for(sm=0; sm<3; sm++)
      for(sn=0; sn<3; sn++)
         in_tensor->m[sm][sn]=static_cast<float>(tensor(sm,sn));
}
/* *************************************************************** */
/* *************************************************************** */
/** @brief SVD
  * @param in input matrix to decompose - in place
  * @param size_m row
  * @param size_n colomn
  * @param w diagonal term
  * @param v rotation part
  */
template <class T>
void svd(T ** in, size_t size_m, size_t size_n, T * w, T ** v)
{
   if(size_m==0 || size_n==0)
   {
      reg_print_fct_error("svd");
      reg_print_msg_error("The specified matrix is empty");
      reg_exit(1);
   }

#ifdef _WIN32
   long sm, sn, sn2;
   long size__m=(long)size_m,size__n=(long)size_n;
#else
   size_t sm, sn, sn2;
   size_t size__m=size_m,size__n=size_n;
#endif
   Eigen::MatrixXd m(size_m,size_n);

#if defined (_OPENMP)
   #pragma omp parallel for default(none) \
   shared(in,m, size__m, size__n) \
   private(sm, sn)
#endif
   for(sm=0; sm<size__m; sm++)
   {
      for(sn=0; sn<size__n; sn++)
      {
         m(sm,sn)=static_cast<double>(in[sm][sn]);
      }
   }

   Eigen::JacobiSVD<Eigen::MatrixXd> svd(m,Eigen::ComputeThinV|Eigen::ComputeThinU);

#if defined (_OPENMP)
   #pragma omp parallel for default(none) \
   shared(in,svd,v,w, size__n,size__m) \
   private(sn2, sn, sm)
#endif
   for(sn=0; sn<size__n; sn++)
   {
      w[sn]=svd.singularValues()(sn);
      for(sn2=0; sn2<size__n; sn2++)
      {
         v[sn2][sn]=static_cast<T>(svd.matrixV()(sn2,sn));
      }
      for(sm=0; sm<size__m; sm++)
      {
         in[sm][sn]=static_cast<T>(svd.matrixU()(sm,sn));
      }
   }
}
template void svd<float>(float ** in, size_t m, size_t n, float * w, float ** v);
template void svd<double>(double ** in, size_t m, size_t n, double * w, double ** v);
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_LUdecomposition(T *mat,
                         size_t dim,
                         size_t *index)
{
   T *vv=(T *)malloc(dim*sizeof(T));
   size_t i,j,k,imax=0;

   for(i=0; i<dim; ++i)
   {
      T big=0.f;
      T temp;
      for(j=0; j<dim; ++j)
         if( (temp=fabs(mat(i,j,dim)))>big)
            big=temp;
      if(big==0.f)
      {
         fprintf(stderr, "[NiftyReg] ERROR Singular matrix in the LU decomposition\n");
         reg_exit(1);
      }
      vv[i]=1.0/big;
   }
   for(j=0; j<dim; ++j)
   {
      for(i=0; i<j; ++i)
      {
         T sum=mat(i,j,dim);
         for(k=0; k<i; k++) sum -= mat(i,k,dim)*mat(k,j,dim);
         mat(i,j,dim)=sum;
      }
      T big=0.f;
      T dum;
      for(i=j; i<dim; ++i)
      {
         T sum=mat(i,j,dim);
         for(k=0; k<j; ++k ) sum -= mat(i,k,dim)*mat(k,j,dim);
         mat(i,j,dim)=sum;
         if( (dum=vv[i]*fabs(sum)) >= big )
         {
            big=dum;
            imax=i;
         }
      }
      if(j != imax)
      {
         for(k=0; k<dim; ++k)
         {
            dum=mat(imax,k,dim);
            mat(imax,k,dim)=mat(j,k,dim);
            mat(j,k,dim)=dum;
         }
         vv[imax]=vv[j];
      }
      index[j]=imax;
      if(mat(j,j,dim)==0) mat(j,j,dim)=1.0e-20;
      if(j!=dim-1)
      {
         dum=1.0/mat(j,j,dim);
         for(i=j+1; i<dim; ++i) mat(i,j,dim) *= dum;
      }
   }
   free(vv);
   return;
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_matrixInvertMultiply(T *mat,
                              size_t dim,
                              size_t *index,
                              T *vec)
{
   // Perform the LU decomposition if necessary
   if(index==NULL)
      reg_LUdecomposition(mat, dim, index);

   int ii=0;
   for(int i=0; i<(int)dim; ++i)
   {
      int ip=index[i];
      T sum = vec[ip];
      vec[ip]=vec[i];
      if(ii!=0)
      {
         for(int j=ii-1; j<i; ++j)
            sum -= mat(i,j,dim)*vec[j];
      }
      else if(sum!=0)
         ii=i+1;
      vec[i]=sum;
   }
   for(int i=(int)dim-1; i>-1; --i)
   {
      T sum=vec[i];
      for(int j=i+1; j<(int)dim; ++j)
         sum -= mat(i,j,dim)*vec[j];
      vec[i]=sum/mat(i,i,dim);
   }
}
template void reg_matrixInvertMultiply<float>(float *, size_t, size_t *, float *);
template void reg_matrixInvertMultiply<double>(double *, size_t, size_t *, double *);
/* *************************************************************** */
/* *************************************************************** */
extern "C++" template <class T>
void reg_matrixMultiply(T *mat1,
                        T *mat2,
                        int *dim1,
                        int *dim2,
                        T * &res)
{
   // First check that the dimension are appropriate
   if(dim1[1]!=dim2[0])
   {
      fprintf(stderr, "Matrices can not be multiplied due to their size: [%i %i] [%i %i]\n",
              dim1[0],dim1[1],dim2[0],dim2[1]);
      reg_exit(1);
   }
   int resDim[2]= {dim1[0],dim2[1]};
   // Allocate the result matrix
   if(res!=NULL)
      free(res);
   res=(T *)calloc(resDim[0]*resDim[1],sizeof(T));
   // Multiply both matrices
   for(int j=0; j<resDim[1]; ++j)
   {
      for(int i=0; i<resDim[0]; ++i)
      {
         double sum=0.0;
         for(int k=0; k<dim1[1]; ++k)
         {
            sum += mat1[k*dim1[0]+i] * mat2[j*dim2[0]+k];
         }
         res[j*resDim[0]+i]=sum;
      } // i
   } // j
}
template void reg_matrixMultiply<float>(float * ,float * ,int *, int * , float * &);
template void reg_matrixMultiply<double>(double * ,double * ,int *, int * , double * &);
/* *************************************************************** */
/* *************************************************************** */
// Heap sort
void reg_heapSort(float *array_tmp, int *index_tmp, int blockNum)
{
   float *array = &array_tmp[-1];
   int *index = &index_tmp[-1];
   int l=(blockNum >> 1)+1;
   int ir=blockNum;
   float val;
   int iVal;
   for(;;)
   {
      if(l>1)
      {
         val=array[--l];
         iVal=index[l];
      }
      else
      {
         val=array[ir];
         iVal=index[ir];
         array[ir]=array[1];
         index[ir]=index[1];
         if(--ir == 1)
         {
            array[1]=val;
            index[1]=iVal;
            break;
         }
      }
      int i=l;
      int j=l+l;
      while(j<=ir)
      {
         if(j<ir && array[j]<array[j+1]) j++;
         if(val<array[j])
         {
            array[i]=array[j];
            index[i]=index[j];
            i=j;
            j<<=1;
         }
         else break;
      }
      array[i]=val;
      index[i]=iVal;
   }
}
/* *************************************************************** */
// Heap sort
template <class DTYPE>
void reg_heapSort(DTYPE *array_tmp, int blockNum)
{
   DTYPE *array = &array_tmp[-1];
   int l=(blockNum >> 1)+1;
   int ir=blockNum;
   DTYPE val;
   for(;;)
   {
      if(l>1)
      {
         val=array[--l];
      }
      else
      {
         val=array[ir];
         array[ir]=array[1];
         if(--ir == 1)
         {
            array[1]=val;
            break;
         }
      }
      int i=l;
      int j=l+l;
      while(j<=ir)
      {
         if(j<ir && array[j]<array[j+1]) j++;
         if(val<array[j])
         {
            array[i]=array[j];
            i=j;
            j<<=1;
         }
         else break;
      }
      array[i]=val;
   }
}
template void reg_heapSort<float>(float *array_tmp, int blockNum);
template void reg_heapSort<double>(double *array_tmp, int blockNum);
/* *************************************************************** */
/* *************************************************************** */
bool operator==(mat44 A,mat44 B)
{
   for(unsigned i=0; i<4; ++i)
   {
      for(unsigned j=0; j<4; ++j)
      {
         if(A.m[i][j]!=B.m[i][j])
            return false;
      }
   }
   return true;
}
/* *************************************************************** */
bool operator!=(mat44 A,mat44 B)
{
   for(unsigned i=0; i<4; ++i)
   {
      for(unsigned j=0; j<4; ++j)
      {
         if(A.m[i][j]!=B.m[i][j])
            return true;
      }
   }
   return false;
}
/* *************************************************************** */
/* *************************************************************** */
float reg_mat44_det(mat44 const* A)
{
   double D =
      (double)A->m[0][0]*A->m[1][1]*A->m[2][2]*A->m[3][3]
      - A->m[0][0]*A->m[1][1]*A->m[3][2]*A->m[2][3]
      - A->m[0][0]*A->m[2][1]*A->m[1][2]*A->m[3][3]
      + A->m[0][0]*A->m[2][1]*A->m[3][2]*A->m[1][3]
      + A->m[0][0]*A->m[3][1]*A->m[1][2]*A->m[2][3]
      - A->m[0][0]*A->m[3][1]*A->m[2][2]*A->m[1][3]
      - A->m[1][0]*A->m[0][1]*A->m[2][2]*A->m[3][3]
      + A->m[1][0]*A->m[0][1]*A->m[3][2]*A->m[2][3]
      + A->m[1][0]*A->m[2][1]*A->m[0][2]*A->m[3][3]
      - A->m[1][0]*A->m[2][1]*A->m[3][2]*A->m[0][3]
      - A->m[1][0]*A->m[3][1]*A->m[0][2]*A->m[2][3]
      + A->m[1][0]*A->m[3][1]*A->m[2][2]*A->m[0][3]
      + A->m[2][0]*A->m[0][1]*A->m[1][2]*A->m[3][3]
      - A->m[2][0]*A->m[0][1]*A->m[3][2]*A->m[1][3]
      - A->m[2][0]*A->m[1][1]*A->m[0][2]*A->m[3][3]
      + A->m[2][0]*A->m[1][1]*A->m[3][2]*A->m[0][3]
      + A->m[2][0]*A->m[3][1]*A->m[0][2]*A->m[1][3]
      - A->m[2][0]*A->m[3][1]*A->m[1][2]*A->m[0][3]
      - A->m[3][0]*A->m[0][1]*A->m[1][2]*A->m[2][3]
      + A->m[3][0]*A->m[0][1]*A->m[2][2]*A->m[1][3]
      + A->m[3][0]*A->m[1][1]*A->m[0][2]*A->m[2][3]
      - A->m[3][0]*A->m[1][1]*A->m[2][2]*A->m[0][3]
      - A->m[3][0]*A->m[2][1]*A->m[0][2]*A->m[1][3]
      + A->m[3][0]*A->m[2][1]*A->m[1][2]*A->m[0][3];
   return static_cast<float>(D);
}
/* *************************************************************** */
//Ported from VNL
mat44 reg_mat44_inv(mat44 const* A)
{
   mat44 R;
   float detA = reg_mat44_det(A);
   if(detA==0)
   {
      fprintf(stderr,"[NiftyReg ERROR] Cannot invert 4x4 matrix with zero determinant.\n");
      fprintf(stderr,"[NiftyReg ERROR] Returning matrix of zeros\n");
      memset(&R,0,sizeof(mat44));
      return R;
   }
   detA = 1.0f / detA;
   R.m[0][0] =  A->m[1][1]*A->m[2][2]*A->m[3][3] - A->m[1][1]*A->m[2][3]*A->m[3][2]
                - A->m[2][1]*A->m[1][2]*A->m[3][3] + A->m[2][1]*A->m[1][3]*A->m[3][2]
                + A->m[3][1]*A->m[1][2]*A->m[2][3] - A->m[3][1]*A->m[1][3]*A->m[2][2];
   R.m[0][1] = -A->m[0][1]*A->m[2][2]*A->m[3][3] + A->m[0][1]*A->m[2][3]*A->m[3][2]
               + A->m[2][1]*A->m[0][2]*A->m[3][3] - A->m[2][1]*A->m[0][3]*A->m[3][2]
               - A->m[3][1]*A->m[0][2]*A->m[2][3] + A->m[3][1]*A->m[0][3]*A->m[2][2];
   R.m[0][2] =  A->m[0][1]*A->m[1][2]*A->m[3][3] - A->m[0][1]*A->m[1][3]*A->m[3][2]
                - A->m[1][1]*A->m[0][2]*A->m[3][3] + A->m[1][1]*A->m[0][3]*A->m[3][2]
                + A->m[3][1]*A->m[0][2]*A->m[1][3] - A->m[3][1]*A->m[0][3]*A->m[1][2];
   R.m[0][3] = -A->m[0][1]*A->m[1][2]*A->m[2][3] + A->m[0][1]*A->m[1][3]*A->m[2][2]
               + A->m[1][1]*A->m[0][2]*A->m[2][3] - A->m[1][1]*A->m[0][3]*A->m[2][2]
               - A->m[2][1]*A->m[0][2]*A->m[1][3] + A->m[2][1]*A->m[0][3]*A->m[1][2];
   R.m[1][0] = -A->m[1][0]*A->m[2][2]*A->m[3][3] + A->m[1][0]*A->m[2][3]*A->m[3][2]
               + A->m[2][0]*A->m[1][2]*A->m[3][3] - A->m[2][0]*A->m[1][3]*A->m[3][2]
               - A->m[3][0]*A->m[1][2]*A->m[2][3] + A->m[3][0]*A->m[1][3]*A->m[2][2];
   R.m[1][1] =  A->m[0][0]*A->m[2][2]*A->m[3][3] - A->m[0][0]*A->m[2][3]*A->m[3][2]
                - A->m[2][0]*A->m[0][2]*A->m[3][3] + A->m[2][0]*A->m[0][3]*A->m[3][2]
                + A->m[3][0]*A->m[0][2]*A->m[2][3] - A->m[3][0]*A->m[0][3]*A->m[2][2];
   R.m[1][2] = -A->m[0][0]*A->m[1][2]*A->m[3][3] + A->m[0][0]*A->m[1][3]*A->m[3][2]
               + A->m[1][0]*A->m[0][2]*A->m[3][3] - A->m[1][0]*A->m[0][3]*A->m[3][2]
               - A->m[3][0]*A->m[0][2]*A->m[1][3] + A->m[3][0]*A->m[0][3]*A->m[1][2];
   R.m[1][3] =  A->m[0][0]*A->m[1][2]*A->m[2][3] - A->m[0][0]*A->m[1][3]*A->m[2][2]
                - A->m[1][0]*A->m[0][2]*A->m[2][3] + A->m[1][0]*A->m[0][3]*A->m[2][2]
                + A->m[2][0]*A->m[0][2]*A->m[1][3] - A->m[2][0]*A->m[0][3]*A->m[1][2];
   R.m[2][0] =  A->m[1][0]*A->m[2][1]*A->m[3][3] - A->m[1][0]*A->m[2][3]*A->m[3][1]
                - A->m[2][0]*A->m[1][1]*A->m[3][3] + A->m[2][0]*A->m[1][3]*A->m[3][1]
                + A->m[3][0]*A->m[1][1]*A->m[2][3] - A->m[3][0]*A->m[1][3]*A->m[2][1];
   R.m[2][1] = -A->m[0][0]*A->m[2][1]*A->m[3][3] + A->m[0][0]*A->m[2][3]*A->m[3][1]
               + A->m[2][0]*A->m[0][1]*A->m[3][3] - A->m[2][0]*A->m[0][3]*A->m[3][1]
               - A->m[3][0]*A->m[0][1]*A->m[2][3] + A->m[3][0]*A->m[0][3]*A->m[2][1];
   R.m[2][2]=  A->m[0][0]*A->m[1][1]*A->m[3][3] - A->m[0][0]*A->m[1][3]*A->m[3][1]
               - A->m[1][0]*A->m[0][1]*A->m[3][3] + A->m[1][0]*A->m[0][3]*A->m[3][1]
               + A->m[3][0]*A->m[0][1]*A->m[1][3] - A->m[3][0]*A->m[0][3]*A->m[1][1];
   R.m[2][3]= -A->m[0][0]*A->m[1][1]*A->m[2][3] + A->m[0][0]*A->m[1][3]*A->m[2][1]
              + A->m[1][0]*A->m[0][1]*A->m[2][3] - A->m[1][0]*A->m[0][3]*A->m[2][1]
              - A->m[2][0]*A->m[0][1]*A->m[1][3] + A->m[2][0]*A->m[0][3]*A->m[1][1];
   R.m[3][0]= -A->m[1][0]*A->m[2][1]*A->m[3][2] + A->m[1][0]*A->m[2][2]*A->m[3][1]
              + A->m[2][0]*A->m[1][1]*A->m[3][2] - A->m[2][0]*A->m[1][2]*A->m[3][1]
              - A->m[3][0]*A->m[1][1]*A->m[2][2] + A->m[3][0]*A->m[1][2]*A->m[2][1];
   R.m[3][1]=  A->m[0][0]*A->m[2][1]*A->m[3][2] - A->m[0][0]*A->m[2][2]*A->m[3][1]
               - A->m[2][0]*A->m[0][1]*A->m[3][2] + A->m[2][0]*A->m[0][2]*A->m[3][1]
               + A->m[3][0]*A->m[0][1]*A->m[2][2] - A->m[3][0]*A->m[0][2]*A->m[2][1];
   R.m[3][2]= -A->m[0][0]*A->m[1][1]*A->m[3][2] + A->m[0][0]*A->m[1][2]*A->m[3][1]
              + A->m[1][0]*A->m[0][1]*A->m[3][2] - A->m[1][0]*A->m[0][2]*A->m[3][1]
              - A->m[3][0]*A->m[0][1]*A->m[1][2] + A->m[3][0]*A->m[0][2]*A->m[1][1];
   R.m[3][3]=  A->m[0][0]*A->m[1][1]*A->m[2][2] - A->m[0][0]*A->m[1][2]*A->m[2][1]
               - A->m[1][0]*A->m[0][1]*A->m[2][2] + A->m[1][0]*A->m[0][2]*A->m[2][1]
               + A->m[2][0]*A->m[0][1]*A->m[1][2] - A->m[2][0]*A->m[0][2]*A->m[1][1];
   return reg_mat44_mul(&R,detA);
}
/* *************************************************************** */
/* *************************************************************** */
mat33 reg_mat44_to_mat33(mat44 const* A)
{
   mat33 out;
   out.m[0][0]=A->m[0][0];
   out.m[0][1]=A->m[0][1];
   out.m[0][2]=A->m[0][2];
   out.m[1][0]=A->m[1][0];
   out.m[1][1]=A->m[1][1];
   out.m[1][2]=A->m[1][2];
   out.m[2][0]=A->m[2][0];
   out.m[2][1]=A->m[2][1];
   out.m[2][2]=A->m[2][2];
   return out;
}
/* *************************************************************** */
/* *************************************************************** */
mat44 reg_mat44_mul(mat44 const* A, mat44 const* B)
{
   mat44 R;
   for(int i=0; i<4; i++)
   {
      for(int j=0; j<4; j++)
      {
         R.m[i][j] = A->m[i][0]*B->m[0][j] +
                     A->m[i][1]*B->m[1][j] +
                     A->m[i][2]*B->m[2][j] +
                     A->m[i][3]*B->m[3][j];
      }
   }
   return R;
}
/* *************************************************************** */
mat44 operator*(mat44 A,mat44 B)
{
   return reg_mat44_mul(&A,&B);
}
/* *************************************************************** */
/* *************************************************************** */
mat44 reg_mat44_add(mat44 const* A, mat44 const* B)
{
   mat44 R;
   for(int i=0; i<4; i++)
   {
      for(int j=0; j<4; j++)
      {
         R.m[i][j] = A->m[i][j] + B->m[i][j];
      }
   }
   return R;
}
/* *************************************************************** */
/* *************************************************************** */
mat44 operator+(mat44 A,mat44 B)
{
   return reg_mat44_add(&A,&B);
}
/* *************************************************************** */
/* *************************************************************** */
mat44 reg_mat44_minus(mat44 const* A, mat44 const* B)
{
   mat44 R;
   for(int i=0; i<4; i++)
   {
      for(int j=0; j<4; j++)
      {
         R.m[i][j] = A->m[i][j]-B->m[i][j];
      }
   }
   return R;
}
/* *************************************************************** */
mat44 operator-(mat44 A,mat44 B)
{
   return reg_mat44_minus(&A,&B);
}
/* *************************************************************** */
/* *************************************************************** */
void reg_mat33_eye (mat33 *mat)
{
   mat->m[0][0]=1.f;
   mat->m[0][1]=mat->m[0][2]=0.f;
   mat->m[1][1]=1.f;
   mat->m[1][0]=mat->m[1][2]=0.f;
   mat->m[2][2]=1.f;
   mat->m[2][0]=mat->m[2][1]=0.f;
}
/* *************************************************************** */
void reg_mat44_eye (mat44 *mat)
{
   mat->m[0][0]=1.f;
   mat->m[0][1]=mat->m[0][2]=mat->m[0][3]=0.f;
   mat->m[1][1]=1.f;
   mat->m[1][0]=mat->m[1][2]=mat->m[1][3]=0.f;
   mat->m[2][2]=1.f;
   mat->m[2][0]=mat->m[2][1]=mat->m[2][3]=0.f;
   mat->m[3][3]=1.f;
   mat->m[3][0]=mat->m[3][1]=mat->m[3][2]=0.f;
}
/* *************************************************************** */
/* *************************************************************** */
float reg_mat44_norm_inf(mat44 const* mat)
{
   float maxval=0.0;
   float newval=0.0;
   for (int i=0; i < 4; i++)
   {
      for (int j=0; j < 4; j++)
      {
         newval = fabsf((float)mat->m[i][j]);
         maxval = (newval > maxval) ? newval : maxval;
      }
   }
   return maxval;
}
/* *************************************************************** */
/* *************************************************************** */
void reg_mat44_mul(mat44 const* mat,
                   float const* in,
                   float *out)
{
   double matD[4][4], inD[3]={in[0],in[1],in[2]};
   for(int i=0;i<4;++i)
      for(int j=0;j<4;++j)
         matD[i][j]=static_cast<double>(mat->m[i][j]);
   out[0]=static_cast<float>(matD[0][0]*inD[0] +
         matD[0][1]*inD[1] +
         matD[0][2]*inD[2] +
         matD[0][3]);
   out[1]=static_cast<float>(matD[1][0]*inD[0] +
         matD[1][1]*inD[1] +
         matD[1][2]*inD[2] +
         matD[1][3]);
   out[2]=static_cast<float>(matD[2][0]*inD[0] +
         matD[2][1]*inD[1] +
         matD[2][2]*inD[2] +
         matD[2][3]);
   return;
}
/* *************************************************************** */
/* *************************************************************** */
void reg_mat44_mul(mat44 const* mat,
                   double const* in,
                   double *out)
{
   double matD[4][4];
   for(int i=0;i<4;++i)
      for(int j=0;j<4;++j)
         matD[i][j]=static_cast<double>(mat->m[i][j]);
   out[0]=matD[0][0]*in[0] +
          matD[0][1]*in[1] +
          matD[0][2]*in[2] +
          matD[0][3];
   out[1]=matD[1][0]*in[0] +
          matD[1][1]*in[1] +
          matD[1][2]*in[2] +
          matD[1][3];
   out[2]=matD[2][0]*in[0] +
          matD[2][1]*in[1] +
          matD[2][2]*in[2] +
          matD[2][3];
   return;
}
/* *************************************************************** */
/* *************************************************************** */
mat44 reg_mat44_mul(mat44 const* A, double scalar)
{
   mat44 out;
   out.m[0][0]=A->m[0][0]*scalar;
   out.m[0][1]=A->m[0][1]*scalar;
   out.m[0][2]=A->m[0][2]*scalar;
   out.m[0][3]=A->m[0][3]*scalar;
   out.m[1][0]=A->m[1][0]*scalar;
   out.m[1][1]=A->m[1][1]*scalar;
   out.m[1][2]=A->m[1][2]*scalar;
   out.m[1][3]=A->m[1][3]*scalar;
   out.m[2][0]=A->m[2][0]*scalar;
   out.m[2][1]=A->m[2][1]*scalar;
   out.m[2][2]=A->m[2][2]*scalar;
   out.m[2][3]=A->m[2][3]*scalar;
   out.m[3][0]=A->m[3][0]*scalar;
   out.m[3][1]=A->m[3][1]*scalar;
   out.m[3][2]=A->m[3][2]*scalar;
   out.m[3][3]=A->m[3][3]*scalar;
   return out;
}
/* *************************************************************** */
/* *************************************************************** */
mat44 reg_mat44_sqrt(mat44 const* mat)
{
   mat44 X;
   Eigen::Matrix4f m;
   for(size_t i=0; i<4; ++i)
   {
      for(size_t j=0; j<4; ++j)
      {
         m(i,j)=static_cast<float>(mat->m[i][j]);
      }
   }
   m=m.sqrt();
   for(size_t i=0; i<4; ++i)
      for(size_t j=0; j<4; ++j)
         X.m[i][j] = static_cast<float>(m(i,j));
   return X;
}
/* *************************************************************** */
/* *************************************************************** */
mat44 reg_mat44_expm(mat44 const* mat)
{
   mat44 X;
   Eigen::Matrix4d m;
   for(size_t i=0; i<4; ++i)
   {
      for(size_t j=0; j<4; ++j)
      {
         m(i,j)=static_cast<double>(mat->m[i][j]);
      }
   }
   m=m.exp();
   for(size_t i=0; i<4; ++i)
      for(size_t j=0; j<4; ++j)
         X.m[i][j] = static_cast<float>(m(i,j));

   return X;
}
/* *************************************************************** */
/* *************************************************************** */
mat44 reg_mat44_logm(mat44 const* mat)
{
   mat44 X;
   Eigen::Matrix4d m;
   for(size_t i=0; i<4; ++i)
   {
      for(size_t j=0; j<4; ++j)
      {
         m(i,j)=static_cast<double>(mat->m[i][j]);
      }
   }
   m=m.log();
   for(size_t i=0; i<4; ++i)
      for(size_t j=0; j<4; ++j)
         X.m[i][j] = static_cast<float>(m(i,j));
   return X;
}
/* *************************************************************** */
/* *************************************************************** */
mat44 reg_mat44_avg2(mat44 const* A, mat44 const* B)
{
   mat44 out;
   mat44 logA=reg_mat44_logm(A);
   mat44 logB=reg_mat44_logm(B);
   for(int i=0;i<4;++i){
      logA.m[3][i]=0.f;
      logB.m[3][i]=0.f;
   }
   logA = reg_mat44_add(&logA,&logB);
   out = reg_mat44_mul(&logA,0.5);
   return reg_mat44_expm(&out);

}
/* *************************************************************** */
/* *************************************************************** */
void reg_mat44_disp(mat44 *mat, char * title)
{
   printf("%s:\n%g\t%g\t%g\t%g\n%g\t%g\t%g\t%g\n%g\t%g\t%g\t%g\n%g\t%g\t%g\t%g\n", title,
          mat->m[0][0], mat->m[0][1], mat->m[0][2], mat->m[0][3],
          mat->m[1][0], mat->m[1][1], mat->m[1][2], mat->m[1][3],
          mat->m[2][0], mat->m[2][1], mat->m[2][2], mat->m[2][3],
          mat->m[3][0], mat->m[3][1], mat->m[3][2], mat->m[3][3]);
}
/* *************************************************************** */
/* *************************************************************** */
void reg_mat33_disp(mat33 *mat, char * title)
{
   printf("%s:\n%g\t%g\t%g\n%g\t%g\t%g\n%g\t%g\t%g\n", title,
          mat->m[0][0], mat->m[0][1], mat->m[0][2],
          mat->m[1][0], mat->m[1][1], mat->m[1][2],
          mat->m[2][0], mat->m[2][1], mat->m[2][2]);
}
/* *************************************************************** */
/* *************************************************************** */
// Calculate pythagorean distance
template <class T>
T pythag(T a, T b)
{
   T absa, absb;
   absa = fabs(a);
   absb = fabs(b);

   if (absa > absb) return (T)(absa * sqrt(1.0f+SQR(absb/absa)));
   else return (absb == 0.0f ? 0.0f : (T)(absb * sqrt(1.0f+SQR(absa/absb))));
}
/* *************************************************************** */
/* *************************************************************** */
#endif // _REG_MATHS_CPP
