#include "_reg_maths.h"

#define EPS 0.00001

int main()
{
   // Create the input matrices
   size_t n=5;
   size_t m=10;
   float **matrix=NULL;
   float *w=NULL;
   float **v=NULL;

   // Create the test matrices
   float *res_w=NULL;
   float **res_v=NULL;

   // Allocate the matrices
   matrix = (float **)malloc(m*sizeof(float*));
   w = (float *)malloc(m*sizeof(float));
   res_w = (float *)malloc(m*sizeof(float));
   v = (float **)malloc(n*sizeof(float*));
   res_v = (float **)malloc(n*sizeof(float*));
   for(size_t i=0;i<m;++i){
      matrix[i] = (float *)malloc(n*sizeof(float));
   }
   for(size_t i=0;i<n;++i){
      v[i] = (float *)malloc(n*sizeof(float));
      res_v[i] = (float *)malloc(n*sizeof(float));
   }

   // Set the matrices
   matrix[0][0]=0.623716f;matrix[0][1]=0.193245f;matrix[0][2]=0.510153f;matrix[0][3]=0.485652f;matrix[0][4]=0.124774f;
   matrix[1][0]=0.236445f;matrix[1][1]=0.895892f;matrix[1][2]=0.906364f;matrix[1][3]=0.894448f;matrix[1][4]=0.730585f;
   matrix[2][0]=0.177124f;matrix[2][1]=0.0990896f;matrix[2][2]=0.628924f;matrix[2][3]=0.137547f;matrix[2][4]=0.646477f;
   matrix[3][0]=0.829643f;matrix[3][1]=0.0441656f;matrix[3][2]=0.101534f;matrix[3][3]=0.390005f;matrix[3][4]=0.833152f;
   matrix[4][0]=0.766922f;matrix[4][1]=0.557295f;matrix[4][2]=0.390855f;matrix[4][3]=0.927356f;matrix[4][4]=0.398282f;
   matrix[5][0]=0.934478f;matrix[5][1]=0.772495f;matrix[5][2]=0.0546166f;matrix[5][3]=0.917494f;matrix[5][4]=0.749822f;
   matrix[6][0]=0.107889f;matrix[6][1]=0.31194f;matrix[6][2]=0.501283f;matrix[6][3]=0.713574f;matrix[6][4]=0.835221f;
   matrix[7][0]=0.182228f;matrix[7][1]=0.178982f;matrix[7][2]=0.431721f;matrix[7][3]=0.618337f;matrix[7][4]=0.32246f;
   matrix[8][0]=0.0990953f;matrix[8][1]=0.338956f;matrix[8][2]=0.99756f;matrix[8][3]=0.343288f;matrix[8][4]=0.552262f;
   matrix[9][0]=0.489764f;matrix[9][1]=0.210146f;matrix[9][2]=0.811603f;matrix[9][3]=0.936027f;matrix[9][4]=0.979129f;
   res_v[0][0]=-0.368836f;res_v[0][1]=-0.66061f;res_v[0][2]=0.209725f;res_v[0][3]=0.560415f;res_v[0][4]=0.263637f;
   res_v[1][0]=-0.326943f;res_v[1][1]=-0.0714797f;res_v[1][2]=-0.662116f;res_v[1][3]=-0.327027f;res_v[1][4]=0.585366f;
   res_v[2][0]=-0.430015f;res_v[2][1]=0.72255f;res_v[2][2]=-0.0312705f;res_v[2][3]=0.529405f;res_v[2][4]=0.108449f;
   res_v[3][0]=-0.551549f;res_v[3][1]=-0.166557f;res_v[3][2]=-0.325048f;res_v[3][3]=-0.0873002f;res_v[3][4]=-0.744832f;
   res_v[4][0]=-0.517638f;res_v[4][1]=0.0930836f;res_v[4][2]=0.64108f;res_v[4][3]=-0.539535f;res_v[4][4]=0.145964f;
   res_w[0]=3.868f;res_w[1]=1.28005f;res_w[2]=0.862982f;res_w[3]=0.625536f;res_w[4]=0.456751f;

   svd<float>(matrix, m, n, w, v);

   for(size_t i=0;i<n;++i){
      float difference=fabsf(res_w[i]-w[i]);
      if(difference>EPS){
         fprintf(stderr, "reg_test_svd - Error in the SVD computation %.8g (>%g)\n", difference, EPS);
         return EXIT_FAILURE;
      }
   }
   for(size_t i=0;i<n;++i){
      for(size_t j=0;j<n;++j){
         float difference=fabsf(res_v[i][j])+-fabsf(v[i][j]);
         if(difference>EPS){
            fprintf(stderr, "reg_test_svd - Error in the SVD computation %.8g (>%g)\n", difference, EPS);
            return EXIT_FAILURE;
         }
      }
   }
   // Free the allocated variables
   for(size_t i=0;i<m;++i){
      free(matrix[i]);
   }
   for(size_t i=0;i<n;++i){
      free(v[i]);
      free(res_v[i]);
   }
   free(matrix);
   free(v);
   free(res_v);
   free(w);
   free(res_w);

   return EXIT_SUCCESS;
}

