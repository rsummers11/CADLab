#include "_reg_maths.h"

#define EPS 0.000001

int check_matrix_difference(mat44 matrix1, mat44 matrix2, char *name)
{
   for(int i=0;i<4;++i){
      for(int j=0;j<4;++j){
         float difference = fabsf(matrix1.m[i][j]-matrix2.m[i][j]);
         if(difference>EPS){
            fprintf(stderr, "reg_test_matrix_operation - %s failed %g>%g\n",
                    name, difference,EPS);
            return 1;
         }
      }
   }
   return 0;
}

int main()
{
   mat44 matrix1,matrix2;
   matrix1.m[0][0]=1.8147f;matrix1.m[0][1]=0.6324f;matrix1.m[0][2]=0.9575f;matrix1.m[0][3]=0.9572f;
   matrix1.m[1][0]=0.9058f;matrix1.m[1][1]=1.0975f;matrix1.m[1][2]=0.9649f;matrix1.m[1][3]=0.4854f;
   matrix1.m[2][0]=0.127f;matrix1.m[2][1]=0.2785f;matrix1.m[2][2]=1.1576f;matrix1.m[2][3]=0.8003f;
   matrix1.m[3][0]=0.f;matrix1.m[3][1]=0.f;matrix1.m[3][2]=0.f;matrix1.m[3][3]=1.f;
   matrix2.m[0][0]=0.2769f;matrix2.m[0][1]=0.6948f;matrix2.m[0][2]=0.4387f;matrix2.m[0][3]=0.1869f;
   matrix2.m[1][0]=0.0462f;matrix2.m[1][1]=0.3171f;matrix2.m[1][2]=0.3816f;matrix2.m[1][3]=0.4898f;
   matrix2.m[2][0]=0.0971f;matrix2.m[2][1]=0.9502f;matrix2.m[2][2]=0.7655f;matrix2.m[2][3]=0.4456f;
   matrix2.m[3][0]=0.f;matrix2.m[3][1]=0.f;matrix2.m[3][2]=0.f;matrix2.m[3][3]=1.f;

   mat44 multResultMatrix;
   multResultMatrix.m[0][0]=0.62468056f;multResultMatrix.m[0][1]=2.3712041f;multResultMatrix.m[0][2]=1.77039898f;multResultMatrix.m[0][3]=2.03277895f;
   multResultMatrix.m[1][0]=0.39521231f;multResultMatrix.m[1][1]=1.89421507f;multResultMatrix.m[1][2]=1.55481141f;multResultMatrix.m[1][3]=1.62220896f;
   multResultMatrix.m[2][0]=0.16043596f;multResultMatrix.m[2][1]=1.27650347f;multResultMatrix.m[2][2]=1.0481333f;multResultMatrix.m[2][3]=1.47627216f;
   multResultMatrix.m[3][0]=0.f;multResultMatrix.m[3][1]=0.f;multResultMatrix.m[3][2]=0.f;multResultMatrix.m[3][3]=1.f;
   if(check_matrix_difference(multResultMatrix,matrix1*matrix2,(char *)"matrix multiplication")) return EXIT_FAILURE;

   mat44 addiResultMatrix;
   addiResultMatrix.m[0][0]=2.0916f;addiResultMatrix.m[0][1]=1.3272f;addiResultMatrix.m[0][2]=1.3962f;addiResultMatrix.m[0][3]=1.1441f;
   addiResultMatrix.m[1][0]=0.952f;addiResultMatrix.m[1][1]=1.4146f;addiResultMatrix.m[1][2]=1.3465f;addiResultMatrix.m[1][3]=0.9752f;
   addiResultMatrix.m[2][0]=0.2241f;addiResultMatrix.m[2][1]=1.2287f;addiResultMatrix.m[2][2]=1.9231f;addiResultMatrix.m[2][3]=1.2459f;
   addiResultMatrix.m[3][0]=0.f;addiResultMatrix.m[3][1]=0.f;addiResultMatrix.m[3][2]=0.f;addiResultMatrix.m[3][3]=2.f;
   if(check_matrix_difference(addiResultMatrix,matrix1+matrix2,(char *)"matrix addition")) return EXIT_FAILURE;

   mat44 subtResultMatrix;
   subtResultMatrix.m[0][0]=1.5378f;subtResultMatrix.m[0][1]=-0.0624f;subtResultMatrix.m[0][2]=0.5188f;subtResultMatrix.m[0][3]=0.7703f;
   subtResultMatrix.m[1][0]=0.8596f;subtResultMatrix.m[1][1]=0.7804f;subtResultMatrix.m[1][2]=0.5833f;subtResultMatrix.m[1][3]=-0.0044f;
   subtResultMatrix.m[2][0]=0.0299f;subtResultMatrix.m[2][1]=-0.6717f;subtResultMatrix.m[2][2]=0.3921f;subtResultMatrix.m[2][3]=0.3547f;
   subtResultMatrix.m[3][0]=0.f;subtResultMatrix.m[3][1]=0.f;subtResultMatrix.m[3][2]=0.f;subtResultMatrix.m[3][3]=0.f;
   if(check_matrix_difference(subtResultMatrix,matrix1-matrix2,(char *)"matrix subtraction")) return EXIT_FAILURE;

   mat44 expmResultMatrix;
   expmResultMatrix.m[0][0]=8.23043768f;expmResultMatrix.m[0][1]=3.82217343f;expmResultMatrix.m[0][2]=6.35262245f;expmResultMatrix.m[0][3]=7.17885858f;
   expmResultMatrix.m[1][0]=4.91885117f;expmResultMatrix.m[1][1]=4.85880412f;expmResultMatrix.m[1][2]=5.45986756f;expmResultMatrix.m[1][3]=5.19066659f;
   expmResultMatrix.m[2][0]=1.20963578f;expmResultMatrix.m[2][1]=1.21165914f;expmResultMatrix.m[2][2]=4.11648121f;expmResultMatrix.m[2][3]=3.22123099f;
   expmResultMatrix.m[3][0]=0.f;expmResultMatrix.m[3][1]=0.f;expmResultMatrix.m[3][2]=0.f;expmResultMatrix.m[3][3]=2.71828183f;
   if(check_matrix_difference(expmResultMatrix,reg_mat44_expm(&matrix1),(char *)"matrix exponentiation")) return EXIT_FAILURE;

   mat44 logmResultMatrix;
   logmResultMatrix.m[0][0]=0.457990707f;logmResultMatrix.m[0][1]=0.431897417f;logmResultMatrix.m[0][2]=0.555724538f;logmResultMatrix.m[0][3]=0.501045308f;
   logmResultMatrix.m[1][0]=0.715106834f;logmResultMatrix.m[1][1]=-0.199037106f;logmResultMatrix.m[1][2]=0.723552453f;logmResultMatrix.m[1][3]=-0.0430404022f;
   logmResultMatrix.m[2][0]=0.00998029758f;logmResultMatrix.m[2][1]=0.272080256f;logmResultMatrix.m[2][2]=0.0339663237f;logmResultMatrix.m[2][3]=0.74338924f;
   logmResultMatrix.m[3][0]=0.f;logmResultMatrix.m[3][1]=0.f;logmResultMatrix.m[3][2]=0.f;logmResultMatrix.m[3][3]=0.f;
   if(check_matrix_difference(logmResultMatrix,reg_mat44_logm(&matrix1),(char *)"matrix logarithm")) return EXIT_FAILURE;

   mat44 inveResultMatrix;
   inveResultMatrix.m[0][0]=0.74738107f;inveResultMatrix.m[0][1]=-0.347228365f;inveResultMatrix.m[0][2]=-0.328763585f;inveResultMatrix.m[0][3]=-0.283739015f;
   inveResultMatrix.m[1][0]=-0.69088061f;inveResultMatrix.m[1][1]=1.47656634f;inveResultMatrix.m[1][2]=-0.659312953f;inveResultMatrix.m[1][3]=0.472233776f;
   inveResultMatrix.m[2][0]=0.0842198117f;inveResultMatrix.m[2][1]=-0.317143852f;inveResultMatrix.m[2][2]=1.05854495f;inveResultMatrix.m[2][3]=-0.773827101f;
   inveResultMatrix.m[3][0]=0.f;inveResultMatrix.m[3][1]=0.f;inveResultMatrix.m[3][2]=0.f;inveResultMatrix.m[3][3]=1.f;
   if(check_matrix_difference(inveResultMatrix,nifti_mat44_inverse(matrix1),(char *)"matrix logarithm")) return EXIT_FAILURE;

   return EXIT_SUCCESS;
}

