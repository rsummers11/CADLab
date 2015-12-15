#include "_reg_maths.h"
#include "nifti1_io.h"
#define PI 3.14159265

int main ()
{
    //Set identity
    mat44 I;
    mat44 sqrtI;
    mat44 logI, expI;
    mat44 Ilog,Iexp;
    reg_mat44_eye(&I);
    reg_mat44_disp(&I,(char*)"Original");
    sqrtI=reg_mat44_sqrt(&I);
    reg_mat44_disp(&sqrtI,(char*)"Square root of Original");
    logI=reg_mat44_logm(&I);
    reg_mat44_disp(&logI,(char*)"Log of original");
    expI=reg_mat44_expm(&I);
    reg_mat44_disp(&expI,(char*)"Expoential of original");
    Ilog=reg_mat44_expm(&logI);
    reg_mat44_disp(&Ilog,(char*)"Log-Exp of Original");
    Iexp=reg_mat44_logm(&expI);
    reg_mat44_disp(&Iexp,(char*)"Exp-Log of Original");
    I.m[0][3]=5.0;
    reg_mat44_disp(&I,(char*)"Original");
    sqrtI=reg_mat44_sqrt(&I);
    reg_mat44_disp(&sqrtI,(char*)"Square root of Original");
    logI=reg_mat44_logm(&I);
    reg_mat44_disp(&logI,(char*)"Log of original");
    expI=reg_mat44_expm(&I);
    reg_mat44_disp(&expI,(char*)"Expoential of original");
    Ilog=reg_mat44_expm(&logI);
    reg_mat44_disp(&Ilog,(char*)"Log-Exp of Original");
    Iexp=reg_mat44_logm(&expI);
    reg_mat44_disp(&Iexp,(char*)"Exp-Log of Original");
    I.m[1][1]=cos(PI/3.0);
    I.m[2][2]=cos(PI/3.0);
    I.m[1][2]=-sin(PI/3.0);
    I.m[2][1]=sin(PI/3.0);
    reg_mat44_disp(&I,(char*)"Original");
    sqrtI=reg_mat44_sqrt(&I);
    reg_mat44_disp(&sqrtI,(char*)"Square root of Original");
    logI=reg_mat44_logm(&I);
    reg_mat44_disp(&logI,(char*)"Log of original");
    expI=reg_mat44_expm(&I);
    reg_mat44_disp(&expI,(char*)"Expoential of original");
    Ilog=reg_mat44_expm(&logI);
    reg_mat44_disp(&Ilog,(char*)"Log-Exp of Original");
    Iexp=reg_mat44_logm(&expI);
    reg_mat44_disp(&Iexp,(char*)"Exp-Log of Original");
    I.m[0][0]=-0.090631;
    I.m[0][1]=-0.066772;
    I.m[0][2]=1.091993;
    I.m[0][3]=-69.984901;
    I.m[1][0]=-0.849045;
    I.m[1][1]=0.250401;
    I.m[1][2]=-0.11861;
    I.m[1][3]=73.011093;
    I.m[2][0]=-0.230847;
    I.m[2][1]=-0.937538;
    I.m[2][2]=-0.089199;
    I.m[2][3]=160.140442;
    I.m[3][0]=0.0;
    I.m[3][1]=0.0;
    I.m[3][2]=0.0;
    I.m[3][3]=1.0;
    reg_mat44_disp(&I,(char*)"Original");
    sqrtI=reg_mat44_sqrt(&I);
    reg_mat44_disp(&sqrtI,(char*)"Square root of Original");
    logI=reg_mat44_logm(&I);
    reg_mat44_disp(&logI,(char*)"Log of original");
    expI=reg_mat44_expm(&I);
    reg_mat44_disp(&expI,(char*)"Expoential of original");
    Ilog=reg_mat44_expm(&logI);
    reg_mat44_disp(&Ilog,(char*)"Log-Exp of Original");
    Iexp=reg_mat44_logm(&expI);
    reg_mat44_disp(&Iexp,(char*)"Exp-Log of Original");

    return 0;
}
