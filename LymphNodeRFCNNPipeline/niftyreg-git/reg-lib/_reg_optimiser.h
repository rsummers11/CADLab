/** @file _reg_optimiser.h
 * @author Marc Modat
 * @date 20/07/2012
 */

#ifndef _REG_OPTIMISER_H
#define _REG_OPTIMISER_H

#include "_reg_maths.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/** @brief Interface between the registration class and the optimiser
 */
class InterfaceOptimiser
{
public:
   /// @brief Returns the registration current objective function value
   virtual double GetObjectiveFunctionValue() = 0;
   /// @brief The transformation parameters are optimised
   virtual void UpdateParameters(float) = 0;
   /// @brief The best objective function values are stored
   virtual void UpdateBestObjFunctionValue() = 0;

protected:
   /// @brief Interface constructor
   InterfaceOptimiser() {}
   /// @brief Interface destructor
   ~InterfaceOptimiser() {}
};
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/** @class reg_optimiser
 * @brief Standard gradient acent optimisation
 */
template <class T>
class reg_optimiser
{
protected:
   bool backward;
   size_t dofNumber;
   size_t dofNumber_b;
   size_t ndim;
   T *currentDOF; // pointer to the cpp nifti image array
   T *currentDOF_b; // pointer to the cpp nifti image array (backward)
   T *bestDOF;
   T *bestDOF_b;
   T *gradient;
   T *gradient_b;
   bool optimiseX;
   bool optimiseY;
   bool optimiseZ;
   size_t maxIterationNumber;
   size_t currentIterationNumber;
   double bestObjFunctionValue;
   double currentObjFunctionValue;
   InterfaceOptimiser *objFunc;

public:
   reg_optimiser();
   virtual ~reg_optimiser();
   virtual void StoreCurrentDOF();
   virtual void RestoreBestDOF();
   virtual size_t GetDOFNumber()
   {
      return this->dofNumber;
   }
   virtual size_t GetDOFNumber_b()
   {
      return this->dofNumber_b;
   }
   virtual size_t GetNDim()
   {
      return this->ndim;
   }
   virtual size_t GetVoxNumber()
   {
      return this->dofNumber/this->ndim;
   }
   virtual size_t GetVoxNumber_b()
   {
      return this->dofNumber_b/this->ndim;
   }
   virtual T* GetBestDOF()
   {
      return this->bestDOF;
   }
   virtual T* GetBestDOF_b()
   {
      return this->bestDOF_b;
   }
   virtual T* GetCurrentDOF()
   {
      return this->currentDOF;
   }
   virtual T* GetCurrentDOF_b()
   {
      return this->currentDOF_b;
   }
   virtual T* GetGradient()
   {
      return this->gradient;
   }
   virtual T* GetGradient_b()
   {
      return this->gradient_b;
   }
   virtual bool GetOptimiseX()
   {
      return this->optimiseX;
   }
   virtual bool GetOptimiseY()
   {
      return this->optimiseY;
   }
   virtual bool GetOptimiseZ()
   {
      return this->optimiseZ;
   }
   virtual size_t GetMaxIterationNumber()
   {
      return this->maxIterationNumber;
   }
   virtual size_t GetCurrentIterationNumber()
   {
      return this->currentIterationNumber;
   }
   virtual double GetBestObjFunctionValue()
   {
      return this->bestObjFunctionValue;
   }
   virtual void SetBestObjFunctionValue(double i)
   {
      this->bestObjFunctionValue=i;
   }
   virtual double GetCurrentObjFunctionValue()
   {
      return this->currentObjFunctionValue;
   }
   virtual void IncrementCurrentIterationNumber()
   {
      this->currentIterationNumber++;
   }
   virtual void Initialise(size_t nvox,
                           int dim,
                           bool optX,
                           bool optY,
                           bool optZ,
                           size_t maxit,
                           size_t start,
                           InterfaceOptimiser *o,
                           T *cppData,
                           T *gradData=NULL,
                           size_t nvox_b=0,
                           T *cppData_b=NULL,
                           T *gradData_b=NULL);
   virtual void Optimise(T maxLength,
                         T smallLength,
                         T &startLength);
   virtual void Perturbation(float length);

   // Function used for testing
   virtual void reg_test_optimiser();
};
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/** @class reg_conjugateGradient
 * @brief Conjugate gradient acent optimisation
 */
template <class T>
class reg_conjugateGradient : public reg_optimiser<T>
{
protected:
   T *array1;
   T *array1_b;
   T *array2;
   T *array2_b;
   bool firstcall;

   void UpdateGradientValues(); /// @brief Update the gradient array

public:
   reg_conjugateGradient();
   ~reg_conjugateGradient();
   virtual void Initialise(size_t nvox,
                           int dim,
                           bool optX,
                           bool optY,
                           bool optZ,
                           size_t maxit,
                           size_t start,
                           InterfaceOptimiser *o,
                           T *cppData=NULL,
                           T *gradData=NULL,
                           size_t nvox_b=0,
                           T *cppData_b=NULL,
                           T *gradData_b=NULL);
   virtual void Optimise(T maxLength,
                         T smallLength,
                         T &startLength);
   virtual void Perturbation(float length);

   // Function used for testing
   virtual void reg_test_optimiser();
};
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/** @class Global optimisation class
 * @brief
 */
template <class T>
class reg_lbfgs : public reg_optimiser<T>
{
protected:
   size_t stepToKeep;
   T *oldDOF;
   T *oldGrad;
   T **diffDOF;
   T **diffGrad;

public:
   reg_lbfgs();
   ~reg_lbfgs();
   virtual void Initialise(size_t nvox,
                           int dim,
                           bool optX,
                           bool optY,
                           bool optZ,
                           size_t maxit,
                           size_t start,
                           InterfaceOptimiser *o,
                           T *cppData=NULL,
                           T *gradData=NULL,
                           size_t nvox_b=0,
                           T *cppData_b=NULL,
                           T *gradData_b=NULL);
   virtual void Optimise(T maxLength,
                         T smallLength,
                         T &startLength);
   virtual void UpdateGradientValues();
};
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
#include "_reg_optimiser.cpp"

#endif // _REG_OPTIMISER_H
