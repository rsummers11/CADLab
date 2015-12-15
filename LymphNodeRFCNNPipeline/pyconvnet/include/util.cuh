/* 
 * Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * 
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef UTIL_H
#define	UTIL_H

#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <string>
#include <Python.h>
#include <nvmatrix.cuh>
#include <matrix.h>

/*
 * The types of passes that the convnet supports. Used in the fprop and bprop functions in
 * ConvNet class. Most of the layers ignore the pass type, but some make use of it.
 */
enum PASS_TYPE {PASS_TRAIN, PASS_TEST, PASS_GC};

// For gradient checking
#define GC_SUPPRESS_PASSES          true
#define GC_REL_ERR_THRESH           0.02

/*
 * Store entire data matrix on GPU if its size does not exceed this many MB.
 * Otherwise store only one minibatch at a time.
 */ 
#define MAX_DATA_ON_GPU             200 

typedef std::vector<Matrix*> MatrixV;
typedef std::vector<NVMatrix*> NVMatrixV;
typedef std::map<std::string,std::vector<double>*> CostMap;
typedef std::map<std::string,double> CostCoeffMap;
typedef std::vector<double> doublev;
typedef std::vector<float> floatv;
typedef std::vector<int> intv;

floatv* getFloatV(PyObject* pyList);
intv* getIntV(PyObject* pyList);
MatrixV* getMatrixV(PyObject* pyList);
int* getIntA(PyObject* pyList);

int pyDictGetInt(PyObject* dict, const char* key);
intv* pyDictGetIntV(PyObject* dict, const char* key);
std::string pyDictGetString(PyObject* dict, const char* key);
float pyDictGetFloat(PyObject* dict, const char* key);
floatv* pyDictGetFloatV(PyObject* dict, const char* key);
Matrix* pyDictGetMatrix(PyObject* dict, const char* key);
MatrixV* pyDictGetMatrixV(PyObject* dict, const char* key);
int* pyDictGetIntA(PyObject* dict, const char* key);

template<typename T>
std::string tostr(T n) {
    std::ostringstream result;
    result << n;
    return result.str();
}

#endif	/* UTIL_H */

