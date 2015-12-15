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

#include <util.cuh>

using namespace std;

floatv* getFloatV(PyObject* pyList) {
    if (pyList == NULL) {
        return NULL;
    }
    floatv* vec = new floatv(); 
    for (int i = 0; i < PyList_GET_SIZE(pyList); i++) {
        vec->push_back(PyFloat_AS_DOUBLE(PyList_GET_ITEM(pyList, i)));
    }
    return vec;
}

intv* getIntV(PyObject* pyList) {
    if (pyList == NULL) {
        return NULL;
    }
    intv* vec = new intv(); 
    for (int i = 0; i < PyList_GET_SIZE(pyList); i++) {
        vec->push_back(PyInt_AS_LONG(PyList_GET_ITEM(pyList, i)));
    }
    return vec;
}

int* getIntA(PyObject* pyList) {
    if (pyList == NULL) {
        return NULL;
    }
    int* arr = new int[PyList_GET_SIZE(pyList)];
    for (int i = 0; i < PyList_GET_SIZE(pyList); i++) {
        arr[i] = PyInt_AS_LONG(PyList_GET_ITEM(pyList, i));
    }
    return arr;
}
MatrixV* getMatrixV(PyObject* pyList) {
    if (pyList == NULL) {
        return NULL;
    }
    MatrixV* vec = new MatrixV(); 
    for (int i = 0; i < PyList_GET_SIZE(pyList); i++) {
        vec->push_back(new Matrix((PyArrayObject*)PyList_GET_ITEM(pyList, i)));
    }
    return vec;
}

int pyDictGetInt(PyObject* dict, const char* key) {
    return PyInt_AS_LONG(PyDict_GetItemString(dict, key));
}

intv* pyDictGetIntV(PyObject* dict, const char* key) {
    return getIntV(PyDict_GetItemString(dict, key));
}

int* pyDictGetIntA(PyObject* dict, const char* key) {
    return getIntA(PyDict_GetItemString(dict, key));
}

string pyDictGetString(PyObject* dict, const char* key) {
    return string(PyString_AS_STRING(PyDict_GetItemString(dict, key)));
}

float pyDictGetFloat(PyObject* dict, const char* key) {
    return PyFloat_AS_DOUBLE(PyDict_GetItemString(dict, key));
}

floatv* pyDictGetFloatV(PyObject* dict, const char* key) {
    return getFloatV(PyDict_GetItemString(dict, key));
}

Matrix* pyDictGetMatrix(PyObject* dict, const char* key) {
    return new Matrix((PyArrayObject*)PyDict_GetItemString(dict, key));
}

MatrixV* pyDictGetMatrixV(PyObject* dict, const char* key) {
    return getMatrixV(PyDict_GetItemString(dict, key));
}