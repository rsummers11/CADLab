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

#ifndef PYCONVNET3_CUH
#define	PYCONVNET3_CUH

#define _QUOTEME(x) #x
#define QUOTEME(x) _QUOTEME(x)

extern "C" void INITNAME();

PyObject* initModel(PyObject *self, PyObject *args);
PyObject* startBatch(PyObject *self, PyObject *args);
PyObject* finishBatch(PyObject *self, PyObject *args);
PyObject* checkGradients(PyObject *self, PyObject *args);
PyObject* syncWithHost(PyObject *self, PyObject *args);
PyObject* startMultiviewTest(PyObject *self, PyObject *args);
PyObject* startFeatureWriter(PyObject *self, PyObject *args);

// ---- my option ---
PyObject* scaleModelEps(PyObject *self, PyObject *args);
PyObject* setDropRate(PyObject *self, PyObject *args);
PyObject* resetModelMom(PyObject *self, PyObject *args);
PyObject* preprocess(PyObject *self, PyObject *args);
PyObject* startMultiviewFeatureWriter(PyObject *self, PyObject *args);
PyObject* enableMCInference( PyObject *self, PyObject *args);

#endif	/* PYCONVNET3_CUH */

