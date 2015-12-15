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

#include <Python.h>
#include <arrayobject.h>
#include <assert.h>
#include <cutil_inline.h>
#include <cublas.h>
#include <time.h>
#include <vector>

#include <matrix.h>
#include <queue.h>
#include <worker.cuh>
#include <util.cuh>
#include <cost.cuh>
#include "tdata/preprocess.hpp"

#include <pyconvnet.cuh>
#include <convnet.cuh>

using namespace std;
static ConvNet* model = NULL;

static PyMethodDef _ConvNetMethods[] = {  
   { "initModel",          initModel,          METH_VARARGS },
   { "startBatch",         startBatch,         METH_VARARGS },
   { "finishBatch",        finishBatch,        METH_VARARGS },
   { "checkGradients",     checkGradients,     METH_VARARGS },
   { "startMultiviewTest", startMultiviewTest, METH_VARARGS },
   { "startFeatureWriter",  startFeatureWriter,         METH_VARARGS },
   { "syncWithHost",       syncWithHost,       METH_VARARGS },
   // ---- my option ---
   { "scaleModelEps",       scaleModelEps,       METH_VARARGS },
   { "resetModelMom",       resetModelMom,       METH_VARARGS },
   { "setDropRate",       setDropRate,       METH_VARARGS },
   { "preprocess",         preprocess,         METH_VARARGS },
   { "startMultiviewFeatureWriter", startMultiviewFeatureWriter, METH_VARARGS },
   { "enableMCInference", enableMCInference, METH_VARARGS },
   { NULL, NULL }
};

#if defined(_WIN64) || defined(_WIN32)
extern "C" __declspec(dllexport) void initpyconvnet() {
    (void) Py_InitModule("pyconvnet", _ConvNetMethods);
    import_array();
}
#else
void INITNAME() {
    (void) Py_InitModule(QUOTEME(MODELNAME), _ConvNetMethods);
    import_array();
}
#endif

PyObject* initModel(PyObject *self, PyObject *args) {
    assert(model == NULL);

    PyListObject* pyLayerParams;
    int pyMinibatchSize;
    int pyDeviceID;

    if (!PyArg_ParseTuple(args, "O!ii",
                          &PyList_Type, &pyLayerParams,
                          &pyMinibatchSize,
                          &pyDeviceID)) {
        return NULL;
    }
    model = new ConvNet(pyLayerParams,
                        pyMinibatchSize,
                        pyDeviceID);

    model->start();
    return Py_BuildValue("i", 0);
}

/*
 * Starts training/testing on the given batch (asynchronous -- returns immediately).
 */
PyObject* startBatch(PyObject *self, PyObject *args) {
    assert(model != NULL);
    PyListObject* data;
    int test = 0;
    if (!PyArg_ParseTuple(args, "O!|i",
        &PyList_Type, &data,
        &test)) {
        return NULL;
    }
    MatrixV& mvec = *getMatrixV((PyObject*)data);
    
    TrainingWorker* wr = new TrainingWorker(*model, *new CPUData(mvec), test);
    model->getWorkerQueue().enqueue(wr);
    return Py_BuildValue("i", 0);
}

/*
 * Starts testing on the given batch (asynchronous -- returns immediately).
 */
PyObject* startMultiviewTest(PyObject *self, PyObject *args) {
    assert(model != NULL);
    PyListObject* data;
    int numViews, logregIdx;
    if (!PyArg_ParseTuple(args, "O!ii",
        &PyList_Type, &data,
        &numViews,
        &logregIdx)) {
        return NULL;
    }
    MatrixV& mvec = *getMatrixV((PyObject*)data);
    
    MultiviewTestWorker* wr = new MultiviewTestWorker(*model, *new CPUData(mvec), numViews, logregIdx);
    model->getWorkerQueue().enqueue(wr);
    return Py_BuildValue("i", 0);
}

PyObject* startFeatureWriter(PyObject *self, PyObject *args) {
    assert(model != NULL);
    PyListObject* data;
    int layerIdx;
    int pass_type;
    if (!PyArg_ParseTuple(args, "O!ii", &PyList_Type, &data, &layerIdx, &pass_type)) 
	{
		return NULL;
    }
    MatrixV& mvec = *getMatrixV((PyObject*)data);
    Matrix& ftrs = *mvec.back();
    mvec.pop_back();
    
    FeatureWorker* wr = new FeatureWorker(*model, *new CPUData(mvec), ftrs, layerIdx);
    wr->set_passType( pass_type );
    model->getWorkerQueue().enqueue(wr);
    return Py_BuildValue("i", 0);
}

/*
 * Waits for the trainer to finish training on the batch given to startBatch.
 */
PyObject* finishBatch(PyObject *self, PyObject *args) {
    assert(model != NULL);
    WorkResult* res = model->getResultQueue().dequeue();
    assert(res != NULL);
    assert(res->getResultType() == WorkResult::BATCH_DONE);
    
    Cost& cost = res->getResults();
    PyObject* dict = PyDict_New();
    CostMap& costMap = cost.getCostMap();
    for (CostMap::const_iterator it = costMap.begin(); it != costMap.end(); ++it) {
        PyObject* v = PyList_New(0);
        for (vector<double>::const_iterator iv = it->second->begin(); iv != it->second->end(); ++iv) {
            PyObject* f = PyFloat_FromDouble(*iv);
            PyList_Append(v, f);
        }
        PyDict_SetItemString(dict, it->first.c_str(), v);
    }
    
    PyObject* retVal = Py_BuildValue("Ni", dict, cost.getNumCases());
    delete res; // Deletes cost too
    return retVal;
}

PyObject* checkGradients(PyObject *self, PyObject *args) {
    assert(model != NULL);
    PyListObject* data;
    if (!PyArg_ParseTuple(args, "O!",
        &PyList_Type, &data)) {
        return NULL;
    }
    MatrixV& mvec = *getMatrixV((PyObject*)data);
    
    GradCheckWorker* wr = new GradCheckWorker(*model, *new CPUData(mvec));
    model->getWorkerQueue().enqueue(wr);
    WorkResult* res = model->getResultQueue().dequeue();
    assert(res != NULL);
    assert(res->getResultType() == WorkResult::BATCH_DONE);
    delete res;
    return Py_BuildValue("i", 0);
}

/*
 * Copies weight matrices from GPU to system memory.
 */
PyObject* syncWithHost(PyObject *self, PyObject *args) {
    assert(model != NULL);
    SyncWorker* wr = new SyncWorker(*model);
    model->getWorkerQueue().enqueue(wr);
    WorkResult* res = model->getResultQueue().dequeue();
    assert(res != NULL);
    assert(res->getResultType() == WorkResult::SYNC_DONE);
    
    delete res;
    return Py_BuildValue("i", 0);
}

/**
 *  scale model learning rate
 */
PyObject* scaleModelEps( PyObject *self, PyObject *args) {
    assert(model != NULL);
    float scale;
    if( !PyArg_ParseTuple( args, "f", &scale ) ) 
        return NULL;
    model->scaleEps( scale );
    return Py_BuildValue("i", 0);
}

/**
 *  reset model mom 
 */
PyObject* resetModelMom( PyObject *self, PyObject *args) {
    assert(model != NULL);
    model->resetMom();
    return Py_BuildValue("i", 0);
}

/**
 *  scale model learning rate
 */
PyObject* setDropRate( PyObject *self, PyObject *args) {
    assert(model != NULL);
    float dropRate;
    if( !PyArg_ParseTuple( args, "f", &dropRate ) ) 
        return NULL;
    model->setDropRate( dropRate );
    return Py_BuildValue("i", 0);
}

PyObject* preprocess(PyObject* self, PyObject* args) {
    float scale, rotate;
    int w, h, d;
    PyListObject* data;
    if( !PyArg_ParseTuple( args, "O!iiiff", 
                &PyList_Type, &data, 
                &w, &h, &d,
                &scale, &rotate) ) {
        return NULL;
    }

    MatrixV& mvec = *getMatrixV((PyObject*)data);
    assert( mvec.size() == 1 );
    preprocessData( w, h, d, scale, rotate, *mvec[0] );

    return Py_BuildValue("i", 1);
}

PyObject* startMultiviewFeatureWriter (PyObject *self, PyObject *args) {
    assert(model != NULL);
    PyListObject* data;
    int numViews;
    int logregIdx;
    if (!PyArg_ParseTuple(args, "O!ii",
        &PyList_Type, &data,
        &numViews,&logregIdx )) {
        return NULL;
    }
    MatrixV& mvec = *getMatrixV((PyObject*)data);
    Matrix& ftrs = *mvec.back();
    mvec.pop_back();
    
    MultiviewFeatureWorker* wr = new MultiviewFeatureWorker(*model, *new CPUData(mvec), 
          ftrs, numViews, logregIdx );
    model->getWorkerQueue().enqueue(wr);
    return Py_BuildValue("i", 0);
}

PyObject* enableMCInference( PyObject *self, PyObject *args) {
    assert(model != NULL);
    int numSamples;
    if( !PyArg_ParseTuple( args, "i", &numSamples ) ) 
        return NULL;
    model->enableMCInference( numSamples );
    return Py_BuildValue("i", 0);
}

