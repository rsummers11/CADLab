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

#ifndef CONVNET3
#define	CONVNET3

#include <vector>
#include <string>
#include <cutil_inline.h>
#include <time.h>
#include <queue.h>
#include <thread.h>
#include <math.h>

#include "layer.cuh"
#include "data.cuh"
#include "worker.cuh"
#include "weights.cuh"

class Worker;
class WorkResult;
class Layer;
class DataLayer;
class CostLayer;

class ConvNet : public Thread {
protected:
    std::vector<Layer*> _layers;
    std::vector<DataLayer*> _dataLayers;
    std::vector<CostLayer*> _costs;
    GPUData* _data;

    DataProvider* _dp;
    int _deviceID;
    
    Queue<Worker*> _workerQueue;
    Queue<WorkResult*> _resultQueue;
    
    // For gradient checking
    int _numFailures;
    int _numTests;
    double _baseErr;
    
    virtual Layer* initLayer(string& layerType, PyObject* paramsDict);
    void initCuda();
    void* run();
public:
    ConvNet(PyListObject* layerParams, int minibatchSize, int deviceID);
    
    Queue<Worker*>& getWorkerQueue();
    Queue<WorkResult*>& getResultQueue();
    DataProvider& getDataProvider();
    
    Layer& operator[](int idx);
    Layer& getLayer(int idx);
    void copyToCPU();
    void copyToGPU();
    void updateWeights();
    void reset();
    int getNumLayers();
    
    void bprop(PASS_TYPE passType);
    void fprop(PASS_TYPE passType);
    void fprop(int miniIdx, PASS_TYPE passType);
    void fprop(GPUData& data, PASS_TYPE passType);

    bool checkGradient(const std::string& name, float eps, Weights& weights); 
    void checkGradients();
    Cost& getCost();
    Cost& getCost(Cost& cost);
    double getCostValue();

    void scaleEps( float scale );
    void setDropRate( float dropRate );
    void resetMom();
    void enableMCInference( int numSamples );
};

#endif	/* CONVNET3 */

