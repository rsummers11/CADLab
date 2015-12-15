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

#include <vector>
#include <iostream> 
#include <string>

#include <nvmatrix.cuh>
#include <nvmatrix_operators.cuh>
#include <matrix.h>
#include <convnet.cuh>
#include <util.cuh>

using namespace std;

/* 
 * =======================
 * ConvNet
 * =======================
 */
ConvNet::ConvNet(PyListObject* layerParams, int minibatchSize, int deviceID) : Thread(false),  _deviceID(deviceID), _data(NULL) {
    //initCuda();
    cudaSetDevice(_deviceID < 0 ? cutGetMaxGflopsDeviceId() : _deviceID);
    try {
        int numLayers = PyList_GET_SIZE(layerParams);
    
        for (int i = 0; i < numLayers; i++) {
            PyObject* paramsDict = PyList_GET_ITEM(layerParams, i);
            string layerType = pyDictGetString(paramsDict, "type");
            
            Layer* l = initLayer(layerType, paramsDict);
            // Connect backward links in graph for this layer
            intv* inputLayers = pyDictGetIntV(paramsDict, "inputs");
            if (inputLayers != NULL) {
                for (int i = 0; i < inputLayers->size(); i++) {
                    l->addPrev(&getLayer(inputLayers->at(i)));
                }
            }
            delete inputLayers;
        }

        // Connect the forward links in the graph
        for (int i = 0; i < _layers.size(); i++) {
            vector<Layer*>& prev = _layers[i]->getPrev();
            for (int j = 0; j < prev.size(); j++) {
                prev[j]->addNext(_layers[i]);
            }
        }
         
        // Execute post-initialization stuff
        for (int i = 0; i < _layers.size(); i++) {
            _layers[i]->postInit();
        }
        
        _dp = new DataProvider(minibatchSize);
    } catch (string& s) {
        cout << "Error creating ConvNet: " << s << endl;
        exit(1);
    }
}

/*
 * Override this in derived classes
 */
Layer* ConvNet::initLayer(string& layerType, PyObject* paramsDict) {
    if (layerType == "fc") {
        _layers.push_back(new FCLayer(this, paramsDict));
    } else if (layerType == "fcdropo") {
        _layers.push_back(new FCDropOutLayer(this, paramsDict));
    } else if (layerType == "fcdropc") {
        _layers.push_back(new FCDropConnectApproxLayer(this, paramsDict));
    } else if (layerType == "fcdropcf") {
        //_layers.push_back(new FCDropConnectLayer(this, paramsDict));
        _layers.push_back(new FCDropConnectBitLayer(this, paramsDict));
    } else if (layerType == "conv") {
        _layers.push_back(new ConvLayer(this, paramsDict));
    } else if (layerType == "local") {
        _layers.push_back(new LocalUnsharedLayer(this, paramsDict));
    } else if (layerType == "pool") {
        _layers.push_back(&PoolLayer::makePoolLayer(this, paramsDict));
    } else if (layerType == "rnorm") {
        _layers.push_back(new ResponseNormLayer(this, paramsDict));
    } else if (layerType == "cmrnorm") {
        _layers.push_back(new CrossMapResponseNormLayer(this, paramsDict));
    } else if (layerType == "cnorm") {
        _layers.push_back(new ContrastNormLayer(this, paramsDict));
    } else if (layerType == "softmax") {
        _layers.push_back(new SoftmaxLayer(this, paramsDict));
    } else if (layerType == "eltsum") {
        _layers.push_back(new EltwiseSumLayer(this, paramsDict));
    } else if (layerType == "eltmax") {
        _layers.push_back(new EltwiseMaxLayer(this, paramsDict));
    } else if (layerType == "neuron") {
        _layers.push_back(new NeuronLayer(this, paramsDict));
    } else if (layerType == "nailbed") {
        _layers.push_back(new NailbedLayer(this, paramsDict));
    } else if (layerType == "blur") {
        _layers.push_back(new GaussianBlurLayer(this, paramsDict));
    } else if (layerType == "resize") {
        _layers.push_back(new ResizeLayer(this, paramsDict));
    } else if (layerType == "rgb2yuv") {
        _layers.push_back(new RGBToYUVLayer(this, paramsDict));
    } else if (layerType == "rgb2lab") {
        _layers.push_back(new RGBToLABLayer(this, paramsDict));
    } else if (layerType == "data") {
        DataLayer *d = new DataLayer(this, paramsDict);
        _layers.push_back(d);
        _dataLayers.push_back(d);
    } else if (strncmp(layerType.c_str(), "cost.", 5) == 0) {
        CostLayer *c = &CostLayer::makeCostLayer(this, layerType, paramsDict);
        _layers.push_back(c);
        _costs.push_back(c);
    } else {
        throw string("Unknown layer type ") + layerType;
    }

    return _layers.back();
}

/*
 * This executes in a new CPU thread so it's OK to initialize CUDA stuff here. 
 */
void ConvNet::initCuda() { 
    cudaSetDevice(_deviceID < 0 ? cutGetMaxGflopsDeviceId() : _deviceID);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    cublasInit();
    NVMatrix::initRandom(time(0));
    copyToGPU();
}

void* ConvNet::run() {
    initCuda();

    while (true) {
        Worker* worker = _workerQueue.dequeue();
        worker->run();
        delete worker;
    }
    return NULL;
}

Queue<Worker*>& ConvNet::getWorkerQueue() {
    return _workerQueue;
}

Queue<WorkResult*>& ConvNet::getResultQueue() {
    return _resultQueue;
}

DataProvider& ConvNet::getDataProvider() {
    return *_dp;
}

Layer& ConvNet::operator[](int idx) {
    return *_layers[idx];
}

Layer& ConvNet::getLayer(int idx) {
    return *_layers[idx];
}

void ConvNet::copyToCPU() {
    for (int i = 0; i < _layers.size(); i++) {
        _layers[i]->copyToCPU();
    }
}

void ConvNet::copyToGPU() {
    for (int i = 0; i < _layers.size(); i++) {
        _layers[i]->copyToGPU();
    }
}

void ConvNet::updateWeights() {
    for (int i = 0; i < _layers.size(); i++) {
        _layers[i]->updateWeights();
    }
}

void ConvNet::reset() {
    for (int i = 0; i < _layers.size(); i++) {
        _layers[i]->reset();
    }
}

int ConvNet::getNumLayers() {
    return _layers.size();
}

void ConvNet::bprop(PASS_TYPE passType) {
    for (int i = 0; i < _costs.size(); i++) {
        _costs[i]->bprop(passType);
    }
    reset();
}

void ConvNet::fprop(PASS_TYPE passType) {
    assert(_data != NULL);
    reset();
    for (int i = 0; i < _dataLayers.size(); i++) {
        _dataLayers[i]->fprop(_data->getData(), passType);
    }
}

void ConvNet::fprop(GPUData& data, PASS_TYPE passType) {
    if (&data != _data) {
        delete _data;
    }
    _data = &data;
    fprop(passType);
}

void ConvNet::fprop(int miniIdx, PASS_TYPE passType) {
    delete _data;
    _data = &_dp->getMinibatch(miniIdx);
    fprop(passType);
}

Cost& ConvNet::getCost() {
    return *new Cost(_data->getNumCases(), _costs);
}

// Same as getCost() but adds results to given cost and returns it
Cost& ConvNet::getCost(Cost& cost) {
    Cost& newCost = getCost();
    cost += newCost;
    delete &newCost;
    return cost;
}

double ConvNet::getCostValue() {
    Cost& cost = getCost();
    double val = cost.getValue();
    delete &cost;
    return val;
}

/*
 * Gradient checking stuff
 */
void ConvNet::checkGradients() {
    _numFailures = 0;
    _numTests = 0;
    fprop(0, PASS_GC);
    _baseErr = getCostValue();
    bprop(PASS_GC);
    
    for (vector<Layer*>::iterator it = _layers.begin(); it != _layers.end(); ++it) {
        (*it)->checkGradients();
    }
    
    cout << "------------------------" << endl;
    if (_numFailures > 0) {
        cout << _numFailures << "/" << _numTests << " TESTS FAILED" << endl;
    } else {
        cout << "ALL " << _numTests << " TESTS PASSED" << endl;
    }
}

/*
 * name: weight matrix name
 * eps: finite difference step
 */
bool ConvNet::checkGradient(const string& name, float eps, Weights& weights) {
    Matrix numGrad(weights.getNumRows(), weights.getNumCols());
    Matrix diff(numGrad);
    numGrad.apply(Matrix::ZERO);
    Matrix weightsCPU;

    weights.getW().copyToHost(weightsCPU, true);

    for(int i = 0; i < weights.getNumRows(); i++) {
        for (int j = 0; j < weights.getNumCols(); j++) {
            float v = weightsCPU(i,j);
            weightsCPU(i,j) += eps;
            weights.getW().copyFromHost(weightsCPU);
            weightsCPU(i,j) = v;
            fprop(PASS_GC);
            double err = getCostValue();
            numGrad(i,j) = (err - _baseErr) / (_data->getNumCases() * eps);
            if (isnan(numGrad(i,j)) || isinf(numGrad(i,j))) {
                cout << "Numerical computation produced nan or inf when checking '" << name << "': " << numGrad(i,j) << endl;
                cout << "Consider reducing the sizes of the weights or finite difference steps." << endl;
                cout << "Exiting." << endl;
                exit(1);
            }
            weights.getW().copyFromHost(weightsCPU);
        }
    }

    Matrix gradCPU;
    weights.getGrad().copyToHost(gradCPU, true);
    gradCPU.scale(-1.0 / _data->getNumCases());
    float analNorm = gradCPU.norm();
    float numNorm = numGrad.norm();
    numGrad.subtract(gradCPU, diff);
    float relErr = diff.norm() / analNorm;
    bool fail = relErr >= GC_REL_ERR_THRESH;
    if (fail || !GC_SUPPRESS_PASSES) {
        cout << "========================" << endl;
        printf("(%s) %s GRADIENT CHECK\n", fail ? "****FAIL****" : "PASS", name.c_str());
        cout << "========================" << endl;
        cout << "Analytic:" << endl;
        gradCPU.print(6,4);
        cout << "Numeric:" << endl;
        numGrad.print(6,4);
        printf("Analytic norm: %e\n", analNorm);
        printf("Numeric norm:  %e\n", numNorm);
        printf("Relative error: %e\n", relErr);
    }
    _numTests++;
    _numFailures += fail;
    return fail;
}

void ConvNet::scaleEps( float scale ) {
    for( int i = 0; i < getNumLayers(); i++ ) {
        WeightLayer* layer = dynamic_cast<WeightLayer*>( _layers[i] );
        if( layer != NULL ) {
            layer->scaleEps( scale );
        }
    }
}

void ConvNet::setDropRate( float dropRate ) {
    for( int i = 0; i < getNumLayers(); i++ ) {
        FCDropLayer* layer = dynamic_cast<FCDropLayer*>( _layers[i] );
        if( layer != NULL ) {
            layer->set_dropRate( dropRate );
        }
    }
}

void ConvNet::resetMom() {
   for( int i = 0; i < getNumLayers(); i++ ) {
      WeightLayer* layer = dynamic_cast<WeightLayer*>( _layers[i] );
      if( layer != NULL ) {
         layer->resetMom( );
      }
   }
}

void ConvNet::enableMCInference( int numSamples ) {
   for( int i = 0; i < getNumLayers(); i++ ) {
      FCDropConnectBitLayer* layer = dynamic_cast<FCDropConnectBitLayer*>( _layers[i] );
      if( layer != NULL ) {
         layer->enableMCInference( numSamples );
      }
   }
}
