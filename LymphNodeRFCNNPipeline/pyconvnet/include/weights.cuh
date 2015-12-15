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

#ifndef WEIGHTS_CUH
#define	WEIGHTS_CUH

#include <string>
#include <vector>
#include <iostream>
#include <cutil_inline.h>
#include <assert.h>
#include <nvmatrix.cuh>
#include <matrix.h>
#include "util.cuh"

using namespace std;

class Weights {
private:
    Matrix* _hWeights, *_hWeightsInc;
    NVMatrix* _weights, *_weightsInc, *_weightsGrad;
    
    float _epsW, _wc, _mom;
    bool _onGPU, _useGrad;
    int _numUpdates;
    static bool _autoCopyToGPU;
    
    // Non-NULL if these weights are really shared from some other layer
    Weights* _srcWeights;
 
public:
    NVMatrix& operator*() {
        return getW();
    }
    
    Weights(Weights& srcWeights, float epsW) : _srcWeights(&srcWeights), _epsW(epsW), _wc(0), _onGPU(false), _numUpdates(0),
                                               _weights(NULL), _weightsInc(NULL), _weightsGrad(NULL){
        _hWeights = &srcWeights.getCPUW();
        _hWeightsInc = &srcWeights.getCPUWInc();
        _mom = srcWeights.getMom();
        _useGrad = srcWeights.isUseGrad();   
        if (_autoCopyToGPU) {
            copyToGPU();
        }
    }
    
    Weights(Matrix& hWeights, Matrix& hWeightsInc, float epsW, float wc, float mom, bool useGrad)
        : _srcWeights(NULL), _hWeights(&hWeights), _hWeightsInc(&hWeightsInc), _numUpdates(0),
          _epsW(epsW), _wc(wc), _mom(mom), _useGrad(useGrad), _onGPU(false), _weights(NULL),
          _weightsInc(NULL), _weightsGrad(NULL) {
        if (_autoCopyToGPU) {
            copyToGPU();
        }
    }
        
    ~Weights() {
        delete _hWeights;
        delete _hWeightsInc;
        if (_srcWeights == NULL) {
            delete _weights;
            delete _weightsInc;
            delete _weightsGrad;
        }
    }

    static void setAutoCopyToGPU(bool autoCopyToGPU) {
        _autoCopyToGPU = autoCopyToGPU;
    }
    
    NVMatrix& getW() {
        assert(_onGPU);
        return *_weights;
    }
    
    NVMatrix& getInc() {
        assert(_onGPU);
        return *_weightsInc;
    }
        
    NVMatrix& getGrad() {
        assert(_onGPU);
        return _useGrad ? *_weightsGrad : *_weightsInc;
    }
    
    Matrix& getCPUW() {
        return *_hWeights;
    }
    
    Matrix& getCPUWInc() {
        return *_hWeightsInc;
    }
    
    int getNumRows() const {
        return _hWeights->getNumRows();
    }
    
    int getNumCols() const {
        return _hWeights->getNumCols();
    }
    
    void copyToCPU() {
        if (_srcWeights == NULL) {
            assert(_onGPU);
            _weights->copyToHost(*_hWeights);
            _weightsInc->copyToHost(*_hWeightsInc);
        }
    }
    
    // This function is assumed to be called in the order in which the layers
    // were defined
    void copyToGPU() {
        if (_srcWeights == NULL) {
            _weights = new NVMatrix();
            _weightsInc = new NVMatrix();
            _weights->copyFromHost(*_hWeights, true);
            _weightsInc->copyFromHost(*_hWeightsInc, true);
            _weightsGrad = _useGrad ? new NVMatrix() : NULL;
        } else {
            _weights = _srcWeights->_weights;
            _weightsInc = _srcWeights->_weightsInc;
            _weightsGrad = _srcWeights->_weightsGrad;
        }
        _onGPU = true;
    }
    
    // Scale your gradient by epsW / numCases!
    void update() {
        // Only true owner of weights updates
        if (_srcWeights == NULL && _epsW > 0) {
            assert(_onGPU);
            if (_useGrad) {
                _weightsInc->add(*_weightsGrad, _mom, 1);
            }
            if (_wc > 0) {
                _weightsInc->add(*_weights, -_wc * _epsW);
            }
            _weights->add(*_weightsInc);
            _numUpdates = 0;
        }
    }
    
    int incNumUpdates() {
        if (_srcWeights != NULL) {
            return _srcWeights->incNumUpdates();
        }
        return _numUpdates++;
    }
    
    // Returns the number of times a gradient has been computed for this
    // weight matrix during the current pass (interval between two calls of update())
    // through the net. This number will only be greater than 1 if this weight matrix
    // is *shared* by multiple layers in the net.
    int getNumUpdates() const {
        if (_srcWeights != NULL) {
            return _srcWeights->getNumUpdates();
        }
        return _numUpdates;
    }
    
    float getEps() const {
        return _epsW;
    }

    void setEps( float epsW ) {
        _epsW = epsW;
    }
    
    float getMom() const {
        return _mom;
    }

    void resetMom() {
       _mom = 0;
    }
    
    float getWC() const {
        return _wc;
    }
    
    bool isUseGrad() const { // is good grammar
        return _useGrad;
    }
};

class WeightList {
private:
    std::vector<Weights*> _weightList;

public:
    Weights& operator[](const int idx) const {
        return *_weightList[idx];
    }
    
    ~WeightList() {
        for (int i = 0; i < _weightList.size(); i++) {
            delete _weightList[i];
        }
    }
    
//    WeightList(MatrixV& hWeights, MatrixV& hWeightsInc, floatv& epsW, floatv& wc, floatv& mom, bool useGrads) : _initialized(false) {
//        initialize(hWeights, hWeightsInc, epsW, wc, mom, useGrads);
//    }
    
    WeightList() {
    }
    
//    void initialize(MatrixV& hWeights, MatrixV& hWeightsInc, floatv& epsW, floatv& wc, floatv& mom, bool useGrads) {
//        for (int i = 0; i < hWeights.size(); i++) {
//            _weightList.push_back(new Weights(*hWeights[i], *hWeightsInc[i], epsW[i], wc[i], mom[i], useGrads));
//        }
//        _initialized = true;
//        delete &hWeights;
//        delete &hWeightsInc;
//        delete &epsW;
//        delete &wc;
//        delete &mom;
//    }
    
    void addWeights(Weights& w) {
        _weightList.push_back(&w);
    }
    
//    void addWeights(WeightList& wl) {
//        for (int i = 0; i < wl.getSize(); i++) {
//            addWeights(wl[i]);
//        }
//    }
    
    void update() {
        for (int i = 0; i < getSize(); i++) {
            _weightList[i]->update();
        }
    }

    void copyToCPU() {
        for (int i = 0; i < getSize(); i++) {
            _weightList[i]->copyToCPU();
        }
    }

    void copyToGPU() {
        for (int i = 0; i < getSize(); i++) {
            _weightList[i]->copyToGPU();
        }
    }
    
    int getSize() {
        return _weightList.size();
    }
};

#endif	/* WEIGHTS_CUH */
