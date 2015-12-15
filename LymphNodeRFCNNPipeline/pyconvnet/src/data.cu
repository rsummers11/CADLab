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

#include <algorithm>
#include <data.cuh>

using namespace std;

DataProvider::DataProvider(int minibatchSize) : 
    _minibatchSize(minibatchSize), _hData(NULL) {

}

GPUData& DataProvider::operator[](int idx) {
    return getMinibatch(idx);
}

void DataProvider::clearData() {
    delete _hData;
    _hData = NULL;
    _dataSize = 0;
}

void DataProvider::setData(CPUData& hData) {
    // This is now deleted by the DataWorker's destructor
//    delete _hData; // Delete old CPU matrices

    _hData = &hData;
    _dataSize = 0;
    for (int i = 0; i < hData.getSize(); i++) {
        _dataSize += hData[i].getNumDataBytes();
    }
    _dataSize /= 1024 * 1024;
    if (_dataSize < MAX_DATA_ON_GPU) {
        for (int i = 0; i < hData.getSize(); i++) {
            if (i >= _data.size()) {
                _data.push_back(new NVMatrix());
            }
            _data[i]->copyFromHost(hData[i], true);
        }
    }
}

GPUData& DataProvider::getMinibatch(int idx) {
    assert(idx >= 0 && idx < getNumMinibatches());
    return getDataSlice(idx * _minibatchSize, (idx + 1) * _minibatchSize);
}

GPUData& DataProvider::getDataSlice(int startCase, int endCase) {
    assert(_hData != NULL);
    assert(_hData->getNumCases() > 0);
    
    NVMatrixV& miniData = *new NVMatrixV();
    
    for (int i = 0; i < _hData->getData().size(); i++) {
        miniData.push_back(new NVMatrix());
        if (_dataSize < MAX_DATA_ON_GPU) {
            if (_data[i]->isTrans()) {
                _data[i]->sliceRows(startCase, min(_hData->getNumCases(), endCase), *miniData[i]);
            } else {
                _data[i]->sliceCols(startCase, min(_hData->getNumCases(), endCase), *miniData[i]);
            }
        } else {
            Matrix tmp;
            if ((*_hData)[i].isTrans()) {
                (*_hData)[i].sliceRows(startCase, min(_hData->getNumCases(), endCase), tmp);
            } else {
                (*_hData)[i].sliceCols(startCase, min(_hData->getNumCases(), endCase), tmp);
            }
            miniData.back()->copyFromHost(tmp, true);
        }
    }

    return *new GPUData(miniData);
}

int DataProvider::getNumMinibatches() {
    assert(_hData != NULL);
    assert(_hData->getNumCases() > 0);
    return DIVUP(_hData->getNumCases(), _minibatchSize);
}

int DataProvider::getMinibatchSize() {
    return _minibatchSize;
}

int DataProvider::getNumCases() {
    assert(_hData != NULL);
    assert(_hData->getNumCases() > 0);
    return _hData->getNumCases();
}

int DataProvider::getNumCasesInMinibatch(int idx) {
    assert(_hData != NULL);
    assert(_hData->getNumCases() > 0);
    assert(idx >= 0 && idx < getNumMinibatches());
    return min(_minibatchSize, max(0, _hData->getNumCases() - idx * _minibatchSize));
}