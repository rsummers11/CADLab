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

#ifndef DATA_CUH
#define	DATA_CUH

#include <vector>
#include <algorithm>
#include "util.cuh"

template <class T>
class Data {
protected:
    std::vector<T*>* _data;
public:
    typedef typename std::vector<T*>::iterator T_iter;
    
    Data(std::vector<T*>& data) : _data(&data) {
        assert(_data->size() > 0);
        for (int i = 1; i < data.size(); i++) {
            assert(data[i-1]->getLeadingDim() == data[i]->getLeadingDim());
        }
        assert(data[0]->getLeadingDim() > 0);
    }

    ~Data() {
        for (T_iter it = _data->begin(); it != _data->end(); ++it) {
            delete *it;
        }
        delete _data;
    }
    
    T& operator [](int idx) {
        return *_data->at(idx);
    }
    
    int getSize() {
        return _data->size();
    }
    
    std::vector<T*>& getData() {
        return *_data;
    }

    int getNumCases() {
        return _data->at(0)->getLeadingDim();
    }
};

typedef Data<NVMatrix> GPUData;
typedef Data<Matrix> CPUData;

class DataProvider {
protected:
    CPUData* _hData;
    NVMatrixV _data;
    int _minibatchSize;
    long int _dataSize;
public:
    DataProvider(int minibatchSize);
    GPUData& operator[](int idx);
    void setData(CPUData&);
    void clearData();
    GPUData& getMinibatch(int idx);
    GPUData& getDataSlice(int startCase, int endCase);
    int getNumMinibatches();
    int getMinibatchSize();
    int getNumCases();
    int getNumCasesInMinibatch(int idx);
};

#endif	/* DATA_CUH */

