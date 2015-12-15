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
#include <util.cuh>
#include <worker.cuh>

using namespace std;

/* 
 * ====================
 * WorkResult
 * ====================
 */
WorkResult::WorkResult(WorkResult::RESULTS resultType, Cost& results) : _resultType(resultType), _results(&results) {
}

WorkResult::WorkResult(WorkResult::RESULTS resultType) : _resultType(resultType), _results(NULL) {
}

WorkResult::~WorkResult() {
    delete _results; // delete NULL is ok
}

Cost& WorkResult::getResults() const {
    return *_results;
}

WorkResult::RESULTS WorkResult::getResultType() const {
    return _resultType;
}

/* 
 * ====================
 * Worker
 * ====================
 */
Worker::Worker(ConvNet& convNet) : _convNet(&convNet) {
}

/* 
 * ====================
 * DataWorker
 * ====================
 */
DataWorker::DataWorker(ConvNet& convNet, CPUData& data) : Worker(convNet), _data(&data) {
    _dp = &convNet.getDataProvider();
}

DataWorker::~DataWorker() {
    _dp->clearData();
}

/* 
 * ====================
 * TrainingWorker
 * ====================
 */
TrainingWorker::TrainingWorker(ConvNet& convNet, CPUData& data, bool test) 
    : DataWorker(convNet, data), _test(test) {
}

// Need to setData here (as opposed to the constructor) because the constructor executes in
// the original CPU thread, which is not the one with GPU access.
void TrainingWorker::run() {
    _dp->setData(*_data);
    Cost& batchCost = *new Cost(0);
    for (int i = 0; i < _dp->getNumMinibatches(); i++) {
        _convNet->fprop(i, _test ? PASS_TEST : PASS_TRAIN);
        _convNet->getCost(batchCost);
        
        if (!_test) {
            _convNet->bprop(PASS_TRAIN);
            _convNet->updateWeights();
        }
    }
    cudaThreadSynchronize();
    _convNet->getResultQueue().enqueue(new WorkResult(WorkResult::BATCH_DONE, batchCost));
}

/*
 * ====================
 * SyncWorker
 * ====================
 */
SyncWorker::SyncWorker(ConvNet& convNet) : Worker(convNet) {
}

void SyncWorker::run() {
    _convNet->copyToCPU();
    _convNet->getResultQueue().enqueue(new WorkResult(WorkResult::SYNC_DONE));
}

/* 
 * ====================
 * GradCheckWorker
 * ====================
 */
GradCheckWorker::GradCheckWorker(ConvNet& convNet, CPUData& data) 
    : DataWorker(convNet, data) {
}

void GradCheckWorker::run() {
    _dp->setData(*_data);
    _convNet->checkGradients();
    exit(0);
}

/* 
 * ====================
 * MultiviewTestWorker
 * ====================
 */
MultiviewTestWorker::MultiviewTestWorker(ConvNet& convNet, CPUData& data, int numViews, int logregIdx) 
    : DataWorker(convNet, data), _numViews(numViews), _logregIdx(logregIdx) {
    assert(_data->getNumCases() % _numViews == 0);
}

void MultiviewTestWorker::run() {
    _dp->setData(*_data);
    Layer& logregLayer = _convNet->getLayer(_logregIdx);

    int numCasesReal = _dp->getNumCases() / _numViews;
    int numMiniReal = DIVUP(numCasesReal, _dp->getMinibatchSize());
    
    Cost& batchCost = *new Cost(0);
    for (int i = 0; i < numMiniReal; i++) {
        NVMatrix softmaxActs;
        for (int v = 0; v < _numViews; v++) {
            GPUData& mini = _dp->getDataSlice(v * numCasesReal + i * _dp->getMinibatchSize(),
                                              min((v + 1) * numCasesReal, v * numCasesReal + (i + 1) * _dp->getMinibatchSize()));
            _convNet->fprop(mini, PASS_TEST);
            if (v == 0) {
                logregLayer.getPrev()[1]->getActs().copy(softmaxActs);
            } else {
                softmaxActs.add(logregLayer.getPrev()[1]->getActs());
            }
        }
        softmaxActs.scale(1.0 / _numViews);
        NVMatrixV logregInput;
        logregInput.push_back(&logregLayer.getPrev()[0]->getActs());
        logregInput.push_back(&softmaxActs);
        
        logregLayer.fprop(logregInput, PASS_TEST);
        
        _convNet->getCost(batchCost);
    }
    cudaThreadSynchronize();

    _convNet->getResultQueue().enqueue(new WorkResult(WorkResult::BATCH_DONE, batchCost));
}

/* 
 * ====================
 * FeatureWorker
 * ====================
 */
FeatureWorker::FeatureWorker(ConvNet& convNet, CPUData& data, Matrix& ftrs, int layerIdx)
    : DataWorker(convNet, data), _ftrs(&ftrs), _layerIdx(layerIdx), _passType(PASS_TEST) {
    assert(ftrs.getNumRows() == data.getNumCases());
    assert(!ftrs.isTrans());
}

FeatureWorker::~FeatureWorker() {
    delete _ftrs;
}

void FeatureWorker::set_passType( int type ) {
   if( type == 0 )
      _passType = PASS_TRAIN;
   else
      _passType = PASS_TEST;
}

void FeatureWorker::run() {
    _dp->setData(*_data);
    Layer& ftrLayer = _convNet->getLayer(_layerIdx);
    Cost& batchCost = *new Cost(0);
    for (int i = 0; i < _dp->getNumMinibatches(); i++) {
        //_convNet->fprop(i, PASS_TEST);
        _convNet->fprop(i, _passType );
        _convNet->getCost(batchCost);
        Matrix& miniFtrs = _ftrs->sliceRows(i * _dp->getMinibatchSize(),
                                            min(_dp->getNumCases(), (i + 1) * _dp->getMinibatchSize()));
        NVMatrix& acts = ftrLayer.getActs();
        NVMatrix acts_T;
        if (acts.isTrans()) {
            NVMatrix& soft_T = acts.getTranspose();
            soft_T.transpose(acts_T);
            delete &soft_T;
        } else {
            acts.transpose(acts_T);
        }
        acts_T.copyToHost(miniFtrs);
        delete &miniFtrs;
    }
    cudaThreadSynchronize();
    _convNet->getResultQueue().enqueue(new WorkResult(WorkResult::BATCH_DONE, batchCost));
}

/* 
 * ======================
 * MultiviewFeatureWorker
 * ======================
 */
MultiviewFeatureWorker::MultiviewFeatureWorker( 
      ConvNet& convNet, CPUData& data, 
      Matrix& ftrs, int numViews, int logregIdx ) : 
   DataWorker( convNet, data ), 
   _ftrs(&ftrs),  _numViews( numViews ), _logregIdx( logregIdx ) 
{
    assert(_data->getNumCases() % _numViews == 0);
}

MultiviewFeatureWorker::~MultiviewFeatureWorker() {
    delete _ftrs;
}

void MultiviewFeatureWorker::run() {
   _dp->setData(*_data);
   Layer& logregLayer = _convNet->getLayer(_logregIdx);

   int numCasesReal = _dp->getNumCases() / _numViews;
   int numMiniReal = DIVUP(numCasesReal, _dp->getMinibatchSize());

   Cost& batchCost = *new Cost(0);
   for (int i = 0; i < numMiniReal; i++) {
      // get softmax acts
      NVMatrix softmaxActs;
      for (int v = 0; v < _numViews; v++) {
         GPUData& mini = _dp->getDataSlice(v * numCasesReal + i * _dp->getMinibatchSize(),
               min((v + 1) * numCasesReal, v * numCasesReal + (i + 1) * _dp->getMinibatchSize()));
         _convNet->fprop(mini, PASS_TEST);
         if (v == 0) {
            logregLayer.getPrev()[1]->getActs().copy(softmaxActs);
         } else {
            softmaxActs.add(logregLayer.getPrev()[1]->getActs());
         }
      }
      softmaxActs.scale(1.0 / _numViews);
      // transpose softmaxActs if necessary
      NVMatrix acts_T;
      NVMatrix& acts = softmaxActs;
      if (acts.isTrans()) {
         NVMatrix& soft_T = acts.getTranspose();
         soft_T.transpose(acts_T);
         delete &soft_T;
      } else {
         acts.transpose(acts_T);
      }

      // copy to ftrs slice
      Matrix& miniFtrs = _ftrs->sliceRows(i * _dp->getMinibatchSize(),
            min(numCasesReal, (i + 1) * _dp->getMinibatchSize()));
      acts_T.copyToHost( miniFtrs );
      delete &miniFtrs;
   }
   cudaThreadSynchronize();

   _convNet->getResultQueue().enqueue(new WorkResult(WorkResult::BATCH_DONE, batchCost));
}
                  
