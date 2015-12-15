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
#include <cutil_inline.h>
#include <iostream>

#include <layer_kernels.cuh>
#include <layer.cuh>
#include <data.cuh>
#include <util.cuh>
#include <cudaconv2.cuh>
#include <matrix.h>

using namespace std;

/* 
 * =======================
 * Layer
 * =======================
 */

Layer::Layer(ConvNet* convNet, PyObject* paramsDict, bool trans) : 
             _convNet(convNet),  _trans(trans) {
    _name = pyDictGetString(paramsDict, "name");
    _type = pyDictGetString(paramsDict, "type");
    
    _numGradProducersNext = 0;
    _foundGradConsumers = false;
    _gradConsumer = pyDictGetInt(paramsDict, "gradConsumer");
    _actsTarget = pyDictGetInt(paramsDict, "actsTarget");
    _actsGradTarget = pyDictGetInt(paramsDict, "actsGradTarget");
    _conserveMem = pyDictGetInt(paramsDict, "conserveMem");
    _outputs = _actsTarget < 0 ? new NVMatrix() : NULL;
    _actsGrad = _actsGradTarget < 0 ? new NVMatrix() : NULL;
}

void Layer::fpropNext(PASS_TYPE passType) {
    for (int i = 0; i < _next.size(); i++) {
        _next[i]->fprop(passType);
    }
}

void Layer::truncBwdActs() {
    // Only truncate actsGrad if I own it
    if (_conserveMem && _actsGradTarget < 0) { 
        getActsGrad().truncate();
    }
    if (_conserveMem) {
        getActs().truncate();
    }
}

void Layer::fprop(PASS_TYPE passType) {
    _rcvdFInputs += 1;
    if (_rcvdFInputs == _prev.size()) {
        NVMatrixV v;
        for (int i = 0; i < _prev.size(); i++) {
            v.push_back(&_prev[i]->getActs());
        }
        fprop(v, passType);
    }
}

void Layer::fprop(NVMatrix& v, PASS_TYPE passType) {
    NVMatrixV vl;
    vl.push_back(&v);
    fprop(vl, passType);
}

void Layer::fprop(NVMatrixV& v, PASS_TYPE passType) {
    assert(v.size() == _prev.size());
    _inputs.clear();
    _inputs.insert(_inputs.begin(), v.begin(), v.end());
    _outputs = _actsTarget < 0 ? _outputs : _inputs[_actsTarget];
    _rcvdFInputs = _prev.size();
    for (NVMatrixV::iterator it = v.begin(); it != v.end(); ++it) {
        (*it)->transpose(_trans);
    }
    getActs().transpose(_trans);
    
    // First do fprop on the input whose acts matrix I'm sharing, if any
    if (_actsTarget >= 0) {
        fpropActs(_actsTarget, 0, passType);
    }
    // Then add the rest of the inputs to that
    for (int i = 0; i < _prev.size(); i++) {
        if (i != _actsTarget) {
            fpropActs(i, _actsTarget >= 0 || i > 0, passType);
        }
    }
    fpropNext(passType);
}

void Layer::bprop(PASS_TYPE passType) {
    if (_rcvdBInputs == _numGradProducersNext) {
        _rcvdBInputs++; // avoid doing bprop computation twice
        bprop(getActsGrad(), passType);
    }
}

void Layer::bprop(NVMatrix& v, PASS_TYPE passType) {
    v.transpose(_trans);
    for (int i = 0; i < _prev.size(); i++) {
        _prev[i]->getActs().transpose(_trans);
        _prev[i]->getActsGrad().transpose(_trans);
    }
    getActs().transpose(_trans);
    
    bpropCommon(v, passType);
    
    if (isGradProducer()) {
        // First propagate activity gradient to all layers whose activity
        // gradient matrix I'm definitely not sharing.
        for (int i = 0; i < _prev.size(); i++) {
            if (_prev[i]->isGradConsumer() && _actsGradTarget != i) {
                bpropActs(v, i, _prev[i]->getRcvdBInputs() > 0 ? 1 : 0, passType);
                _prev[i]->incRcvdBInputs();
            }
        }
        // Then propagate activity gradient to the layer whose activity gradient
        // matrix I'm sharing, if any.
        if (_actsGradTarget >= 0 && _prev[_actsGradTarget]->isGradConsumer()) {
            bpropActs(v, _actsGradTarget, _prev[_actsGradTarget]->getRcvdBInputs() > 0 ? 1 : 0, passType);
            _prev[_actsGradTarget]->incRcvdBInputs();
        }
    }
    truncBwdActs();
    
    if (isGradProducer()) {
        for (int i = 0; i < _prev.size(); i++) {
            if (_prev[i]->isGradConsumer()) {
                _prev[i]->bprop(passType);
            }
        }
    }
}

void Layer::reset() {
    _rcvdFInputs = 0;
    _rcvdBInputs = 0;
}

string& Layer::getName() {
    return _name;
}

string& Layer::getType() {
    return _type;
}

int Layer::getRcvdFInputs() {
    return _rcvdFInputs;
}

int Layer::getRcvdBInputs() {
    return _rcvdBInputs;
}

int Layer::incRcvdBInputs() {
    return ++_rcvdBInputs;
}

void Layer::addNext(Layer* l) {
    _next.push_back(l);
    _numGradProducersNext += l->isGradProducer();
}

void Layer::addPrev(Layer* l) {
    _prev.push_back(l);
}

void Layer::postInit() {
//    _outputs = _actsTarget < 0 ? new NVMatrix() : &_prev[_actsTarget]->getActs();
    _actsGrad = _actsGradTarget < 0 ? new NVMatrix() : &_prev[_actsGradTarget]->getActsGrad();
}

// Does this layer, or some layer below it, need the gradient
// for parameter updates?
// Only weight layers should be grad consumers themselves.
bool Layer::isGradConsumer() {
    if (!_foundGradConsumers) {
        for (int i = 0; i < _prev.size(); i++) {
            _gradConsumer |= _prev[i]->isGradConsumer();
        }
        _foundGradConsumers = true;
    }
    return _gradConsumer;
}

// Does this layer produce gradient for layers below?
bool Layer::isGradProducer() {
    return true;
}

vector<Layer*>& Layer::getPrev() {
    return _prev;
}

vector<Layer*>& Layer::getNext() {
    return _next;
}

NVMatrix& Layer::getActs() {
    assert(_outputs != NULL);
    return *_outputs;
}

NVMatrix& Layer::getActsGrad() {
    assert(_actsGrad != NULL);
    return *_actsGrad;
}

/* 
 * =======================
 * NeuronLayer
 * =======================
 */
NeuronLayer::NeuronLayer(ConvNet* convNet, PyObject* paramsDict) 
    : Layer(convNet, paramsDict, true) {
    _neuron = &Neuron::makeNeuron(PyDict_GetItemString(paramsDict, "neuron"));
}

void NeuronLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    _neuron->computeInputGrad(v, _prev[0]->getActsGrad(), scaleTargets > 0);
}

void NeuronLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    _neuron->activate(*_inputs[0], getActs());
}

/* 
 * =======================
 * WeightLayer
 * =======================
 */
WeightLayer::WeightLayer(ConvNet* convNet, PyObject* paramsDict, bool trans, bool useGrad) : 
    Layer(convNet, paramsDict, trans) {
    
    MatrixV& hWeights = *pyDictGetMatrixV(paramsDict, "weights");
    MatrixV& hWeightsInc = *pyDictGetMatrixV(paramsDict, "weightsInc");
    Matrix& hBiases = *pyDictGetMatrix(paramsDict, "biases");
    Matrix& hBiasesInc = *pyDictGetMatrix(paramsDict, "biasesInc");
    
    floatv& momW = *pyDictGetFloatV(paramsDict, "momW");
    float momB = pyDictGetFloat(paramsDict, "momB");
    floatv& epsW = *pyDictGetFloatV(paramsDict, "epsW");
    float epsB = pyDictGetFloat(paramsDict, "epsB");
    floatv& wc = *pyDictGetFloatV(paramsDict, "wc");
    
    // Source layers for shared weights
    intv& weightSourceLayerIndices = *pyDictGetIntV(paramsDict, "weightSourceLayerIndices");
    // Weight matrix indices (inside the above source layers) for shared weights
    intv& weightSourceMatrixIndices = *pyDictGetIntV(paramsDict, "weightSourceMatrixIndices");
    
    for (int i = 0; i < weightSourceLayerIndices.size(); i++) {
        int srcLayerIdx = weightSourceLayerIndices[i];
        int matrixIdx = weightSourceMatrixIndices[i];
        if (srcLayerIdx == convNet->getNumLayers()) { // Current layer
            _weights.addWeights(*new Weights(_weights[matrixIdx], epsW[i]));
        } else if (srcLayerIdx >= 0) {
            WeightLayer& srcLayer = *static_cast<WeightLayer*>(&convNet->getLayer(srcLayerIdx));
            Weights* srcWeights = &srcLayer.getWeights(matrixIdx);
            _weights.addWeights(*new Weights(*srcWeights, epsW[i]));
        } else {
            _weights.addWeights(*new Weights(*hWeights[i], *hWeightsInc[i], epsW[i], wc[i], momW[i], useGrad));
        }
    }
    
    _biases = new Weights(hBiases, hBiasesInc, epsB, 0, momB, true);

    // Epsilons for finite-difference gradient checking operation
    _wStep = 0.001;
    _bStep = 0.002;
    
    delete &weightSourceLayerIndices;
    delete &weightSourceMatrixIndices;
    delete &hWeights;
    delete &hWeightsInc;
    delete &momW;
    delete &epsW;
    delete &wc;
}

void WeightLayer::bpropCommon(NVMatrix& v, PASS_TYPE passType) {
    if (_biases->getEps() > 0) {
        bpropBiases(v, passType);
    }
    for (int i = 0; i < _weights.getSize(); i++) {
        if (_weights[i].getEps() > 0) {
            bpropWeights(v, i, passType);
            // Increment its number of updates
            _weights[i].incNumUpdates();
        }
    }
}

void WeightLayer::updateWeights() {
    _weights.update();
    _biases->update();
}

void WeightLayer::copyToCPU() {
    _weights.copyToCPU();
    _biases->copyToCPU();
}

void WeightLayer::copyToGPU() {
    _weights.copyToGPU();
    _biases->copyToGPU();
}

void WeightLayer::checkGradients() {
    for (int i = 0; i < _weights.getSize(); i++) {
        _convNet->checkGradient(_name + " weights[" + tostr(i) + "]", _wStep, _weights[i]);
    }
    _convNet->checkGradient(_name + " biases", _bStep, *_biases);
}

Weights& WeightLayer::getWeights(int idx) {
    return _weights[idx];
}

void WeightLayer::scaleEps( float scale ) {
    for( int i = 0; i < _weights.getSize(); i++ ) {
        Weights& wi = _weights[i];
        float eps = wi.getEps();
        eps *= scale;
        wi.setEps( eps );
    }

    float eps = _biases->getEps();
    eps *= scale;
    _biases->setEps( eps );
}

void WeightLayer::resetMom() {
   for( int i = 0; i < _weights.getSize(); i++ ) {
      Weights& wi = _weights[i];
      wi.resetMom();
   }
   _biases->resetMom();
}
/* 
 * =======================
 * DataVisualizeLayer
 * =======================
 */
DataVisualizeLayer::DataVisualizeLayer(ConvNet* convNet, PyObject* paramsDict) : 
    WeightLayer(convNet, paramsDict, false, false) 
{
    _wStep = 0.1;
    _bStep = 0.01;
}

void DataVisualizeLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    throw string( "call fprop without params, this is input layer" );
}

void DataVisualizeLayer::fprop(PASS_TYPE passType) {
    // pre-condition check
    assert( _weights.getSize() == 1 );
    // copy params to output 
    getActs().add( *_weights[0] );
    fpropNext( passType );
    // no-bias term here, because weights[0] represent input images
}

void DataVisualizeLayer::fprop(NVMatrixV& data, PASS_TYPE passType) {
    throw string( "call fprop without params, this is input layer" );
}

void DataVisualizeLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, 
        PASS_TYPE passType) {
    // do nothing for bprop acts because this must be the first layer
    throw string(" bprop acts is no-define on DataViaulaizeLayer" );
}

void DataVisualizeLayer::bpropBiases(NVMatrix& v, PASS_TYPE passType) {
    // do nothing for bias update because no biase is need here
}

void DataVisualizeLayer::bpropWeights(NVMatrix& v, int inpIdx, PASS_TYPE passType) {
    assert( inpIdx == 0 );
    int numCases = v.getNumRows();
    float scaleInc = (_weights[inpIdx].getNumUpdates() == 0 && passType != PASS_GC) * _weights[inpIdx].getMom();
    float scaleGrad = passType == PASS_GC ? 1 : _weights[inpIdx].getEps() / numCases;
    _weights[inpIdx].getInc().add(v, scaleInc, scaleGrad);
}

bool DataVisualizeLayer::isGradProducer() {
    return false;
}


/* 
 * =======================
 * FCLayer
 * =======================
 */
FCLayer::FCLayer(ConvNet* convNet, PyObject* paramsDict) : WeightLayer(convNet, paramsDict, true, false) {
    _wStep = 0.1;
    _bStep = 0.01;
}
void FCLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    getActs().addProduct(*_inputs[inpIdx], *_weights[inpIdx], scaleTargets, 1);
    if (scaleTargets == 0) {
        getActs().addVector(_biases->getW());
    }
}

void FCLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    NVMatrix& weights_T = _weights[inpIdx].getW().getTranspose();
    _prev[inpIdx]->getActsGrad().addProduct(v, weights_T, scaleTargets, 1);
    delete &weights_T;
}

void FCLayer::bpropBiases(NVMatrix& v, PASS_TYPE passType) {
    int numCases = v.getNumRows();
    float scaleBGrad = passType == PASS_GC ? 1 : _biases->getEps() / numCases;
    _biases->getGrad().addSum(v, 0, 0, scaleBGrad);
}

void FCLayer::bpropWeights(NVMatrix& v, int inpIdx, PASS_TYPE passType) {
    int numCases = v.getNumRows();

    NVMatrix& prevActs_T = _prev[inpIdx]->getActs().getTranspose();
    float scaleInc = (_weights[inpIdx].getNumUpdates() == 0 && passType != PASS_GC) * _weights[inpIdx].getMom();
    float scaleGrad = passType == PASS_GC ? 1 : _weights[inpIdx].getEps() / numCases;
    
    _weights[inpIdx].getInc().addProduct(prevActs_T, v, scaleInc, scaleGrad);
    
    delete &prevActs_T;
}

/* 
 * =======================
 * FCLDropOutLayer
 * =======================
 */
FCDropLayer::FCDropLayer( ConvNet* convNet, PyObject* paramsDict) : FCLayer( 
        convNet, paramsDict ) {
    _dropRate = pyDictGetFloat( paramsDict, "rate" );
    _maxDropRate = _dropRate;
}

void FCDropLayer::set_dropRate( float dropRate ) {
   if( dropRate <= _maxDropRate ){
      _dropRate = dropRate;
      cout << "name: " << _name << " "; 
      cout << "type: " << _type << " ";
      cout << "set drop rate: " << dropRate << endl;
   }
}


/* 
 * =======================
 * FCLDropOutLayer
 * =======================
 */
FCDropOutLayer::FCDropOutLayer(ConvNet* convNet, PyObject* paramsDict) : FCDropLayer( 
        convNet, paramsDict ) {
}

void FCDropOutLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    FCLayer::fpropActs( inpIdx, scaleTargets, passType );
    // generate/apply mask
    if( passType == PASS_TEST ) {
        // test case, multiple neuron by dropout rate
        getActs().scale( 1.0f - _dropRate );
        //getActs().scale( 0.5 );
    }
    else if( passType == PASS_GC ){
        // non test case, drop out with a fixed rate
        // fix mask for debugging
        if( _mask.getNumRows() == 0 || _mask.getNumCols() == 0 ){
            _mask.resize( getActs() );
            _mask.randomizeUniform();
            _mask.biggerThanScalar( _dropRate );
        }
        getActs().eltwiseMult( _mask );
    }
    else{ 
        _mask.resize( getActs() );
        _mask.randomizeUniform();
        _mask.biggerThanScalar( _dropRate );
        getActs().eltwiseMult( _mask );
    }

}

void FCDropOutLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    v.eltwiseMult( _mask );
    FCLayer::bpropActs( v, inpIdx, scaleTargets, passType );
}
void FCDropOutLayer::bpropBiases(NVMatrix& v, PASS_TYPE passType) {
    v.eltwiseMult( _mask );
    FCLayer::bpropBiases( v, passType );
}

void FCDropOutLayer::bpropWeights(NVMatrix& v, int inpIdx, PASS_TYPE passType) {
    v.eltwiseMult( _mask );
    //NVMatrix v2( v );
    //v.eltwiseMult( _mask, v2 );
    FCLayer::bpropWeights( v, inpIdx, passType );
}

//--------------------------------------------------
// old implementation: approximate drop connection
// modify date: Dec28-2012
//--------------------------------------------------
/* 
 * =======================
 * FCLDropConnectApproxLayer
 * =======================
 */
FCDropConnectApproxLayer::FCDropConnectApproxLayer(
      ConvNet* convNet, PyObject* paramsDict) : FCDropLayer( 
        convNet, paramsDict ) {
}

void FCDropConnectApproxLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
   assert( inpIdx == 0 );
   _weights[0].getW().copy( _maskWeight );
   _biases->getW().copy( _maskBias );
   // generate _mask matrix
   if( passType == PASS_TEST ) {
      // test case, scale test weight
      _maskWeight.scale( 1.0f - _dropRate );
      _maskBias.scale( 1.0f - _dropRate );
   }
   else if( passType == PASS_GC ){
      // fix mask for debugging
      // weights mask
      if( _mask.getNumRows() == 0 || _mask.getNumCols() == 0 ){
         _mask.resize( _maskWeight );
         _mask.randomizeUniform();
         _mask.biggerThanScalar( _dropRate );
      }
      _maskWeight.eltwiseMult( _mask );
      // bias mask
      if( _mask2.getNumRows() == 0 || _mask2.getNumCols() == 0 ) {
          _mask2.resize( _maskBias );
          _mask2.randomizeUniform();
          _mask2.biggerThanScalar( _dropRate );
      }
      _maskBias.eltwiseMult( _mask2 );
   }
   else{ 
       // weights mask
      _mask.resize( _maskWeight );
      _mask.randomizeUniform();
      _mask.biggerThanScalar( _dropRate );
      _maskWeight.eltwiseMult( _mask );
      // bias mask
      _mask2.resize( _maskBias );
      _mask2.randomizeUniform();
      _mask2.biggerThanScalar( _dropRate );
      _maskBias.eltwiseMult( _mask2 );
   }

   // compute outputs
   getActs().addProduct(*_inputs[inpIdx], _maskWeight, scaleTargets, 1);
   if (scaleTargets == 0) {
      //getActs().addVector(_biases->getW());
      getActs().addVector( _maskBias );
   }
}

void FCDropConnectApproxLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, 
      PASS_TYPE passType) {

    NVMatrix& weights_T = _maskWeight.getTranspose();
    _prev[inpIdx]->getActsGrad().addProduct(v, weights_T, scaleTargets, 1);
    delete &weights_T;
}

void FCDropConnectApproxLayer::bpropBiases(NVMatrix& v, PASS_TYPE passType) {
   //FCDropLayer::bpropBiases( v, passType );
   int numCases = v.getNumRows();
   float scaleBGrad = passType == PASS_GC ? 1 : _biases->getEps() / numCases;
   _biases->getGrad().addSum(v, 0, 0, scaleBGrad);

   // mask out invlaid update
   _biases->getGrad().eltwiseMult( _mask2 );
}

void FCDropConnectApproxLayer::bpropWeights(NVMatrix& v, int inpIdx, PASS_TYPE passType) {
    int numCases = v.getNumRows();

    NVMatrix& prevActs_T = _prev[inpIdx]->getActs().getTranspose();
    float scaleInc = (_weights[inpIdx].getNumUpdates() == 0 && passType != PASS_GC) * _weights[inpIdx].getMom();
    float scaleGrad = passType == PASS_GC ? 1 : _weights[inpIdx].getEps() / numCases;
    
    _weights[inpIdx].getInc().addProduct(prevActs_T, v, scaleInc, scaleGrad);

    // mask out invalid update
    _weights[inpIdx].getInc().eltwiseMult( _mask );
    
    delete &prevActs_T;
}

/* 
 * =======================
 * FCLDropConnectLayer
 * =======================
 */
FCDropConnectLayer::FCDropConnectLayer(ConvNet* convNet, PyObject* paramsDict) : FCDropLayer( 
        convNet, paramsDict ) {
}

void FCDropConnectLayer::mallocRandomMask( int m, int n, int d, PASS_TYPE passType ) {
   assert( passType == PASS_GC || passType == PASS_TRAIN );
   NVMatrix& _maskWeights = _mask;  // alians for _mask
   if( passType == PASS_GC ){
      // fix mask for debugging
      // weights mask
      if( _maskWeights.getNumRows() == 0 || _maskWeights.getNumCols() == 0 ){
         _maskWeights.setTrans( true ); // col major matrix
         _maskWeights.resize( n, m*d );
         _maskWeights.randomizeUniform();
         _maskWeights.biggerThanScalar( _dropRate );
      }
      // bias mask
      if( _maskBiases.getNumRows() == 0 || _maskBiases.getNumCols() == 0 ) {
         _maskBiases.setTrans( true ); // row major matrix
         _maskBiases.resize( d, m );
         _maskBiases.randomizeUniform();
         _maskBiases.biggerThanScalar( _dropRate );
      }
   }
   else { // passType == PASS_TRAIN
      // int prev_d = _maskWeights.getNumCols()/m; // num of col = m*d for maskWeight
      // TODO: provide optimization when prev_d > d, no need to alloc again
      _maskWeights.setTrans( true ); // col major matrix
      _maskWeights.resize( n, m*d );
      _maskWeights.randomizeUniform();
      _maskWeights.biggerThanScalar( _dropRate );

      _maskBiases.setTrans( true ); // row major matrix
      _maskBiases.resize( d, m );
      _maskBiases.randomizeUniform();
      _maskBiases.biggerThanScalar( _dropRate );
   }
}

void FCDropConnectLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
   // current implementation only has one input layer before this
   assert( inpIdx == 0 ); 
   NVMatrix& w = _weights[0].getW();
   NVMatrix& b = _biases->getW();
   NVMatrix& x = *_inputs[inpIdx];
   NVMatrix& y = getActs();
   int m = w.getNumCols();  // output dimension
   int n = x.getNumCols();  // input dimension
   int d = x.getNumRows();  // number of data in this bacth
   assert( n == w.getNumRows() );
   assert( m == b.getNumCols() );

   // easy form for inference: only take mean of connection
   if( passType == PASS_TEST ) {
      NVMatrix tempMaskWeights;
      NVMatrix tempMaskBiases;
      _weights[0].getW().copy( tempMaskWeights );
      _biases->getW().copy( tempMaskBiases );
      // test case, scale test weight
      tempMaskWeights.scale( 1.0f - _dropRate );
      tempMaskBiases.scale( 1.0f - _dropRate );
      // compute outputs
      getActs().addProduct( x, tempMaskWeights, scaleTargets, 1);
      if (scaleTargets == 0) {
         //getActs().addVector(_biases->getW());
         getActs().addVector( tempMaskBiases );
      }
      return;
   }
   mallocRandomMask( m, n, d, passType );
   NVMatrix& _maskWeights = _mask;  // alians for _mask
   y.resize( d, m );
   y.setTrans( true );
   y.apply( NVMatrixOps::Zero() );
   computeFCDropC_fprop( x, w, b, _maskWeights, _maskBiases, y );
}

void FCDropConnectLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, 
      PASS_TYPE passType) {
    //NVMatrix& weights_T = _maskWeight.getTranspose();
    //_prev[inpIdx]->getActsGrad().addProduct(v, weights_T, scaleTargets, 1);
    //delete &weights_T;
   assert( inpIdx == 0);
   NVMatrix& w = _weights[0].getW();
   int m = w.getNumCols();  // output dimension
   int n = w.getNumRows();
   int d = v.getNumRows();
   NVMatrix& da = _prev[inpIdx]->getActsGrad();
   NVMatrix& _maskWeights = _mask;  // alians for _mask
   da.resize( d, n );
   da.setTrans( true );
   da.apply( NVMatrixOps::Zero() );
   computeFCDropC_bpropActs( v, w, 1 , _maskWeights, da, scaleTargets );
}

void FCDropConnectLayer::bpropBiases(NVMatrix& v, PASS_TYPE passType) {
   int numCases = v.getNumRows();
   float scaleBGrad = passType == PASS_GC ? 1 : _biases->getEps() / numCases;
   NVMatrix p;
   v.eltwiseMult( _maskBiases, p );
   _biases->getGrad().addSum(p, 0, 0, scaleBGrad);
}

void FCDropConnectLayer::bpropWeights(NVMatrix& v, int inpIdx, PASS_TYPE passType) {
    int numCases = v.getNumRows();
    float scaleInc = (_weights[inpIdx].getNumUpdates() == 0 && passType != PASS_GC) * _weights[inpIdx].getMom();
    float scaleGrad = passType == PASS_GC ? 1 : _weights[inpIdx].getEps() / numCases;
    
    //NVMatrix& prevActs_T = _prev[inpIdx]->getActs().getTranspose();
    //_weights[inpIdx].getInc().addProduct(prevActs_T, v, scaleInc, scaleGrad);

    //// mask out invalid update
    //_weights[inpIdx].getInc().eltwiseMult( _mask );
    //
    //delete &prevActs_T;
    NVMatrix& a = _prev[inpIdx]->getActs();
    NVMatrix& dw = _weights[inpIdx].getInc();
    NVMatrix& _maskWeights = _mask;  // alians for _mask
    computeFCDropC_bpropWeights(
          a, v, scaleGrad, _maskWeights,
          dw, scaleInc );
}

/* 
 * =======================
 * FCLDropConnectBitLayer
 * =======================
 */
FCDropConnectBitLayer::FCDropConnectBitLayer(ConvNet* convNet, PyObject* paramsDict) : 
   FCDropLayer( convNet, paramsDict ), _mcInference(false), _numSamples(0) {
       _maskWeights.set_onProb( 1 - _dropRate );
}

void FCDropConnectBitLayer::set_dropRate( float dropRate ) {
    FCDropLayer::set_dropRate( dropRate );
    // MaskWeights  use onProb rather than off Prob
   if( dropRate <= _maxDropRate ){
        _maskWeights.set_onProb( 1 - dropRate );
   }


}

void FCDropConnectBitLayer::enableMCInference( int numSamples ) {
    _mcInference = true;
    assert( numSamples > 0 );
    _numSamples = numSamples;
}

void FCDropConnectBitLayer::mallocRandomMask( int m, int n, int d, PASS_TYPE passType ) {
   assert( passType == PASS_GC || passType == PASS_TRAIN );
   if( passType == PASS_GC ){
      // fix mask for debugging
      // weights mask
      if( _maskWeights.get_width() == 0 || _maskWeights.get_height() == 0 ) {
         _maskWeights.resize( m, n, d );
         _maskWeights.randomize();
      }
      // bias mask
      if( _maskBiases.getNumRows() == 0 || _maskBiases.getNumCols() == 0 ) {
         _maskBiases.setTrans( true ); // row major matrix
         _maskBiases.resize( d, m );
         _maskBiases.randomizeUniform();
         _maskBiases.biggerThanScalar( _dropRate );
      }
   }
   else { // passType == PASS_TRAIN
      _maskWeights.resize( m, n, d );
      _maskWeights.randomize();

      _maskBiases.setTrans( true ); // row major matrix
      _maskBiases.resize( d, m );
      _maskBiases.randomizeUniform();
      _maskBiases.biggerThanScalar( _dropRate );
   }
}

void FCDropConnectBitLayer::inference( int inpIdx, float scaleTargets) {
    // get dimenison info
    assert( inpIdx == 0 ); 
    NVMatrix& w = _weights[0].getW();
    NVMatrix& b = _biases->getW();
    NVMatrix& x = *_inputs[inpIdx];
    NVMatrix& y = getActs();
    int m = w.getNumCols();  // output dimension
    int n = x.getNumCols();  // input dimension
    int d = x.getNumRows();  // number of data in this bacth
    float p = 1.0f - _dropRate;
    assert( n == w.getNumRows() );
    assert( m == b.getNumCols() );
    assert( scaleTargets == 0 );
    // inference
    if( _mcInference ) {
        NVMatrix mu( d, n, true );
        NVMatrix var( d, n, true );
        // compute mean
        mu.addProduct( x, w, 0, 1 );
        mu.addVector( b );
        mu.scale( p );
        // compute var
        NVMatrix w2;
        w.copy(w2);
        w2.eltwiseMult( w );

        NVMatrix x2;
        x.copy(x2);
        x2.eltwiseMult( x );
        var.addProduct( x2, w2, 0, p*(1-p) );
        // init y
        y.resize( d, m );
        y.setTrans( true );
        y.apply( NVMatrixOps::Zero() );
        // call inference kernel
        computeFCDropC_bit_inference( mu, var, _numSamples, y );
    }
    else { // E[F(x)] = F(E[X])
        // get weights and biases
        NVMatrix tempMaskWeights;
        NVMatrix tempMaskBiases;
        _weights[0].getW().copy( tempMaskWeights );
        _biases->getW().copy( tempMaskBiases );
        // test case, scale test weight
        tempMaskWeights.scale( 1.0f - _dropRate );
        tempMaskBiases.scale( 1.0f - _dropRate );
        // compute outputs
        getActs().addProduct( x, tempMaskWeights, scaleTargets, 1);
        if (scaleTargets == 0) {
            //getActs().addVector(_biases->getW());
            getActs().addVector( tempMaskBiases );
        }
    }
}

void FCDropConnectBitLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
   // current implementation only has one input layer before this
   assert( inpIdx == 0 ); 
   NVMatrix& w = _weights[0].getW();
   NVMatrix& b = _biases->getW();
   NVMatrix& x = *_inputs[inpIdx];
   NVMatrix& y = getActs();
   int m = w.getNumCols();  // output dimension
   int n = x.getNumCols();  // input dimension
   int d = x.getNumRows();  // number of data in this bacth
   assert( n == w.getNumRows() );
   assert( m == b.getNumCols() );

   // easy form for inference: only take mean of connection
   if( passType == PASS_TEST ) {
       inference( inpIdx, scaleTargets );
       return;
   }
   mallocRandomMask( m, n, d, passType );
   //NVMatrix& _maskWeights = _mask;  // alians for _mask
   y.resize( d, m );
   y.setTrans( true );
   y.apply( NVMatrixOps::Zero() );
   computeFCDropC_bit_fprop( x, w, b, _maskWeights, _maskBiases, y );
}

void FCDropConnectBitLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, 
      PASS_TYPE passType) {
    //NVMatrix& weights_T = _maskWeight.getTranspose();
    //_prev[inpIdx]->getActsGrad().addProduct(v, weights_T, scaleTargets, 1);
    //delete &weights_T;
   assert( inpIdx == 0);
   NVMatrix& w = _weights[0].getW();
   int m = w.getNumCols();  // output dimension
   int n = w.getNumRows();
   int d = v.getNumRows();
   NVMatrix& da = _prev[inpIdx]->getActsGrad();
   //NVMatrix& _maskWeights = _mask;  // alians for _mask
   da.resize( d, n );
   da.setTrans( true );
   da.apply( NVMatrixOps::Zero() );
   computeFCDropC_bit_bpropActs( v, w, 1 , _maskWeights, da, scaleTargets );
}

void FCDropConnectBitLayer::bpropBiases(NVMatrix& v, PASS_TYPE passType) {
   int numCases = v.getNumRows();
   float scaleBGrad = passType == PASS_GC ? 1 : _biases->getEps() / numCases;
   NVMatrix p;
   v.eltwiseMult( _maskBiases, p );
   _biases->getGrad().addSum(p, 0, 0, scaleBGrad);
}

void FCDropConnectBitLayer::bpropWeights(NVMatrix& v, int inpIdx, PASS_TYPE passType) {
    int numCases = v.getNumRows();
    float scaleInc = (_weights[inpIdx].getNumUpdates() == 0 && passType != PASS_GC) * _weights[inpIdx].getMom();
    float scaleGrad = passType == PASS_GC ? 1 : _weights[inpIdx].getEps() / numCases;
    
    //NVMatrix& prevActs_T = _prev[inpIdx]->getActs().getTranspose();
    //_weights[inpIdx].getInc().addProduct(prevActs_T, v, scaleInc, scaleGrad);

    //// mask out invalid update
    //_weights[inpIdx].getInc().eltwiseMult( _mask );
    //
    //delete &prevActs_T;
    NVMatrix& a = _prev[inpIdx]->getActs();
    NVMatrix& dw = _weights[inpIdx].getInc();
    //NVMatrix& _maskWeights = _mask;  // alians for _mask
    computeFCDropC_bit_bpropWeights(
          a, v, scaleGrad, _maskWeights,
          dw, scaleInc );
}

/* 
 * =======================
 * LocalLayer
 * =======================
 */
LocalLayer::LocalLayer(ConvNet* convNet, PyObject* paramsDict, bool useGrad) 
    : WeightLayer(convNet, paramsDict, false, useGrad) {
    _padding = pyDictGetIntV(paramsDict, "padding");
    _stride = pyDictGetIntV(paramsDict, "stride");
    _filterSize = pyDictGetIntV(paramsDict, "filterSize");
    _channels = pyDictGetIntV(paramsDict, "channels");
    _imgSize = pyDictGetIntV(paramsDict, "imgSize");
    _numFilters = pyDictGetInt(paramsDict, "filters");
    _groups = pyDictGetIntV(paramsDict, "groups");
    _filterChannels = pyDictGetIntV(paramsDict, "filterChannels");
    _randSparse = pyDictGetIntV(paramsDict, "randSparse");
    _overSample = pyDictGetIntV(paramsDict, "overSample");
    _filterPixels = pyDictGetIntV(paramsDict, "filterPixels");
    _imgPixels = pyDictGetIntV(paramsDict, "imgPixels");
    
    _modulesX = pyDictGetInt(paramsDict, "modulesX");
    _modules = pyDictGetInt(paramsDict, "modules");

    // It's a vector on the heap to be consistent with all the others...
    _filterConns = new vector<FilterConns>();
    PyObject* pyFilterConns = PyDict_GetItemString(paramsDict, "filterConns");
    for (int i = 0; i < _randSparse->size(); i++) {
        FilterConns fc;
        if (_randSparse->at(i)) {
            fc.hFilterConns = getIntA(PyList_GET_ITEM(pyFilterConns, i));
        }
        _filterConns->push_back(fc);
    }
}

void LocalLayer::copyToGPU() {
    WeightLayer::copyToGPU();
    for  (int i = 0; i < _prev.size(); i++) {
        if (_randSparse->at(i)) { // Copy to GPU vector that describes sparse random connectivity
            cudaMalloc(&_filterConns->at(i).dFilterConns, sizeof(int) * _groups->at(i) * _filterChannels->at(i));
            cudaMemcpy(_filterConns->at(i).dFilterConns, _filterConns->at(i).hFilterConns,
                       sizeof(int) * _groups->at(i) * _filterChannels->at(i), cudaMemcpyHostToDevice);
            cutilCheckMsg("cudaMemcpy: failed");
        }
    }
}

/* 
 * =======================
 * ConvLayer
 * =======================
 */
ConvLayer::ConvLayer(ConvNet* convNet, PyObject* paramsDict) : LocalLayer(convNet, paramsDict, true) {
    _partialSum = pyDictGetInt(paramsDict, "partialSum");
    _sharedBiases = pyDictGetInt(paramsDict, "sharedBiases");
}

void ConvLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    if (_randSparse->at(inpIdx)) {
        convFilterActsSparse(*_inputs[inpIdx], *_weights[inpIdx], getActs(), _filterConns->at(inpIdx).dFilterConns,
                             _imgSize->at(inpIdx), _modulesX, _modulesX, _padding->at(inpIdx), _stride->at(inpIdx), _channels->at(inpIdx),
                             _filterChannels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);
    } else {
        convFilterActs(*_inputs[inpIdx], *_weights[inpIdx], getActs(), _imgSize->at(inpIdx), _modulesX, _modulesX, _padding->at(inpIdx),
                       _stride->at(inpIdx), _channels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);
    }
    
    if (scaleTargets == 0) {
        if (_sharedBiases) {
            getActs().reshape(_numFilters, getActs().getNumElements() / _numFilters);
            getActs().addVector(_biases->getW());
            getActs().reshape(_numFilters * _modules, getActs().getNumElements() / (_numFilters * _modules));
        } else {
            getActs().addVector(_biases->getW());
        }
    }
}

void ConvLayer::bpropBiases(NVMatrix& v, PASS_TYPE passType) {
    int numCases = v.getNumCols();
    float scaleBGrad = passType == PASS_GC ? 1 : _biases->getEps() / numCases;
    if (_sharedBiases) {
        v.reshape(_numFilters, v.getNumElements() / _numFilters);
        _biases->getGrad().addSum(v, 1, 0, scaleBGrad);
        v.reshape(_numFilters * _modules, v.getNumElements() / (_numFilters * _modules));
    } else {
        _biases->getGrad().addSum(v, 1, 0, scaleBGrad);
    }
}

void ConvLayer::bpropWeights(NVMatrix& v, int inpIdx, PASS_TYPE passType) {
    int numCases = v.getNumCols();

    NVMatrix& tgt = _partialSum > 0 ? _weightGradTmp : _weights[inpIdx].getGrad();
    float scaleWGrad = passType == PASS_GC ? 1 : _weights[inpIdx].getEps() / numCases;
    float scaleTargets = _weights[inpIdx].getNumUpdates() > 0 && _partialSum == 0; // ? 1 : 0;
    if (_randSparse->at(inpIdx)) {
        convWeightActsSparse(_prev[inpIdx]->getActs(), v, tgt, _filterConns->at(inpIdx).dFilterConns, _imgSize->at(inpIdx), _modulesX, _modulesX,
                             _filterSize->at(inpIdx), _padding->at(inpIdx), _stride->at(inpIdx), _channels->at(inpIdx),
                             _filterChannels->at(inpIdx), _groups->at(inpIdx), _partialSum, scaleTargets, scaleWGrad);
    } else {
        convWeightActs(_prev[inpIdx]->getActs(), v, tgt, _imgSize->at(inpIdx), _modulesX, _modulesX, _filterSize->at(inpIdx), _padding->at(inpIdx),
                       _stride->at(inpIdx), _channels->at(inpIdx), _groups->at(inpIdx), _partialSum, scaleTargets, scaleWGrad);
    }
    if (_partialSum > 0) {
        scaleTargets = _weights[inpIdx].getNumUpdates() > 0;
        _weightGradTmp.reshape(_modules / _partialSum, _filterChannels->at(inpIdx) * _filterPixels->at(inpIdx) * _numFilters);
        _weights[inpIdx].getGrad().addSum(_weightGradTmp, 0, scaleTargets, 1);
        _weights[inpIdx].getGrad().reshape(_filterChannels->at(inpIdx) * _filterPixels->at(inpIdx), _numFilters);
    }
}

void ConvLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    if (_randSparse->at(inpIdx)) {
        NVMatrix& tgt = _overSample->at(inpIdx) > 1 ? _actGradTmp : _prev[inpIdx]->getActsGrad();
        convImgActsSparse(v, *_weights[inpIdx], tgt, _filterConns->at(inpIdx).dFilterConns,
                          _imgSize->at(inpIdx), _imgSize->at(inpIdx), _modulesX, _padding->at(inpIdx), _stride->at(inpIdx),
                          _channels->at(inpIdx), _filterChannels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);
        if (_overSample->at(inpIdx) > 1) {
            _actGradTmp.reshape(_overSample->at(inpIdx), _actGradTmp.getNumElements() / _overSample->at(inpIdx));
            _actGradTmp.sum(0, _prev[inpIdx]->getActsGrad());
            _prev[inpIdx]->getActsGrad().reshape(_prev[inpIdx]->getActsGrad().getNumElements() / v.getNumCols(), v.getNumCols());
        }
    } else {
        convImgActs(v, *_weights[inpIdx], _prev[inpIdx]->getActsGrad(), _imgSize->at(inpIdx), _imgSize->at(inpIdx), _modulesX,
                    _padding->at(inpIdx), _stride->at(inpIdx), _channels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);
    }
}

void ConvLayer::truncBwdActs() {
    LocalLayer::truncBwdActs();
    if (_conserveMem) {
        _weightGradTmp.truncate();
        _actGradTmp.truncate();
    }
}
/* 
 * =======================
 * LocalUnsharedLayer
 * =======================
 */
LocalUnsharedLayer::LocalUnsharedLayer(ConvNet* convNet, PyObject* paramsDict) : LocalLayer(convNet, paramsDict, false) {
}

void LocalUnsharedLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    if (_randSparse->at(inpIdx)) {
        localFilterActsSparse(*_inputs[inpIdx], *_weights[inpIdx], getActs(), _filterConns->at(inpIdx).dFilterConns,
                              _imgSize->at(inpIdx), _modulesX, _modulesX, _padding->at(inpIdx), _stride->at(inpIdx), _channels->at(inpIdx),
                              _filterChannels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);
    } else {
        localFilterActs(*_inputs[inpIdx], *_weights[inpIdx], getActs(), _imgSize->at(inpIdx), _modulesX, _modulesX, _padding->at(inpIdx),
                        _stride->at(inpIdx), _channels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);

    }  
    if (scaleTargets == 0) {
        getActs().addVector(_biases->getW());
    }
}

void LocalUnsharedLayer::bpropBiases(NVMatrix& v, PASS_TYPE passType) {
    int numCases = v.getNumCols();
    float scaleBGrad = passType == PASS_GC ? 1 : _biases->getEps() / numCases;
    _biases->getGrad().addSum(v, 1, 0, scaleBGrad);
}

void LocalUnsharedLayer::bpropWeights(NVMatrix& v, int inpIdx, PASS_TYPE passType) {
    int numCases = v.getNumCols();
    
    float scaleInc = (passType != PASS_GC && _weights[inpIdx].getNumUpdates() == 0) * _weights[inpIdx].getMom(); // momentum
    float scaleWGrad = passType == PASS_GC ? 1 : _weights[inpIdx].getEps() / numCases; // eps / numCases
    if (_randSparse->at(inpIdx)) {
        localWeightActsSparse(_prev[inpIdx]->getActs(), v, _weights[inpIdx].getInc(), _filterConns->at(inpIdx).dFilterConns,
                              _imgSize->at(inpIdx), _modulesX, _modulesX, _filterSize->at(inpIdx), _padding->at(inpIdx), _stride->at(inpIdx),
                              _channels->at(inpIdx), _filterChannels->at(inpIdx), _groups->at(inpIdx), scaleInc, scaleWGrad);
    } else {
        localWeightActs(_prev[inpIdx]->getActs(), v, _weights[inpIdx].getInc(), _imgSize->at(inpIdx), _modulesX, _modulesX, _filterSize->at(inpIdx),
                        _padding->at(inpIdx), _stride->at(inpIdx), _channels->at(inpIdx), _groups->at(inpIdx), scaleInc, scaleWGrad);
    }
}

void LocalUnsharedLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    if (_randSparse->at(inpIdx)) {
        localImgActsSparse(v, *_weights[inpIdx], _prev[inpIdx]->getActsGrad(), _filterConns->at(inpIdx).dFilterConns,
                           _imgSize->at(inpIdx), _imgSize->at(inpIdx), _modulesX, _padding->at(inpIdx), _stride->at(inpIdx), _channels->at(inpIdx),
                           _filterChannels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);
    } else {
        localImgActs(v, *_weights[inpIdx], _prev[inpIdx]->getActsGrad(),_imgSize->at(inpIdx), _imgSize->at(inpIdx), _modulesX,
                    _padding->at(inpIdx),  _stride->at(inpIdx), _channels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);
    }
}

/* 
 * =======================
 * SoftmaxLayer
 * =======================
 */
SoftmaxLayer::SoftmaxLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, true) {
}

void SoftmaxLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    NVMatrix& input = *_inputs[0];
    NVMatrix& max = input.max(1);
    input.addVector(max, -1, getActs());
    getActs().apply(NVMatrixOps::Exp());
    NVMatrix& sum = getActs().sum(1);
    getActs().eltwiseDivideByVector(sum);
    
    delete &max;
    delete &sum;
}

void SoftmaxLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    assert(inpIdx == 0);
    bool doLogregGrad = _next.size() == 1 && _next[0]->getType() == "cost.logreg";
    if (doLogregGrad) {
        NVMatrix& labels = _next[0]->getPrev()[0]->getActs();
        float gradCoeff = dynamic_cast<CostLayer*>(_next[0])->getCoeff();
        computeLogregSoftmaxGrad(labels, getActs(), _prev[0]->getActsGrad(), scaleTargets == 1, gradCoeff);
    } else {
        computeSoftmaxGrad(getActs(), v, _prev[0]->getActsGrad(), scaleTargets == 1);
    }
}

/* 
 * =======================
 * EltwiseSumLayer
 * =======================
 */
EltwiseSumLayer::EltwiseSumLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {
    _coeffs = pyDictGetFloatV(paramsDict, "coeffs");
}

void EltwiseSumLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    if (scaleTargets == 0) {
        _inputs[inpIdx]->scale(_coeffs->at(inpIdx), getActs());
    } else {
        getActs().add(*_inputs[inpIdx], _coeffs->at(inpIdx));
    }
}

void EltwiseSumLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    if (scaleTargets == 0 ) {
        v.scale(_coeffs->at(inpIdx), _prev[inpIdx]->getActsGrad());
    } else {
        assert(&_prev[inpIdx]->getActsGrad() != &v);
        _prev[inpIdx]->getActsGrad().add(v, scaleTargets, _coeffs->at(inpIdx));
    }
}

/* 
 * =======================
 * EltwiseMaxLayer
 * =======================
 */
EltwiseMaxLayer::EltwiseMaxLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {
}

void EltwiseMaxLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    if (inpIdx == 1) { // First input, do nothing
        _inputs[inpIdx]->applyBinary(NVMatrixAggs::Max(), *_inputs[0], getActs());
    } else if (inpIdx > 1) {
        getActs().applyBinary(NVMatrixAggs::Max(), *_inputs[inpIdx]);
    }
}

void EltwiseMaxLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    computeEltwiseMaxGrad(v, *_inputs[inpIdx], getActs(), _prev[inpIdx]->getActsGrad(), scaleTargets != 0);
}

/* 
 * =======================
 * DataLayer
 * =======================
 */
DataLayer::DataLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {
    _dataIdx = pyDictGetInt(paramsDict, "dataIdx");
}

void DataLayer::fprop(PASS_TYPE passType) {
    throw string("No dava given!");
}

void DataLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
}

void DataLayer::fprop(NVMatrixV& data, PASS_TYPE passType) {
    _outputs = data[_dataIdx];
    fpropNext(passType);
}

bool DataLayer::isGradProducer() {
    return false;
}

/* 
 * =====================
 * PoolLayer
 * =====================
 */
PoolLayer::PoolLayer(ConvNet* convNet, PyObject* paramsDict, bool trans) 
    : Layer(convNet, paramsDict, trans) {
    _channels = pyDictGetInt(paramsDict, "channels");
    _sizeX = pyDictGetInt(paramsDict, "sizeX");
    _start = pyDictGetInt(paramsDict, "start");
    _stride = pyDictGetInt(paramsDict, "stride");
    _outputsX = pyDictGetInt(paramsDict, "outputsX");
    _imgSize = pyDictGetInt(paramsDict, "imgSize");
    _pool = pyDictGetString(paramsDict, "pool");
}

PoolLayer& PoolLayer::makePoolLayer(ConvNet* convNet, PyObject* paramsDict) {
    string _pool = pyDictGetString(paramsDict, "pool");
    if (_pool == "max") {
        return *new MaxPoolLayer(convNet, paramsDict);
    } else if(_pool == "avg") {
        return *new AvgPoolLayer(convNet, paramsDict);
    }
    throw string("Unknown pooling layer type ") + _pool;
}

/* 
 * =====================
 * AvgPoolLayer
 * =====================
 */
AvgPoolLayer::AvgPoolLayer(ConvNet* convNet, PyObject* paramsDict) : PoolLayer(convNet, paramsDict, false) {
}

void AvgPoolLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convLocalPool(*_inputs[0], getActs(), _channels, _sizeX, _start, _stride, _outputsX, AvgPooler());
}

void AvgPoolLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convLocalAvgUndo(v, _prev[0]->getActsGrad(), _sizeX, _start, _stride, _outputsX, _imgSize, scaleTargets, 1);
}

/* 
 * =====================
 * MaxPoolLayer
 * =====================
 */
MaxPoolLayer::MaxPoolLayer(ConvNet* convNet, PyObject* paramsDict) : PoolLayer(convNet, paramsDict, false) {
}

void MaxPoolLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convLocalPool(*_inputs[0], getActs(), _channels, _sizeX, _start, _stride, _outputsX, MaxPooler());
}

void MaxPoolLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convLocalMaxUndo(_prev[0]->getActs(), v, getActs(), _prev[inpIdx]->getActsGrad(), _sizeX, _start, _stride, _outputsX, scaleTargets, 1);
}

/* 
 * =====================
 * NailbedLayer
 * =====================
 */
NailbedLayer::NailbedLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {
    _channels = pyDictGetInt(paramsDict, "channels");
    _start = pyDictGetInt(paramsDict, "start");
    _stride = pyDictGetInt(paramsDict, "stride");
    _outputsX = pyDictGetInt(paramsDict, "outputsX");
    _imgSize = pyDictGetInt(paramsDict, "imgSize");
}

void NailbedLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convBedOfNails(*_inputs[0], getActs(), _channels, _imgSize, _start, _stride, 0, 1);
}

void NailbedLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convBedOfNailsUndo(v, _prev[0]->getActsGrad(), _channels, _imgSize, _start, _stride, scaleTargets, 1);
}

/* 
 * =====================
 * GaussianBlurLayer
 * =====================
 */
GaussianBlurLayer::GaussianBlurLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {
    _channels = pyDictGetInt(paramsDict, "channels");
    _hFilter = pyDictGetMatrix(paramsDict, "filter");
}

void GaussianBlurLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convGaussianBlur(*_inputs[0], _filter, getActs(), true, _channels, 0, 1);
    convGaussianBlur(getActs(), _filter, getActs(), false, _channels, 0, 1);
}

// This is here just for completeness' sake. Why would you backpropagate
// through a blur filter?
void GaussianBlurLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    NVMatrix& tgt1 = _prev[0]->getRcvdBInputs() > 0 ? _actGradsTmp : _prev[0]->getActsGrad();
    convGaussianBlur(v, _filter, tgt1, true, _channels, 0, 1);
    convGaussianBlur(tgt1, _filter, _prev[0]->getActsGrad(), false, _channels, scaleTargets, 1);
}

void GaussianBlurLayer::copyToGPU() {
    _filter.copyFromHost(*_hFilter, true);
}

/* 
 * =====================
 * ResizeLayer
 * =====================
 */
ResizeLayer::ResizeLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {
    _channels = pyDictGetInt(paramsDict, "channels");
    _imgSize = pyDictGetInt(paramsDict, "imgSize");
    _tgtSize = pyDictGetInt(paramsDict, "tgtSize");
    _scale = pyDictGetFloat(paramsDict, "scale");
}

void ResizeLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convResizeBilinear(*_inputs[0], getActs(), _imgSize, _tgtSize, _scale);
}

// Can't do this
void ResizeLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    assert(false);
}

/* 
 * =====================
 * RGBToYUVLayer
 * =====================
 */
RGBToYUVLayer::RGBToYUVLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {
}

void RGBToYUVLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convRGBToYUV(*_inputs[0], getActs());
}

// Can't do this
void RGBToYUVLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    assert(false);
}

/* 
 * =====================
 * RGBToLABLayer
 * =====================
 */
RGBToLABLayer::RGBToLABLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {
    _center = pyDictGetInt(paramsDict, "center");
}

void RGBToLABLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convRGBToLAB(*_inputs[0], getActs(), _center);
}

// Can't do this
void RGBToLABLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    assert(false);
}

/* 
 * =====================
 * ResponseNormLayer
 * =====================
 */
ResponseNormLayer::ResponseNormLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {
    _channels = pyDictGetInt(paramsDict, "channels");
    _size = pyDictGetInt(paramsDict, "size");

    _scale = pyDictGetFloat(paramsDict, "scale");
    _pow = pyDictGetFloat(paramsDict, "pow");
}

void ResponseNormLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convResponseNorm(*_inputs[0], _denoms, getActs(), _channels, _size, _scale, _pow);
}

void ResponseNormLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convResponseNormUndo(v, _denoms, _prev[0]->getActs(), getActs(), _prev[0]->getActsGrad(), _channels, _size, _scale, _pow, scaleTargets, 1);
}

void ResponseNormLayer::truncBwdActs() {
    Layer::truncBwdActs();
    if (_conserveMem) {
        _denoms.truncate();
    }
}

/* 
 * =====================
 * CrossMapResponseNormLayer
 * =====================
 */
CrossMapResponseNormLayer::CrossMapResponseNormLayer(ConvNet* convNet, PyObject* paramsDict) : ResponseNormLayer(convNet, paramsDict) {
    _blocked = pyDictGetInt(paramsDict, "blocked");
}

void CrossMapResponseNormLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convResponseNormCrossMap(*_inputs[0], _denoms, getActs(), _channels, _size, _scale, _pow, _blocked);
}

void CrossMapResponseNormLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convResponseNormCrossMapUndo(v, _denoms, _prev[0]->getActs(), getActs(), _prev[0]->getActsGrad(), _channels, _size, _scale, _pow, _blocked, scaleTargets, 1);
}


/* 
 * =====================
 * ContrastNormLayer
 * =====================
 */
ContrastNormLayer::ContrastNormLayer(ConvNet* convNet, PyObject* paramsDict) : ResponseNormLayer(convNet, paramsDict) {
    _imgSize = pyDictGetInt(paramsDict, "imgSize");
}

void ContrastNormLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    NVMatrix& images = *_inputs[0];
    convLocalPool(images, _meanDiffs, _channels, _size, -_size/2, 1, _imgSize, AvgPooler());
    _meanDiffs.add(images, -1, 1);
    convContrastNorm(images, _meanDiffs, _denoms, getActs(), _channels, _size, _scale, _pow);
}

void ContrastNormLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convContrastNormUndo(v, _denoms, _meanDiffs, getActs(), _prev[inpIdx]->getActsGrad(), _channels, _size, _scale, _pow, scaleTargets, 1);
}

void ContrastNormLayer::truncBwdActs() {
    ResponseNormLayer::truncBwdActs();
    if (_conserveMem) {
        _meanDiffs.truncate();
    }
}

/* 
 * =====================
 * CostLayer
 * =====================
 */
CostLayer::CostLayer(ConvNet* convNet, PyObject* paramsDict, bool trans) 
    : Layer(convNet, paramsDict, trans) {
    _coeff = pyDictGetFloat(paramsDict, "coeff");
}

float CostLayer::getCoeff() {
    return _coeff;
}

void CostLayer::bprop(PASS_TYPE passType) {
    if (_coeff != 0) {
        Layer::bprop(passType);
    }
}

bool CostLayer::isGradProducer() {
    return _coeff != 0;
}

doublev& CostLayer::getCost() {
    doublev& v = *new doublev();
    v.insert(v.begin(), _costv.begin(), _costv.end());
    return v;
}

CostLayer& CostLayer::makeCostLayer(ConvNet* convNet, string& type, PyObject* paramsDict) {
    if (type == "cost.logreg") {
        return *new LogregCostLayer(convNet, paramsDict);
    } else if (type == "cost.sum2") {
        return *new SumOfSquaresCostLayer(convNet, paramsDict);
    }
    throw string("Unknown cost layer type ") + type;
}

/* 
 * =====================
 * LogregCostLayer
 * =====================
 */
LogregCostLayer::LogregCostLayer(ConvNet* convNet, PyObject* paramsDict) : CostLayer(convNet, paramsDict, false) {
}

void LogregCostLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    // This layer uses its two inputs together
    if (inpIdx == 0) {
        NVMatrix& labels = *_inputs[0];
        NVMatrix& probs = *_inputs[1];
        int numCases = labels.getNumElements();
        NVMatrix& trueLabelLogProbs = getActs(), correctProbs;
        computeLogregCost(labels, probs, trueLabelLogProbs, correctProbs);
        _costv.clear();
        _costv.push_back(-trueLabelLogProbs.sum());
        _costv.push_back(numCases - correctProbs.sum());
    }
}

void LogregCostLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    assert(inpIdx == 1);
    NVMatrix& labels = _prev[0]->getActs();
    NVMatrix& probs = _prev[1]->getActs();
    NVMatrix& target = _prev[1]->getActsGrad();
    // Numerical stability optimization: if the layer below me is a softmax layer, let it handle
    // the entire gradient computation to avoid multiplying and dividing by a near-zero quantity.
    bool doWork = _prev[1]->getNext().size() > 1 || _prev[1]->getType() != "softmax";
    if (doWork) {
        computeLogregGrad(labels, probs, target, scaleTargets == 1, _coeff);
    }
}

/* 
 * =====================
 * SumOfSquaresCostLayer
 * =====================
 */
SumOfSquaresCostLayer::SumOfSquaresCostLayer(ConvNet* convNet, PyObject* paramsDict) : CostLayer(convNet, paramsDict, false) {
}

void SumOfSquaresCostLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    _inputs[0]->apply(NVMatrixOps::Square(), getActs());
    _costv.clear();
    _costv.push_back(getActs().sum());
}

void SumOfSquaresCostLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    _prev[inpIdx]->getActsGrad().add(*_inputs[0], scaleTargets, -2 * _coeff);
}
