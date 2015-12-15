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

#ifndef NEURONS_CUH
#define	NEURONS_CUH

#include <assert.h>
#include <string>
#include <nvmatrix.cuh>
#include <cutil_inline.h>

template <class GradientOp>
class AddGradientBinaryOperator {
    GradientOp _op;
public:
    AddGradientBinaryOperator(GradientOp op) : _op(op) {
    }
    __device__ inline float operator()(const float unitActGrad, const float unitAct, const float target) const {
        return target + _op(unitActGrad, unitAct); 
    }
};

template <class GradientOp>
class AddGradientOperator {
    GradientOp _op;
public:
    AddGradientOperator(GradientOp op) : _op(op) {
    }
    __device__ inline float operator()(const float unitActGrad, const float target) const {
        return target + _op(unitActGrad); 
    }
};

/* =======================
 * Neuron
 * -----------------------
 * 
 * f(x) = x
 * =======================
 */
class Neuron {
protected:
    bool _activated;
    // Inputs and outputs potentially point to the same matrix, depending on the neuron
    NVMatrix* _inputs, *_outputs; 
    virtual void _activate() {
        if (_inputs != _outputs) {
            _inputs->copy(*_outputs);
        }
    }
    virtual void _computeInputGrad(NVMatrix& actsGrad, NVMatrix& target) {
        if (&target != &actsGrad) {
            actsGrad.copy(target);
        }
    }
    virtual void _addInputGrad(NVMatrix& actsGrad, NVMatrix& target) {
        if (&target != &actsGrad) {
            target.add(actsGrad);
        }
    }
public:
    Neuron() : _activated(false), _inputs(NULL), _outputs(NULL) {
    }
    virtual void activate(NVMatrix& inputs, NVMatrix& outputs) {
        _activated = true;
        _inputs = &inputs;
        _outputs = &outputs;
        _activate();
    }

    virtual void computeInputGrad(NVMatrix& actsGrad, NVMatrix& target, bool add) {
        assert(_activated);
        if (!add) {
            target.resize(actsGrad);
            _computeInputGrad(actsGrad, target);
        } else {
            _addInputGrad(actsGrad, target);
        }
    }
        
    static Neuron& makeNeuron(PyObject* neuronDict);
};

/* =======================
 * LogisticNeuron
 * -----------------------
 * 
 * f(x) = 1 / (1 + e^-x)
 * =======================
 */
class LogisticNeuron : public Neuron {
protected:
    void _activate() {
        _inputs->apply(NVMatrixOps::Logistic(), *_outputs);
    }

    void _computeInputGrad(NVMatrix& actsGrad, NVMatrix& target) {
        actsGrad.applyBinary(LogisticGradientOperator(), *_outputs, target);
    }
    
    void _addInputGrad(NVMatrix& actsGrad, NVMatrix& target) {
        actsGrad.applyTernary(AddGradientBinaryOperator<LogisticGradientOperator>(LogisticGradientOperator()), *_outputs, target, target);
    }
public:
    class LogisticGradientOperator {
    public:
        __device__ inline float operator()(float unitActGrad, float unitAct) const {
            return unitActGrad * unitAct * (1.0f - unitAct); 
        }
    };
    
    LogisticNeuron() : Neuron() {
    }
};

/* =======================
 * ReluNeuron
 * -----------------------
 * 
 * f(x) = max(0, x)
 * =======================
 */
class ReluNeuron : public Neuron {
protected:
    void _activate() {
        _inputs->apply(ReluOperator(), *_outputs);
    }

    void _computeInputGrad(NVMatrix& actsGrad, NVMatrix& target) {
        actsGrad.applyBinary(ReluGradientOperator(), *_outputs, target);
    }
    
    void _addInputGrad(NVMatrix& actsGrad, NVMatrix& target) {
        actsGrad.applyTernary(AddGradientBinaryOperator<ReluGradientOperator>(ReluGradientOperator()), *_outputs, target, target);
    }
public:
    class ReluOperator {
    public:    
        __device__ inline float operator()(float x) const {
            return x < 0.0f ? 0.0f : x;
        }
    };

    class ReluGradientOperator {
    public:
        __device__ inline float operator()(float unitActGrad, float unitAct) const  {
            return unitActGrad * (unitAct > 0.0f); 
        }
    };
    
    ReluNeuron() : Neuron() {
    }
};

/* =======================
 * BoundedReluNeuron
 * -----------------------
 * 
 * f(x) = min(a, max(0, x))
 * =======================
 */
class BoundedReluNeuron : public Neuron {
protected:
    float _a;
    
    void _activate() {
        _inputs->apply(BoundedReluOperator(_a), *_outputs);
    }

    void _computeInputGrad(NVMatrix& actsGrad, NVMatrix& target) {
        actsGrad.applyBinary(BoundedReluGradientOperator(_a), *_outputs, target);
    }
    
    void _addInputGrad(NVMatrix& actsGrad, NVMatrix& target) {
        actsGrad.applyTernary(AddGradientBinaryOperator<BoundedReluGradientOperator>(BoundedReluGradientOperator(_a)), *_outputs, target, target);
    }
public:
    class BoundedReluOperator {
    private:
        float _a;
    public:
        BoundedReluOperator(float a) : _a(a) {
        }
        __device__ inline float operator()(float x) const {
            return x < 0.0f ? 0.0f : x > _a ? _a : x;
        }
    };

    class BoundedReluGradientOperator {
    private:
        float _a;
    public:
        BoundedReluGradientOperator(float a) : _a(a) {
        }
        __device__ inline float operator()(float unitActGrad, float unitAct) const  {
            return unitActGrad * (unitAct > 0.0f) * (unitAct < _a); 
        }
    };
    
    BoundedReluNeuron(float a) : Neuron(), _a(a) {
    }
};

/* =======================
 * AbsNeuron
 * -----------------------
 * 
 * f(x) = abs(x)
 * =======================
 */
class AbsNeuron : public Neuron {
protected:
    void _activate() {
        assert(_inputs != _outputs);
        _inputs->apply(NVMatrixOps::Abs(), *_outputs);
    }

    void _computeInputGrad(NVMatrix& actsGrad, NVMatrix& target) {
        actsGrad.applyBinary(AbsGradientOperator(), *_inputs, target);
    }
    
    void _addInputGrad(NVMatrix& actsGrad, NVMatrix& target) {
        actsGrad.applyTernary(AddGradientBinaryOperator<AbsGradientOperator>(AbsGradientOperator()), *_inputs, target, target);
    }
public:
    class AbsGradientOperator {
    public:
        __device__ inline float operator()(float unitActGrad, float unitInput) const  {
            return unitActGrad * (unitInput > 0.0f ? 1.0f : -1.0f); 
        }
    };
    
    AbsNeuron() : Neuron() {
    }
};

/* =======================
 * TanhNeuron
 * -----------------------
 * 
 * f(x) = a*tanh(b*x)
 * =======================
 */
class TanhNeuron : public Neuron {
protected:
    float _a, _b;

    void _activate() {
        _inputs->apply(TanhOperator(_a, _b), *_outputs);
    }

    void _computeInputGrad(NVMatrix& actsGrad, NVMatrix& target) {
        actsGrad.applyBinary(TanhGradientOperator(_a, _b), *_outputs, target);
    }
    
    void _addInputGrad(NVMatrix& actsGrad, NVMatrix& target) {
        actsGrad.applyTernary(AddGradientBinaryOperator<TanhGradientOperator>(TanhGradientOperator(_a, _b)), *_outputs, target, target);
    }
public:
    class TanhOperator {
    private:
        float _a, _n2b;
    public:
        TanhOperator(float a, float b) : _a(a), _n2b(-2*b) {
        }
        virtual __device__ inline float operator()(float x) const {
            return _a * (__fdividef(2.0f, 1.0f + __expf(x * _n2b)) - 1.0f);
        }
    };

    class TanhGradientOperator {
    private:
        float _n4ab, _a;
    public:
        TanhGradientOperator(float a, float b) : _n4ab(-4*a*b), _a(a) {
        }
        __device__ inline float operator()(float unitActGrad, float unitAct) const  {
            const float t = (1.0f - __fdividef(unitAct, _a)) / 2.0f;
            return unitActGrad * _n4ab * (t * (t - 1.0f));
        }
    };
    
    TanhNeuron(float a, float b) : Neuron(), _a(a), _b(b) {
    }
};

/* =======================
 * SoftReluNeuron
 * -----------------------
 * 
 * f(x) = log(1 + e^x)
 * =======================
 */
class SoftReluNeuron : public Neuron {
protected:
    void _activate() {
        assert(_inputs != _outputs);
        _inputs->apply(SoftReluOperator(), *_outputs);
    }

    void _computeInputGrad(NVMatrix& actsGrad, NVMatrix& target) {
        actsGrad.applyBinary(SoftReluGradientOperator(), *_outputs, target);
    }
    
    void _addInputGrad(NVMatrix& actsGrad, NVMatrix& target) {
        actsGrad.applyTernary(AddGradientBinaryOperator<SoftReluGradientOperator>(SoftReluGradientOperator()), *_outputs, target, target);
    }
public:
    class SoftReluOperator {
    public:    
        __device__ inline float operator()(float x) const {
            // This piece-wise implementation has better numerical stability than
            // simply computing log(1 + e^x).
            return x > 4.0f ? x : __logf(1.0f + __expf(x));
        }
    };

    class SoftReluGradientOperator {
    public:
        __device__ inline float operator()(float unitActGrad, float unitInput) const  {
            if (unitInput > 4.0f) {
                return unitActGrad;
            }
            const float f = __expf(unitInput);
            return unitActGrad * __fdividef(f, 1.0f + f); 
        }
    };
    
    SoftReluNeuron() : Neuron() {
    }
};

/* =======================
 * SquareNeuron
 * -----------------------
 * 
 * f(x) = x^2
 * =======================
 */
class SquareNeuron : public Neuron {
protected:
    void _activate() {
        assert(_inputs != _outputs);
        _inputs->apply(NVMatrixOps::Square(), *_outputs);
    }

    void _computeInputGrad(NVMatrix& actsGrad, NVMatrix& target) {
        actsGrad.applyBinary(SquareGradientOperator(), *_inputs, target);
    }
    
    void _addInputGrad(NVMatrix& actsGrad, NVMatrix& target) {
        actsGrad.applyTernary(AddGradientBinaryOperator<SquareGradientOperator>(SquareGradientOperator()), *_inputs, target, target);
    }
public:
    class SquareGradientOperator {
    public:
        __device__ inline float operator()(float unitActGrad, float unitInput) const {
            return unitActGrad * 2.0f * unitInput; 
        }
    };
    
    SquareNeuron() : Neuron() {
    }
};

/* =======================
 * SqrtNeuron
 * -----------------------
 * 
 * f(x) = sqrt(x)
 * =======================
 */
class SqrtNeuron : public Neuron {
protected:
    void _activate() {
        _inputs->apply(NVMatrixOps::Sqrt(), *_outputs);
    }

    void _computeInputGrad(NVMatrix& actsGrad, NVMatrix& target) {
        actsGrad.applyBinary(SqrtGradientOperator(), *_outputs, target);
    }
    
    void _addInputGrad(NVMatrix& actsGrad, NVMatrix& target) {
        actsGrad.applyTernary(AddGradientBinaryOperator<SqrtGradientOperator>(SqrtGradientOperator()), *_outputs, target, target);
    }
public:
    class SqrtGradientOperator {
    public:
        __device__ inline float operator()(float unitActGrad, float unitAct) const {
            return __fdividef(unitActGrad, 2.0f * unitAct); 
        }
    };
    
    SqrtNeuron() : Neuron() {
    }
};

/* =======================
 * LinearNeuron
 * -----------------------
 * 
 * f(x) = a*x + b
 * =======================
 */
class LinearNeuron : public Neuron {
protected:
    float _a, _b;
    void _activate() {
        _inputs->apply(LinearOperator(_a, _b), *_outputs);
    }

    void _computeInputGrad(NVMatrix& actsGrad, NVMatrix& target) {
        actsGrad.scale(_a, target);
    }
    
    void _addInputGrad(NVMatrix& actsGrad, NVMatrix& target) {
        actsGrad.applyBinary(AddGradientOperator<NVMatrixOps::MultByScalar>(NVMatrixOps::MultByScalar(_a)), target, target);
    }
public:
    class LinearOperator {
    protected:
        float _a, _b;
    public:    
        __device__ inline float operator()(float x) const {
            return _a * x + _b;
        }
        LinearOperator(float a, float b) : _a(a), _b(b) {
        }
    };
    
    LinearNeuron(float a, float b) : Neuron(), _a(a), _b(b) {
    }
};
#endif	/* NEURONS_CUH */

