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

#ifndef COST_CUH
#define	COST_CUH

#include <vector>
#include <map>
#include <cutil_inline.h>

#include "layer.cuh"
#include "util.cuh"

class CostLayer;

/*
 * Wrapper for dictionary mapping cost name to vector of returned values.
 */
class Cost {
private:
    int _numCases;
    CostMap _costMap;
    CostCoeffMap _costCoeffMap;
public:
    Cost(int numCases);
    Cost(int numCases, std::vector<CostLayer*>& costs);
    doublev& operator [](const std::string s);
    CostMap& getCostMap();
    CostCoeffMap& getCostCoeffMap();
    int getNumCases();
    /*
     * Returns sum of first values returned by all the costs, weighted by the cost coefficients.
     */
    double getValue();
    Cost& operator += (Cost& er);
    Cost& operator /= (const double v);
    virtual ~Cost();
};


#endif	/* COST_CUH */

