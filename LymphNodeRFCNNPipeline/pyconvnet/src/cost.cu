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

#include <iostream>
#include <cost.cuh>

using namespace std;

/* 
 * =====================
 * Cost
 * =====================
 */

Cost::Cost(int numCases) : _numCases(numCases) {
}

Cost::Cost(int numCases, vector<CostLayer*>& costs) : _numCases(numCases) {
    for (vector<CostLayer*>::iterator it = costs.begin(); it != costs.end(); ++it) {
        _costMap[(*it)->getName()] = &(*it)->getCost();
        _costCoeffMap[(*it)->getName()] = (*it)->getCoeff();
    }
}

int Cost::getNumCases() {
    return _numCases;
}

doublev& Cost::operator [](const string s) {
    return *_costMap[s];
}

CostMap& Cost::getCostMap() {
    return _costMap;
}

CostCoeffMap& Cost::getCostCoeffMap() {
    return _costCoeffMap;
}

double Cost::getValue() {
    double val = 0;
    for (CostMap::iterator it = _costMap.begin(); it != _costMap.end(); ++it) {
        val += _costCoeffMap[it->first] * it->second->at(0);
    }
    return val;
}

Cost& Cost::operator += (Cost& er) {
    CostMap& otherMap = er.getCostMap();
    CostCoeffMap& otherCoeffMap = er.getCostCoeffMap();
    for (CostMap::const_iterator it = otherMap.begin(); it != otherMap.end(); ++it) {
        if (_costMap.count(it->first) == 0) {
            _costMap[it->first] = new doublev();
            _costCoeffMap[it->first] = otherCoeffMap[it->first];
        }
        
        vector<double>& myVec = *_costMap[it->first];
        vector<double>& otherVec = *otherMap[it->first];
        for (int i = 0; i < otherVec.size(); i++) {
            if (myVec.size() <= i) {
                myVec.push_back(0);
            }
            myVec[i] += otherVec[i];
        }
    }
    _numCases += er.getNumCases();
    return *this;
}

Cost& Cost::operator /= (const double v) {
    for (CostMap::const_iterator it = _costMap.begin(); it != _costMap.end(); ++it) {
        for (doublev::iterator it2 = it->second->begin(); it2 != it->second->end(); ++it2) {
            *it2 /= v;
        }
    }
    return *this;
}

Cost::~Cost() {
    for (CostMap::const_iterator it = _costMap.begin(); it != _costMap.end(); ++it) {
        delete it->second;
    }
}