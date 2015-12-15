/*
 *  _reg_mutualinformation.h
 *
 *
 *  Created by Marc Modat on 25/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_MUTUALINFORMATION_H
#define _REG_MUTUALINFORMATION_H

#include "nifti1_io.h"
#include <vector>
#ifdef _OPENMP
  #include "omp.h"
#endif

// Simple class to dynamically manage an array of pointers
// Needed for multi channel NMI
template<class DataTYPE>
class SafeArray
{
public:
    /// Constructor
    SafeArray(int items)
    {
        data = new DataTYPE[items];
    }

    /// Destructor
    ~SafeArray()
    {
        delete[] data;
    }

    /// Implicit conversion
    operator DataTYPE *()
    {
        return data;
    }

private:
    void operator=(const SafeArray &){};
    SafeArray(const SafeArray &){};

    DataTYPE *data;
};

//-----------------------------------------------------------------------------
// Template for emulating nested multiple loops, where the number of nested loops
// is only known at runtime.
// The index type may be any incrementable type, including pointers and iterators.
// 'end' values are like the STL ranges, where they signify one past the last value.
//-----------------------------------------------------------------------------
template<typename T>
class Multi_Loop
{
public:
    /// Add a for loop to the list
    void Add(T begin_value, T end_value){
        begin.push_back(begin_value);
        end.push_back(end_value);
    }

    // Initialises the loops before use.
    void Initialise(){
        current.resize(Count());
        std::copy(begin.begin(), begin.end(), current.begin());
    }

    /// Gets the index or iterator for the specified loop.
    T Index(int index) const{
        return (current[index]);
    }

    /// Gets the index or iterator for the specified loop.
    const T &operator [](int index) const{
        return (current[index]);
    }

    /// Tests to see if the loops continue.
    bool Continue() const{
        return (current[0] != end[0]);
    }

    /// Compute the next set of indexes or iterators in the sequence.
    void Next(){
        int position = begin.size() - 1;
        bool finished = false;

        while (!finished)
        {
            ++current[position];
            // Finished incrementing?
            if ((current[position] != end[position]) || (position == 0)){
                finished = true;
            }
            else{
                // Reset this index, and move on to the previous one.
                current[position] = begin[position];
                --position;
            }
        }
    }

    /// Returns the number of 'for' loops added.
    int Count() const{
        return (static_cast<int>(begin.size()));
    }

private:
    std::vector<T> begin;   // Start for each loop.
    std::vector<T> end;     // End for each loop.
    std::vector<T> current; // Current position of each loop
};

/// Some methods that will be needed for generating the multi-channel histogram
/// Needed for multi channel NMI
inline int calculate_product(int dim, int *dimensions)
{
    int product = 1;
    for(int i = 0; i < dim; ++i) product *= dimensions[i];

    return product;
}

inline int calculate_index(int num_dims, int *dimensions, int *indices)
{
    int index = 0;
    for(int i = 0; i < num_dims; ++i) index += indices[i] * calculate_product(i, dimensions);

    return index;
}

inline int previous(int current, int num_dims){
  if(current > 0) return current - 1;

  return num_dims - 1;
}

/// Multi channel NMI version - Entropy
extern "C++"
void reg_getEntropies(nifti_image *targetImages,
                      nifti_image *resultImages,
                      unsigned int *target_bins, // should be an array of size num_target_volumes
                      unsigned int *result_bins, // should be an array of size num_result_volumes
                      double *probaJointHistogram,
                      double *logJointHistogram,
                      double *entropies,
                      int *mask,
                      bool approx);

/// Multi channel NMI version - Gradient
extern "C++"
void reg_getVoxelBasedNMIGradientUsingPW(nifti_image *targetImages,
                                         nifti_image *resultImages,
                                         nifti_image *resultImageGradient,
                                         unsigned int *target_bins,
                                         unsigned int *result_bins,
                                         double *logJointHistogram,
                                         double *entropies,
                                         nifti_image *nmiGradientImage,
                                         int *mask,
                                         bool approx);

#endif
