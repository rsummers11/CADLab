/*
 *  _reg_mutualinformation.cpp
 *
 *
 *  Created by Marc Modat on 25/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_MUTUALINFORMATION_CPP
#define _REG_MUTUALINFORMATION_CPP

#include "_reg_mutualinformation.h"
#include "_reg_tools.h"
#include <iostream>

/// Smooth the histogram along the given axes. Uses recursion
template<class PrecisionTYPE>
void smooth_axes(int axes, int current, PrecisionTYPE *histogram,
                 PrecisionTYPE *result, PrecisionTYPE *window,
                 int num_dims, int *dimensions, int *indices)
{
    int temp, index;
    PrecisionTYPE value;
    for(indices[current] = 0; indices[current] < dimensions[current]; ++indices[current])
    {
        if(axes == current) {
            temp = indices[current];
            indices[current]--;
            value = (PrecisionTYPE)0;
            for(int it=0; it<3; it++) {
                if(-1<indices[current] && indices[current]<dimensions[current]) {
                    index = calculate_index(num_dims, dimensions, indices);
                    value += histogram[index] * window[it];
                }
                indices[current]++;
            }
            indices[current] = temp;
            index = calculate_index(num_dims, dimensions, indices);
            result[index] = value;
        }
        else {
            smooth_axes<PrecisionTYPE>(axes, previous(current, num_dims), histogram,
                                       result, window, num_dims, dimensions, indices);
        }
    }
}

/// Traverse the histogram along the specified axes and smooth along it
template<class PrecisionTYPE>
void traverse_and_smooth_axes(int axes, PrecisionTYPE *histogram,
                              PrecisionTYPE *result, PrecisionTYPE *window,
                              int num_dims, int *dimensions)
{
    SafeArray<int> indices(num_dims);
    for(int dim = 0; dim < num_dims; ++dim) indices[dim] = 0;

    smooth_axes<PrecisionTYPE>(axes, previous(axes, num_dims), histogram,
                               result, window, num_dims, dimensions, indices);
}

/// Sum along the specified axes. Uses recursion
template<class PrecisionTYPE>
void sum_axes(int axes, int current, PrecisionTYPE *histogram, PrecisionTYPE *&sums,
              int num_dims, int *dimensions, int *indices)
{
    int index;
    PrecisionTYPE value = (PrecisionTYPE)0;

    for(indices[current] = 0; indices[current] < dimensions[current]; ++indices[current])
    {
        if(axes == current) {
            index = calculate_index(num_dims, dimensions, indices);
            value += histogram[index];
        }
        else {
            sum_axes<PrecisionTYPE>(axes, previous(current, num_dims), histogram,
                                    sums, num_dims, dimensions, indices);
        }
    }
    // Store the sum along the current line and increment the storage pointer
    if (axes == current)
    {
        *(sums) = value;
        ++sums;
    }
}

/// Traverse and sum along an axes
template<class PrecisionTYPE>
void traverse_and_sum_axes(int axes, PrecisionTYPE *histogram, PrecisionTYPE *&sums,
                           int num_dims, int *dimensions)
{
    SafeArray<int> indices(num_dims);
    for(int dim = 0; dim < num_dims; ++dim) indices[dim] = 0;
    sum_axes<PrecisionTYPE>(axes, previous(axes, num_dims), histogram, sums,
                            num_dims, dimensions, indices);
}


/* *************************************************************** */
template<class PrecisionTYPE>
PrecisionTYPE GetBasisSplineValue(PrecisionTYPE x)
{
    x=fabs(x);
    PrecisionTYPE value=0.0;
    if(x<2.0){
        if(x<1.0)
            value = (PrecisionTYPE)(2.0f/3.0f + (0.5f*x-1.0)*x*x);
        else{
            x-=2.0f;
            value = -x*x*x/6.0f;
        }
    }
    return value;
}
/* *************************************************************** */
template<class PrecisionTYPE>
PrecisionTYPE GetBasisSplineDerivativeValue(PrecisionTYPE ori)
{
    PrecisionTYPE x=fabs(ori);
    PrecisionTYPE value=0.0;
    if(x<2.0){
        if(x<1.0)
            value = (PrecisionTYPE)((1.5f*x-2.0)*ori);
        else{
            x-=2.0f;
            value = -0.5f * x * x;
            if(ori<0.0f)value =-value;
        }
    }
    return value;
}

/* *************************************************************** */
/* *************************************************************** */
/// Multi channel NMI joint histogram and entropy calculation
template<class DTYPE>
void reg_getEntropies1(nifti_image *targetImage,
                       nifti_image *resultImage,
                       unsigned int *target_bins,
                       unsigned int *result_bins,
                       double *probaJointHistogram,
                       double *logJointHistogram,
                       double *entropies,
                       int *mask,
                       bool approx)
{
    int num_target_volumes = targetImage->nt;
    int num_result_volumes = resultImage->nt;
    int i, j, index;

    if(num_target_volumes>1 || num_result_volumes>1) approx=true;

    int targetVoxelNumber = targetImage->nx * targetImage->ny * targetImage->nz;

    DTYPE *targetImagePtr = static_cast<DTYPE *>(targetImage->data);
    DTYPE *resultImagePtr = static_cast<DTYPE *>(resultImage->data);

    // Build up this arrays of offsets that will help us index the histogram entries
    SafeArray<int> target_offsets(num_target_volumes);
    SafeArray<int> result_offsets(num_result_volumes);

    int num_histogram_entries = 1;
    int total_target_entries = 1;
    int total_result_entries = 1;

    // Data pointers
    SafeArray<int> histogram_dimensions(num_target_volumes + num_result_volumes);

    // Calculate some constants and initialize the data pointers
    for (i = 0; i < num_target_volumes; ++i) {
        num_histogram_entries *= target_bins[i];
        total_target_entries *= target_bins[i];
        histogram_dimensions[i] = target_bins[i];

        target_offsets[i] = 1;
        for (j = i; j > 0; --j) target_offsets[i] *= target_bins[j - 1];
    }

    for (i = 0; i < num_result_volumes; ++i) {
        num_histogram_entries *= result_bins[i];
        total_result_entries *= result_bins[i];
        histogram_dimensions[num_target_volumes + i] = result_bins[i];

        result_offsets[i] = 1;
        for (j = i; j > 0; --j) result_offsets[i] *= result_bins[j-1];
    }

    int num_probabilities = num_histogram_entries;

    // Space for storing the marginal entropies.
    num_histogram_entries += total_target_entries + total_result_entries;

    memset(probaJointHistogram, 0, num_histogram_entries * sizeof(double));
    memset(logJointHistogram, 0, num_histogram_entries * sizeof(double));

    // These hold the current target and result values
    // No more than 10 timepoints are assumed
    DTYPE target_values[10];
    DTYPE result_values[10];

    bool valid_values;

    DTYPE target_flat_index, result_flat_index;
    double voxel_number = 0., added_value;

    // For now we only use the approximate PW approach for filling the joint histogram.
    // Fill the joint histogram using the classical approach
#ifdef _OPENMP
    int maxThreadNumber = omp_get_max_threads(), tid;
    double **tempHistogram=(double **)malloc(maxThreadNumber*sizeof(double *));
    for(i=0;i<maxThreadNumber;++i)
        tempHistogram[i]=(double *)calloc(num_histogram_entries,sizeof(double));
#pragma omp parallel for default(none) \
    shared(tempHistogram, num_target_volumes, num_result_volumes, mask, \
    targetImagePtr, resultImagePtr, targetVoxelNumber, target_bins, result_bins, \
    target_offsets, result_offsets, total_target_entries, approx) \
    private(index, i, valid_values, target_flat_index, tid, \
    result_flat_index, target_values, result_values, added_value) \
    reduction(+:voxel_number)
#endif
    for (index=0; index<targetVoxelNumber; ++index){
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        if (mask[index] > -1) {
            added_value=0.;
            valid_values = true;
            target_flat_index = 0;

            // Get the target values
            for (i = 0; i < num_target_volumes; ++i) {
                target_values[i] = targetImagePtr[index+i*targetVoxelNumber];
                if (target_values[i] < (DTYPE)0 ||
                    target_values[i] >= (DTYPE)target_bins[i] ||
                    target_values[i] != target_values[i]) {
                    valid_values = false;
                    break;
                }
                target_flat_index += target_values[i] * (DTYPE)(target_offsets[i]);
            }

            if (valid_values) {
                result_flat_index = 0;
                // Get the result values
                for (i = 0; i < num_result_volumes; ++i){
                    result_values[i] = resultImagePtr[index+i*targetVoxelNumber];
                    if (result_values[i] < (DTYPE)0 ||
                        result_values[i] >= (DTYPE)result_bins[i] ||
                        result_values[i] != result_values[i]) {
                        valid_values = false;
                        break;
                    }
                    result_flat_index += result_values[i] * (DTYPE)(result_offsets[i]);
                }
            }
            if (valid_values) {
                if(approx){ // standard joint histogram filling
#ifdef _OPENMP
                    tempHistogram[tid][static_cast<int>(round(target_flat_index)) +
                            (static_cast<int>(round(result_flat_index)) * total_target_entries)]++;
#else
                    probaJointHistogram[static_cast<int>(round(target_flat_index)) +
                            (static_cast<int>(round(result_flat_index)) * total_target_entries)]++;
#endif
                    added_value=1;
                }
                else{ // Parzen window joint histogram filling
                    for(int t=static_cast<int>(target_values[0]-1.); t<static_cast<int>(target_values[0]+2.); ++t){
                        if(t>=0 || t<static_cast<int>(target_bins[0])){
                            double target_weight = GetBasisSplineValue<double>(double(target_values[0])-double(t));
                            for(int r=static_cast<int>(result_values[0]-1.); r<static_cast<int>(result_values[0]+2.); ++r){
                                if(r>=0 || r<static_cast<int>(result_bins[0])){
                                    double weight = target_weight * GetBasisSplineValue<double>(double(result_values[0])-double(r));
                                    added_value+= weight;
#ifdef _OPENMP
                                    tempHistogram[tid][t + r * total_target_entries]  += weight;
#else
                                    probaJointHistogram[t + r * total_target_entries] += weight;
#endif
                                }
                            }
                        }
                    }
                }
            }
            voxel_number+=added_value;
        } //mask
    }
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(maxThreadNumber, num_histogram_entries, probaJointHistogram, tempHistogram) \
    private(i,j)
    for(i=0;i<num_histogram_entries;++i){
        for(j=0;j<maxThreadNumber;++j){
            probaJointHistogram[i] += tempHistogram[j][i];
        }
    }
    for(j=0;j<maxThreadNumber;++j)
        free(tempHistogram[j]);
    free(tempHistogram);
#endif

    int num_axes = num_target_volumes + num_result_volumes;
    if(approx || targetImage->nt>1 || resultImage->nt>1){
    // standard joint histogram filling has been used
    // Joint histogram has to be smoothed
        double window[3];
        window[0] = window[2] = GetBasisSplineValue((double)(-1.0));
        window[1] = GetBasisSplineValue((double)(0.0));

        double *histogram=NULL;
        double *result=NULL;

        // Smooth along each of the axes
        for (i = 0; i < num_axes; ++i)
        {
            // Use the arrays for storage of results
            if (i % 2 == 0) {
                result = logJointHistogram;
                histogram = probaJointHistogram;
            }
            else {
                result = probaJointHistogram;
                histogram = logJointHistogram;
            }
            traverse_and_smooth_axes<double>(i, histogram, result, window,
                                             num_axes, histogram_dimensions);
        }

        // We may need to transfer the result
        if (result == logJointHistogram) memcpy(probaJointHistogram, logJointHistogram,
                                                sizeof(double)*num_probabilities);
    }// approx
    memset(logJointHistogram, 0, num_histogram_entries * sizeof(double));

    // Convert to probabilities
    for(i = 0; i < num_probabilities; ++i) {
        if (probaJointHistogram[i]) probaJointHistogram[i] /= voxel_number;
    }

    // Marginalise over all the result axes to generate the target entropy
    double *data = probaJointHistogram;
    double *store = logJointHistogram;
    double current_value, current_log;

    int count;
    double target_entropy = 0;
    {
        SafeArray<double> scratch (num_probabilities/histogram_dimensions[num_axes - 1]);
        // marginalise over the result axes
        for (i = num_result_volumes-1, count = 0; i >= 0; --i, ++count)
        {
            traverse_and_sum_axes<double>(num_axes - count - 1,
                                          data, store, num_axes - count,
                                          histogram_dimensions);

            if (count % 2 == 0) {
                data = logJointHistogram;
                store = scratch;
            }
            else {
                data = scratch;
                store = logJointHistogram;
            }
        }

        // Generate target entropy
        double *log_joint_target = &logJointHistogram[num_probabilities];

        for (i = 0; i < total_target_entries; ++i)
        {
            current_value = data[i];            
            current_log = 0;
            if (current_value) current_log = log(current_value);
            target_entropy -= current_value * current_log;
            log_joint_target[i] = current_log;
        }
    }
    memset(logJointHistogram, 0, num_probabilities * sizeof(double));
    data = probaJointHistogram;
    store = logJointHistogram;

    // Marginalise over the target axes
    double result_entropy = 0;
    {
        SafeArray<double> scratch (num_probabilities / histogram_dimensions[0]);
        for (i = 0; i < num_target_volumes; ++i)
        {
            traverse_and_sum_axes<double>(0, data, store, num_axes - i, &histogram_dimensions[i]);
            if (i % 2 == 0) {
                data = logJointHistogram;
                store = scratch;
            }
            else {
                data = scratch;
                store = logJointHistogram;
            }
        }
        // Generate result entropy
        double *log_joint_result = &logJointHistogram[num_probabilities+total_target_entries];

        for (i = 0; i < total_result_entries; ++i)
        {
            current_value = data[i];            
            current_log = 0;
            if (current_value) current_log = log(current_value);
            result_entropy -= current_value * current_log;
            log_joint_result[i] = current_log;
        }
    }

    // Generate joint entropy
    double joint_entropy = 0;
    for (i = 0; i < num_probabilities; ++i)
    {
        current_value = probaJointHistogram[i];        
        current_log = 0;
        if (current_value) current_log = log(current_value);
        joint_entropy -= current_value * current_log;
        logJointHistogram[i] = current_log;
    }

    entropies[0] = target_entropy;
    entropies[1] = result_entropy;
    entropies[2] = joint_entropy;
    entropies[3] = voxel_number;

    return;
}
/***************************************************************** */
extern "C++"
void reg_getEntropies(nifti_image *targetImage,
                      nifti_image *resultImage,
                      unsigned int *target_bins, // should be an array of size num_target_volumes
                      unsigned int *result_bins, // should be an array of size num_result_volumes
                      double *probaJointHistogram,
                      double *logJointHistogram,
                      double *entropies,
                      int *mask,
                      bool approx)
{
    if(targetImage->datatype != resultImage->datatype){
        fprintf(stderr, "[NiftyReg ERROR] reg_getEntropies\n");
        fprintf(stderr, "[NiftyReg ERROR] Both input images are exepected to have the same type\n");
        exit(1);
    }

    switch(targetImage->datatype){
    case NIFTI_TYPE_FLOAT32:
        reg_getEntropies1<float>
                (targetImage, resultImage, /*type,*/ target_bins, result_bins, probaJointHistogram,
                 logJointHistogram, entropies, mask, approx);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_getEntropies1<double>
                (targetImage, resultImage, /*type,*/ target_bins, result_bins, probaJointHistogram,
                 logJointHistogram, entropies, mask, approx);
        break;
    default:
        fprintf(stderr,"[NiftyReg ERROR] reg_getEntropies\tThe target image data type is not supported\n");
        exit(1);
    }
    return;
}
/* *************************************************************** */
/* *************************************************************** */
/// Voxel based multichannel gradient computation
template<class DTYPE,class GradTYPE>
void reg_getVoxelBasedNMIGradientUsingPW2D(nifti_image *targetImage,
                                           nifti_image *resultImage,
                                           //int type, //! Not used at the moment
                                           nifti_image *resultImageGradient,
                                           unsigned int *target_bins,
                                           unsigned int *result_bins,
                                           double *logJointHistogram,
                                           double *entropies,
                                           nifti_image *nmiGradientImage,
                                           int *mask,
                                           bool approx)
{
    unsigned int num_target_volumes=targetImage->nt;
    unsigned int num_result_volumes=resultImage->nt;
    unsigned int num_loops = num_target_volumes + num_result_volumes;

    if(num_target_volumes>1 || num_result_volumes>1) approx=true;

    unsigned targetVoxelNumber = targetImage->nx * targetImage->ny;

    DTYPE *targetImagePtr = static_cast<DTYPE *>(targetImage->data);
    DTYPE *resultImagePtr = static_cast<DTYPE *>(resultImage->data);
    GradTYPE *resulImageGradientPtrX = static_cast<GradTYPE *>(resultImageGradient->data);
    GradTYPE *resulImageGradientPtrY = &resulImageGradientPtrX[targetVoxelNumber*num_result_volumes];

    // Build up this arrays of offsets that will help us index the histogram entries
    int target_offsets[10];
    int result_offsets[10];

    int total_target_entries = 1;
    int total_result_entries = 1;

    // The 4D
    for (unsigned int i = 0; i < num_target_volumes; ++i) {
        total_target_entries *= target_bins[i];
        target_offsets[i] = 1;
        for (int j = i; j > 0; --j) target_offsets[i] *= target_bins[j - 1];
    }

    for (unsigned int i = 0; i < num_result_volumes; ++i) {
        total_result_entries *= result_bins[i];
        result_offsets[i] = 1;
        for (int j = i; j > 0; --j) result_offsets[i] *= result_bins[j - 1];
    }

    int num_probabilities = total_target_entries * total_result_entries;

    int *maskPtr = &mask[0];
    double NMI = (entropies[0] + entropies[1]) / entropies[2];

    // Hold current values.
    // target and result images limited to 10 max for speed.
    DTYPE voxel_values[20];
    GradTYPE result_gradient_x_values[10];
    GradTYPE result_gradient_y_values[10];

    bool valid_values;
    double common_target_value;

    double jointEntropyDerivative_X;
    double movingEntropyDerivative_X;
    double fixedEntropyDerivative_X;

    double jointEntropyDerivative_Y;
    double movingEntropyDerivative_Y;
    double fixedEntropyDerivative_Y;

    double jointLog, targetLog, resultLog;
    double joint_entropy = (double)(entropies[2]);

    GradTYPE *nmiGradientPtrX = static_cast<GradTYPE *>(nmiGradientImage->data);
    GradTYPE *nmiGradientPtrY = &nmiGradientPtrX[targetVoxelNumber];
    memset(nmiGradientPtrX,0,nmiGradientImage->nvox*nmiGradientImage->nbyper);

    // Set up the multi loop
    Multi_Loop<int> loop;
    for (unsigned int i = 0; i < num_loops; ++i) loop.Add(-1, 2);

    SafeArray<int> bins(num_loops);
    for (unsigned int i = 0; i < num_target_volumes; ++i) bins[i] = target_bins[i];
    for (unsigned int i = 0; i < num_result_volumes; ++i) bins[i + num_target_volumes] = result_bins[i];

    double coefficients[20];
    double positions[20];
    int relative_positions[20];

    double result_common[2];
    double der_term[3];

    // Loop over all the voxels
    for (unsigned int index = 0; index < targetVoxelNumber; ++index) {
        if(*maskPtr++>-1){
            valid_values = true;
            // Collect the target intensities and do some sanity checking
            for (unsigned int i = 0; i < num_target_volumes; ++i) {
                voxel_values[i] = targetImagePtr[index+i*targetVoxelNumber];
                if (voxel_values[i] <= (DTYPE)0 ||
                    voxel_values[i] >= (DTYPE)target_bins[i] ||
                    voxel_values[i] != voxel_values[i]) {
                    valid_values = false;
                    break;
                }
//                if(approx) // standard joint histogram filling
                    voxel_values[i] = (DTYPE)static_cast<int>(round(voxel_values[i]));
            }

            // Collect the result intensities and do some sanity checking
            if (valid_values) {
                for (unsigned int i = 0; i < num_result_volumes; ++i) {
                    unsigned int currentIndex = index+i*targetVoxelNumber;
                    DTYPE temp = resultImagePtr[currentIndex];
                    result_gradient_x_values[i] = resulImageGradientPtrX[currentIndex];
                    result_gradient_y_values[i] = resulImageGradientPtrY[currentIndex];

                    if (temp <= (DTYPE)0 ||
                        temp >= (DTYPE)result_bins[i] ||
                        temp != temp ||
                        result_gradient_x_values[i] != result_gradient_x_values[i] ||
                        result_gradient_y_values[i] != result_gradient_y_values[i]) {
                        valid_values = false;
                        break;
                    }
//                    if(approx) // standard joint histogram filling
                        voxel_values[num_target_volumes + i] = (DTYPE)static_cast<int>(round(temp));
//                    else voxel_values[num_target_volumes + i] = temp;
                }
            }
            if (valid_values) {
                jointEntropyDerivative_X = 0.0;
                movingEntropyDerivative_X = 0.0;
                fixedEntropyDerivative_X = 0.0;

                jointEntropyDerivative_Y = 0.0;
                movingEntropyDerivative_Y = 0.0;
                fixedEntropyDerivative_Y = 0.0;

                int target_flat_index, result_flat_index;

                for (loop.Initialise(); loop.Continue(); loop.Next()) {
                    target_flat_index = result_flat_index = 0;
                    valid_values = true;

                    for(unsigned int lc = 0; lc < num_target_volumes; ++lc){
                        int relative_pos = int(voxel_values[lc] + loop.Index(lc));
                        if(relative_pos< 0 || relative_pos >= bins[lc]){
                            valid_values = false; break;
                        }
                        double common_value = GetBasisSplineValue<double>((double)relative_pos-(double)voxel_values[lc]);
                        coefficients[lc] = common_value;
                        positions[lc] = (double)relative_pos-(double)voxel_values[lc];
                        relative_positions[lc] = relative_pos;
                    }

                    for(unsigned int jc = num_target_volumes; jc < num_loops; ++jc){
                        int relative_pos = int(voxel_values[jc] + loop.Index(jc));
                        if(relative_pos< 0 || relative_pos >= bins[jc]){
                            valid_values = false; break;
                        }
                        if (num_result_volumes > 1) {
                            double common_value = GetBasisSplineValue<double>((double)relative_pos-(double)voxel_values[jc]);
                            coefficients[jc] = common_value;
                        }
                        positions[jc] = (double)relative_pos-(double)voxel_values[jc];
                        relative_positions[jc] = relative_pos;
                    }

                    if(valid_values) {
                        common_target_value = (double)1.0;
                        for (unsigned int i = 0; i < num_target_volumes; ++i) common_target_value *= coefficients[i];

                        result_common[0] = result_common[1] = (double)0.0;

                        for (unsigned int i = 0; i < num_result_volumes; ++i)
                        {
                            der_term[0] = der_term[1] = der_term[2] = (double)1.0;
                            for (unsigned int j = 0; j < num_result_volumes; ++j)
                            {
                                if (i == j) {
                                    double reg = GetBasisSplineDerivativeValue<double>
                                            ((double)positions[j + num_target_volumes]);
                                    der_term[0] *= reg * (double)result_gradient_x_values[j];
                                    der_term[1] *= reg * (double)result_gradient_y_values[j];
                                }
                                else {
                                    der_term[0] *= coefficients[j+num_target_volumes];
                                    der_term[1] *= coefficients[j+num_target_volumes];
                                }
                            }
                            result_common[0] += der_term[0];
                            result_common[1] += der_term[1];
                        }

                        result_common[0] *= common_target_value;
                        result_common[1] *= common_target_value;

                        for (unsigned int i = 0; i < num_target_volumes; ++i) target_flat_index += relative_positions[i] * target_offsets[i];
                        for (unsigned int i = 0; i < num_result_volumes; ++i) result_flat_index += relative_positions[i + num_target_volumes] * result_offsets[i];

                        jointLog = logJointHistogram[target_flat_index + (result_flat_index * total_target_entries)];
                        targetLog = logJointHistogram[num_probabilities + target_flat_index];
                        resultLog = logJointHistogram[num_probabilities + total_target_entries + result_flat_index];

                        jointEntropyDerivative_X -= result_common[0] * jointLog;
                        fixedEntropyDerivative_X -= result_common[0] * targetLog;
                        movingEntropyDerivative_X -= result_common[0] * resultLog;

                        jointEntropyDerivative_Y -= result_common[1] * jointLog;
                        fixedEntropyDerivative_Y -= result_common[1] * targetLog;
                        movingEntropyDerivative_Y -= result_common[1] * resultLog;
                    }

                    *nmiGradientPtrX = (GradTYPE)((fixedEntropyDerivative_X + movingEntropyDerivative_X - NMI * jointEntropyDerivative_X) / joint_entropy);
                    *nmiGradientPtrY = (GradTYPE)((fixedEntropyDerivative_Y + movingEntropyDerivative_Y - NMI * jointEntropyDerivative_Y) / joint_entropy);
                }
            }
        }

        nmiGradientPtrX++; nmiGradientPtrY++;
    }
}
/* *************************************************************** */
/* *************************************************************** */
/// Voxel based multichannel gradient computation
template<class DTYPE,class GradTYPE>
void reg_getVoxelBasedNMIGradientUsingPW3D(nifti_image *targetImage,
                                           nifti_image *resultImage,
                                           nifti_image *resultImageGradient,
                                           unsigned int *target_bins,
                                           unsigned int *result_bins,
                                           double *logJointHistogram,
                                           double *entropies,
                                           nifti_image *nmiGradientImage,
                                           int *mask,
                                           bool approx)
{
    int num_target_volumes=targetImage->nt;
    int num_result_volumes=resultImage->nt;
    int num_loops = num_target_volumes + num_result_volumes;

    if(num_target_volumes>1 || num_result_volumes>1) approx=true;

    int targetVoxelNumber = targetImage->nx * targetImage->ny * targetImage->nz;

    DTYPE *targetImagePtr = static_cast<DTYPE *>(targetImage->data);
    DTYPE *resultImagePtr = static_cast<DTYPE *>(resultImage->data);
    GradTYPE *resultImageGradientPtrX = static_cast<GradTYPE *>(resultImageGradient->data);
    GradTYPE *resultImageGradientPtrY = &resultImageGradientPtrX[targetVoxelNumber*num_result_volumes];
    GradTYPE *resultImageGradientPtrZ = &resultImageGradientPtrY[targetVoxelNumber*num_result_volumes];

    // Build up this arrays of offsets that will help us index the histogram entries
    int target_offsets[10];
    int result_offsets[10];

    int total_target_entries = 1;
    int total_result_entries = 1;

    // The 4D
    for (int i = 0; i < num_target_volumes; ++i) {
        total_target_entries *= target_bins[i];
        target_offsets[i] = 1;
        for (int j = i; j > 0; --j) target_offsets[i] *= target_bins[j - 1];
    }

    for (int i = 0; i < num_result_volumes; ++i) {
        total_result_entries *= result_bins[i];
        result_offsets[i] = 1;
        for (int j = i; j > 0; --j) result_offsets[i] *= result_bins[j - 1];
    }

    int num_probabilities = total_target_entries * total_result_entries;

    double NMI = (entropies[0] + entropies[1]) / entropies[2];

    // Hold current values.
    // target and result images limited to 10 max for speed.
    DTYPE voxel_values[20];
    GradTYPE result_gradient_x_values[10];
    GradTYPE result_gradient_y_values[10];
    GradTYPE result_gradient_z_values[10];

    bool valid_values;
    double common_target_value;

    double jointEntropyDerivative_X;
    double movingEntropyDerivative_X;
    double fixedEntropyDerivative_X;

    double jointEntropyDerivative_Y;
    double movingEntropyDerivative_Y;
    double fixedEntropyDerivative_Y;

    double jointEntropyDerivative_Z;
    double movingEntropyDerivative_Z;
    double fixedEntropyDerivative_Z;

    double jointLog, targetLog, resultLog;
    double joint_entropy = (double)(entropies[2]);

    GradTYPE *nmiGradientPtrX = static_cast<GradTYPE *>(nmiGradientImage->data);
    GradTYPE *nmiGradientPtrY = &nmiGradientPtrX[targetVoxelNumber];
    GradTYPE *nmiGradientPtrZ = &nmiGradientPtrY[targetVoxelNumber];
    memset(nmiGradientPtrX,0,nmiGradientImage->nvox*nmiGradientImage->nbyper);

    // Set up the multi loop
    Multi_Loop<int> loop;
    for (int i = 0; i < num_loops; ++i) loop.Add(-1, 2);

    SafeArray<int> bins(num_loops);
    for (int i = 0; i < num_target_volumes; ++i) bins[i] = target_bins[i];
    for (int i = 0; i < num_result_volumes; ++i) bins[i + num_target_volumes] = result_bins[i];

    GradTYPE coefficients[20];
    GradTYPE positions[20];
    int relative_positions[20];

    GradTYPE result_common[3];
    GradTYPE der_term[3];

    int index, currentIndex, relative_pos, i, j, lc, jc;
    int target_flat_index, result_flat_index;
    DTYPE temp;
    GradTYPE reg;

    // Loop over all the voxels
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    firstprivate(loop) \
    private(index, valid_values, i, j, lc, jc, voxel_values, currentIndex, temp, \
    result_gradient_x_values, result_gradient_y_values, result_gradient_z_values, \
    jointEntropyDerivative_X, jointEntropyDerivative_Y, jointEntropyDerivative_Z, \
    movingEntropyDerivative_X, movingEntropyDerivative_Y, movingEntropyDerivative_Z, \
    fixedEntropyDerivative_X, fixedEntropyDerivative_Y, fixedEntropyDerivative_Z, \
    target_flat_index, result_flat_index, relative_pos, coefficients, reg, \
    positions, relative_positions, common_target_value, result_common, der_term, \
    jointLog, targetLog, resultLog) \
    shared(targetImagePtr, targetVoxelNumber, mask, num_target_volumes, target_bins, \
    num_result_volumes, result_bins, resultImagePtr, bins, num_loops, num_probabilities, \
    result_offsets, target_offsets, total_target_entries, logJointHistogram, \
    resultImageGradientPtrX, resultImageGradientPtrY, resultImageGradientPtrZ, \
    nmiGradientPtrX, nmiGradientPtrY, nmiGradientPtrZ, NMI, joint_entropy, approx)
#endif // _OPENMP
    for (index = 0; index < targetVoxelNumber; ++index){
        if(mask[index]>-1){
            valid_values = true;
            // Collect the target intensities and do some sanity checking
            for (i = 0; i < num_target_volumes; ++i) {
                voxel_values[i] = targetImagePtr[index+i*targetVoxelNumber];
                if (voxel_values[i] <= (DTYPE)0 ||
                    voxel_values[i] >= (DTYPE)target_bins[i] ||
                    voxel_values[i] != voxel_values[i]) {
                    valid_values = false;
                    break;
                }
//                if(approx) // standard joint histogram filling
                    voxel_values[i] = (DTYPE)static_cast<int>(round(voxel_values[i]));
            }

            // Collect the result intensities and do some sanity checking
            if (valid_values) {
                for (i = 0; i < num_result_volumes; ++i) {
                    currentIndex = index+i*targetVoxelNumber;
                    temp = resultImagePtr[currentIndex];
                    result_gradient_x_values[i] = resultImageGradientPtrX[currentIndex];
                    result_gradient_y_values[i] = resultImageGradientPtrY[currentIndex];
                    result_gradient_z_values[i] = resultImageGradientPtrZ[currentIndex];

                    if (temp <= (DTYPE)0 ||
                        temp >= (DTYPE)result_bins[i] ||
                        temp != temp ||
                        result_gradient_x_values[i] != result_gradient_x_values[i] ||
                        result_gradient_y_values[i] != result_gradient_y_values[i] ||
                        result_gradient_z_values[i] != result_gradient_z_values[i]) {
                        valid_values = false;
                        break;
                    }
//                    if(approx) // standard joint histogram filling
                        voxel_values[num_target_volumes + i] = (DTYPE)static_cast<int>(round(temp));
//                    else voxel_values[num_target_volumes + i] = temp;
                }
            }
            if (valid_values) {
                jointEntropyDerivative_X = 0.0;
                movingEntropyDerivative_X = 0.0;
                fixedEntropyDerivative_X = 0.0;

                jointEntropyDerivative_Y = 0.0;
                movingEntropyDerivative_Y = 0.0;
                fixedEntropyDerivative_Y = 0.0;

                jointEntropyDerivative_Z = 0.0;
                movingEntropyDerivative_Z = 0.0;
                fixedEntropyDerivative_Z = 0.0;

                for (loop.Initialise(); loop.Continue(); loop.Next()) {
                    target_flat_index = result_flat_index = 0;
                    valid_values = true;

                    for(lc = 0; lc < num_target_volumes; ++lc){
                        relative_pos = static_cast<int>(voxel_values[lc] + loop.Index(lc));
                        if(relative_pos< 0 || relative_pos >= bins[lc]){
                            valid_values = false; break;
                        }
                        double common_value = GetBasisSplineValue<double>((double)relative_pos-(double)voxel_values[lc]);
                        coefficients[lc] = common_value;
                        positions[lc] = (GradTYPE)relative_pos-(GradTYPE)voxel_values[lc];
                        relative_positions[lc] = relative_pos;
                    }

                    for(jc = num_target_volumes; jc < num_loops; ++jc){
                        relative_pos = static_cast<int>(voxel_values[jc] + loop.Index(jc));
                        if(relative_pos< 0 || relative_pos >= bins[jc]){
                            valid_values = false; break;
                        }
                        if (num_result_volumes > 1) {
                            double common_value = GetBasisSplineValue<double>((double)relative_pos-(double)voxel_values[jc]);
                            coefficients[jc] = common_value;
                        }
                        positions[jc] = (GradTYPE)relative_pos-(GradTYPE)voxel_values[jc];
                        relative_positions[jc] = relative_pos;
                    }

                    if(valid_values) {
                        common_target_value = (GradTYPE)1.0;
                        for (i = 0; i < num_target_volumes; ++i)
                            common_target_value *= coefficients[i];

                        result_common[0] = result_common[1] = result_common[2] = (GradTYPE)0.0;

                        for (i = 0; i < num_result_volumes; ++i){
                            der_term[0] = der_term[1] = der_term[2] = (GradTYPE)1.0;
                            for (j = 0; j < num_result_volumes; ++j){
                                if (i == j) {
                                    reg = GetBasisSplineDerivativeValue<double>
                                            ((double)positions[j + num_target_volumes]);
                                    der_term[0] *= reg * (GradTYPE)result_gradient_x_values[j];
                                    der_term[1] *= reg * (GradTYPE)result_gradient_y_values[j];
                                    der_term[2] *= reg * (GradTYPE)result_gradient_z_values[j];
                                }
                                else {
                                    der_term[0] *= coefficients[j+num_target_volumes];
                                    der_term[1] *= coefficients[j+num_target_volumes];
                                    der_term[2] *= coefficients[j+num_target_volumes];
                                }
                            }
                            result_common[0] += der_term[0];
                            result_common[1] += der_term[1];
                            result_common[2] += der_term[2];
                        }

                        result_common[0] *= common_target_value;
                        result_common[1] *= common_target_value;
                        result_common[2] *= common_target_value;

                        for (i = 0; i < num_target_volumes; ++i)
                            target_flat_index += relative_positions[i] * target_offsets[i];
                        for (i = 0; i < num_result_volumes; ++i)
                            result_flat_index += relative_positions[i + num_target_volumes] * result_offsets[i];

                        jointLog = logJointHistogram[target_flat_index + (result_flat_index * total_target_entries)];
                        targetLog = logJointHistogram[num_probabilities + target_flat_index];
                        resultLog = logJointHistogram[num_probabilities + total_target_entries + result_flat_index];

                        jointEntropyDerivative_X -= result_common[0] * jointLog;
                        fixedEntropyDerivative_X -= result_common[0] * targetLog;
                        movingEntropyDerivative_X -= result_common[0] * resultLog;

                        jointEntropyDerivative_Y -= result_common[1] * jointLog;
                        fixedEntropyDerivative_Y -= result_common[1] * targetLog;
                        movingEntropyDerivative_Y -= result_common[1] * resultLog;

                        jointEntropyDerivative_Z -= result_common[2] * jointLog;
                        fixedEntropyDerivative_Z -= result_common[2] * targetLog;
                        movingEntropyDerivative_Z -= result_common[2] * resultLog;
                    }

                    nmiGradientPtrX[index] = (GradTYPE)((fixedEntropyDerivative_X + movingEntropyDerivative_X - NMI * jointEntropyDerivative_X) / joint_entropy);
                    nmiGradientPtrY[index] = (GradTYPE)((fixedEntropyDerivative_Y + movingEntropyDerivative_Y - NMI * jointEntropyDerivative_Y) / joint_entropy);
                    nmiGradientPtrZ[index] = (GradTYPE)((fixedEntropyDerivative_Z + movingEntropyDerivative_Z - NMI * jointEntropyDerivative_Z) / joint_entropy);
                }
            }
        }
    }
}
/* *************************************************************** */
template<class DTYPE>
void reg_getVoxelBasedNMIGradientUsingPW1(nifti_image *targetImage,
                                          nifti_image *resultImage,
                                          nifti_image *resultImageGradient,
                                          unsigned int *target_bins,
                                          unsigned int *result_bins,
                                          double *logJointHistogram,
                                          double *entropies,
                                          nifti_image *nmiGradientImage,
                                          int *mask,
                                          bool approx)
{
    if(resultImageGradient->datatype != nmiGradientImage->datatype){
        fprintf(stderr, "[NiftyReg ERROR] reg_getVoxelBasedNMIGradientUsingPW\n");
        fprintf(stderr, "[NiftyReg ERROR] Both gradient images are exepected to have the same type\n");
        exit(1);
    }

    if(nmiGradientImage->nz==1){
        switch(resultImageGradient->datatype){
        case NIFTI_TYPE_FLOAT32:
            reg_getVoxelBasedNMIGradientUsingPW2D<DTYPE,float>
                    (targetImage, resultImage, resultImageGradient, target_bins, result_bins, logJointHistogram,
                     entropies, nmiGradientImage, mask, approx);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_getVoxelBasedNMIGradientUsingPW2D<DTYPE,double>
                    (targetImage, resultImage, resultImageGradient, target_bins, result_bins, logJointHistogram,
                     entropies, nmiGradientImage, mask, approx);
            break;
        default:
            fprintf(stderr,"[NiftyReg ERROR] reg_getVoxelBasedNMIGradientUsingPW\tThe gradient images data type is not supported\n");
            exit(1);
        }
    }
    else{
        switch(resultImageGradient->datatype){
        case NIFTI_TYPE_FLOAT32:
            reg_getVoxelBasedNMIGradientUsingPW3D<DTYPE,float>
                    (targetImage, resultImage, resultImageGradient, target_bins, result_bins, logJointHistogram,
                     entropies, nmiGradientImage, mask, approx);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_getVoxelBasedNMIGradientUsingPW3D<DTYPE,double>
                    (targetImage, resultImage, resultImageGradient, target_bins, result_bins, logJointHistogram,
                     entropies, nmiGradientImage, mask, approx);
            break;
        default:
            fprintf(stderr,"[NiftyReg ERROR] reg_getVoxelBasedNMIGradientUsingPW\tThe gradient images data type is not supported\n");
            exit(1);
        }

    }
}
/* *************************************************************** */
void reg_getVoxelBasedNMIGradientUsingPW(nifti_image *targetImage,
                                         nifti_image *resultImage,
                                         nifti_image *resultImageGradient,
                                         unsigned int *target_bins,
                                         unsigned int *result_bins,
                                         double *logJointHistogram,
                                         double *entropies,
                                         nifti_image *nmiGradientImage,
                                         int *mask,
                                         bool approx)
{
    if(targetImage->datatype != resultImage->datatype){
        fprintf(stderr, "[NiftyReg ERROR] reg_getVoxelBasedNMIGradientUsingPW\n");
        fprintf(stderr, "[NiftyReg ERROR] Both input images are exepected to have the same type\n");
        exit(1);
    }

    switch(targetImage->datatype){
    case NIFTI_TYPE_FLOAT32:
        reg_getVoxelBasedNMIGradientUsingPW1<float>
                (targetImage, resultImage, resultImageGradient, target_bins, result_bins, logJointHistogram,
                 entropies, nmiGradientImage, mask, approx);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_getVoxelBasedNMIGradientUsingPW1<double>
                (targetImage, resultImage, resultImageGradient, target_bins, result_bins, logJointHistogram,
                 entropies, nmiGradientImage, mask, approx);
        break;
    default:
        fprintf(stderr,"[NiftyReg ERROR] reg_getVoxelBasedNMIGradientUsingPW\tThe input image data type is not supported\n");
        exit(1);
    }
}
/* *************************************************************** */
/* *************************************************************** */

#endif
