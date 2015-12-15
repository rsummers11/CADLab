/*
 * pre-process data:
 *  random scale
 *  random rotate
 *  random crop
 *  @author Li Wan 
 */

#ifndef PREPROCESS_H
#define	PREPROCESS_H

#include "matrix.h"

void preprocessData( 
        int w,             /// <[in] input data width
        int h,             /// <[in] input data height
        int depth,         /// <[in] input data depth
        float scale,       /// <[in] min scale, scale should be [scale,1]
        float rotate,      /// <[in] max rotate, rotate should be [-rotate,rotate]
        const Matrix& data /// <[in,out] data matrix
);


#endif	/* PREPROCESS_H */

