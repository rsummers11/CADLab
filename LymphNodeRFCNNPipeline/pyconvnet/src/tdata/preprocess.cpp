/*
 * pre-process data:
 *  random scale
 *  random rotate
 *  random crop
 *  @author Li Wan 
 */

#include "preprocess.hpp"

#define USE_OPEN_MP
//#define DEBUG_PREPROCESS_CPP

#ifdef USE_OPEN_MP
#include <omp.h>
#endif

#include <cassert>
#include <cstdlib>
#include <iostream>

#include "cimg_extension.h"

using namespace std;
using namespace cimg_library;

// rotate & scale image, 
// if image is smaller than origional image, it will be filled by 
// origional image back ground
// if image is large, simple cropped image will return
void rotate_scale_image( CImg<float>& image, float rotate, float scale ){
    assert( scale >=0 && scale <=1 );
    int old_w = image.width();
    int old_h = image.height();
    float random_number;
    // get rotate degree
    random_number = 2*(static_cast<float>( rand() ) / RAND_MAX - 0.5); //[-1,1]
    float rotate_degree = rotate * random_number;
#ifdef DEBUG_PREPROCESS_CPP
    cout << "Rotate Degree: " << rotate_degree; 
#endif
    // get scale factor
    random_number = 2*(static_cast<float>( rand() ) / RAND_MAX - 0.5);
    float scale_factor = 1.0 + scale * random_number; 
#ifdef DEBUG_PREPROCESS_CPP
    cout << "   Scale Factor: " << scale_factor << endl;
#endif
    // rotate
    image.rotate( rotate_degree, 1 );
    // scale
    int w = int( scale_factor * image.width() + 0.5f );
    int h = int( scale_factor * image.height() + 0.5f );
    image.resize( w, h, -100, -100, 1, 1 );

    // crop image based on old w&h
    int x0 = (w - old_w)/2;
    int x1 = x0 + old_w - 1;
    int y0 = (h - old_h)/2;
    int y1 = y0 + old_h - 1;
    image.crop( x0, y0, 0, 0, x1, y1, 0, 2, true );
    assert( image.width() == old_w && image.height() == old_h );
}

void preprocessData( 
        int w,             /// <[in] input data width
        int h,             /// <[in] input data height
        int d,             /// <[in] input data depth
        float scale,       /// <[in] min scale, scale should be [scale,1]
        float rotate,      /// <[in] max rotate, rotate should be [-rotate,rotate]
        const Matrix& data /// <[in,out] data matrix
){
    // pre-condition check
    assert( w == h );
    assert( data.getNumRows() == w*h*d );
    int numImage = data.getNumCols();

#ifdef USE_OPEN_MP 
    omp_set_num_threads(2);
#endif
#pragma omp parallel for
    for( int imageIndex = 0; imageIndex < numImage; imageIndex++ ) {
        // construct cimg
        CImg<float> image( w, h,1, d, 0 );
        for( int k = 0; k < d; k++ ) 
            for( int j = 0; j < h; j++ )
                for( int  i = 0; i < w; i++ ) {
                    int index = k * w* h+ j * w+ i;
                    image( i, j, 0, k) = data( index , imageIndex );
                }
#ifdef DEBUG_PREPROCESS_CPP
        image.display();
        CImgList<float> image_list;
        // process data
        for( int i = 0; i < 10; i++ ){
            CImg<float> im = image;
            rotate_scale_image( im, rotate, scale );
            image_list.push_back( im );
        }
        horizontalCatCImgList( image_list ).display();
#else
        rotate_scale_image( image, rotate, scale );
#endif

        // copy back
        for( int k = 0; k < d; k++ ) 
            for( int j = 0; j < h; j++ )
                for( int  i = 0; i < w; i++ ) {
                    int index = k * w* h+ j * w+ i;
                    data( index , imageIndex ) = image( i, j, 0, k) ;
                }
    }
}
