/**
 * CImg library with some simple function extension
 *
 * @authro Li Wan (NYU)
 */

#ifndef __CIMG_EXTENSION_H__
#define __CIMG_EXTENSION_H__

#include "CImg.h"
#include <string>
#include <algorithm>
#include <cassert>
#include <vector>
#include <boost/lexical_cast.hpp>

namespace cimg_library{
   template<class T>
   cimg_library::CImg<T> horizontalCatCImgList( const cimg_library::CImgList<T>& list ) {
      using namespace cimg_library;
      // collect width and height information
      int width = 0;
      int height = 0;
      int depth = 0;
      typedef typename CImgList<T>::const_iterator  const_itr_t;
      for( const_itr_t itr = list.begin(); itr < list.end(); ++itr ) {
         const CImg<T>& image = (*itr);
         width += image.width();
         height = std::max( image.height(), height );
         depth = image.spectrum();
      }
      assert( depth == 1 || depth == 3 );
      // allocate new image
      CImg<T> im( width, height, 1, depth, 0 );
      // copy image data
      int currentWidthStart = 0;
      for( const_itr_t itr = list.begin(); itr < list.end(); ++itr ) {
         const CImg<T>& image = (*itr);
         // copy data
         for( int k = 0; k < depth; k++ )
            for( int j = 0; j < image.height(); j++ )
               for( int i = 0; i < image.width(); i++ ) {
                  int dstWidth = i + currentWidthStart;
                  im( dstWidth, j, 0, k ) = image( i, j, 0, k );
               }
         // inc width start
         currentWidthStart += image.width();
      }
      // return
      return im;
   }

   template<class T>
   cimg_library::CImg<T> verticalCatCImgList( const cimg_library::CImgList<T>& list ) {
      using namespace cimg_library;
      // collect width and height information
      int width = 0;
      int height = 0;
      int depth = 0;
      typedef typename CImgList<T>::const_iterator  const_itr_t;
      for( const_itr_t itr = list.begin(); itr < list.end(); ++itr ) {
         const CImg<T>& image = (*itr);
         width = max( image.width(), width );
         height += image.height();
         depth = image.spectrum();
      }
      assert( depth == 1 || depth == 3 );
      // allocate new image
      CImg<T> im( width, height, 1, depth, 0 );
      // copy image data
      int currentHeightStart = 0;
      for( const_itr_t itr = list.begin(); itr < list.end(); ++itr ) {
         const CImg<T>& image = (*itr);
         // copy data
         for( int k = 0; k < depth; k++ )
            for( int j = 0; j < image.height(); j++ )
               for( int i = 0; i < image.width(); i++ ) {
                  int dstHeight = j + currentHeightStart;
                  im( i, dstHeight, 0, k ) = image( i, j, 0, k );
               }
         // inc width start
         currentHeightStart += image.height();
      }
      // return
      return im;
   }

   const unsigned char PEN_COLOR_GREEN[] = {0, 255,0};
   const unsigned char PEN_COLOR_BLUE[] = {0, 0, 255};
   const unsigned char PEN_COLOR_RED[] = {255,0,0};
   const unsigned char PEN_COLOR_WHITE[] = {255,255,255};
   const unsigned char PEN_COLOR_BLACK[] = {0,0,0};
   const unsigned char PEN_COLOR_BROWN[] = { 255,165,0 };
   const unsigned char PEN_COLOR_YELLOW[] = { 255,255,0 };
   const unsigned char PEN_COLOR_GOLD[] = { 255,215,0 };

   /**
    * Draw rectange in a image: (thick = 3)
    * @param[in,out] img     CImg object
    * @param[in]     left    min x
    * @param[in]     top     min y
    * @param[in]     right   max x
    * @param[in]     bottom  max y
    * @param[in]     color   pen color
    */
   template<class T>
   void drawRectangle(cimg_library::CImg<T>& img, int left, int top, 
         int right, int bottom, const unsigned char color[]) {
      img.draw_rectangle(left-1, top-1, right+1, bottom+1, color, 1.0f,(~0U));
      img.draw_rectangle(left, top, right, bottom, color, 1.0f,(~0U));
      img.draw_rectangle(left+1, top+1, right-1, bottom-1, color, 1.0f,(~0U));
   }

   /**
    * draw text on left top conor image
    * @param[in,out] img     CImg object
    * @parma[in]     text    text to draw on image
    * @param[in]     color   pen color
    */
   template<class T>
   void drawText( cimg_library::CImg<T>& img, const std::string& s, const unsigned char color[] ) {
         img.draw_text( 2, 2, s.c_str(), color, PEN_COLOR_WHITE, 5 );
   }

   /**
    * draw vector as text on left top conor image
    * @param[in,out] img     CImg object
    * @parma[in]     data    vector of data
    * @param[in]     color   pen color
    */
   template<class T, class U>
   void drawText( cimg_library::CImg<T>& img, const std::vector<U>& v, const unsigned char color[] ) {
      if( v.empty() )
         return;
      std::string text = "[";
      for( int i = 0; i < v.size(); i++ ) {
         text += boost::lexical_cast<std::string>( v[i] );
         if( i != v.size() -1 )
            text += ", ";
      }
      text += "]";
      drawText( img, text, color );
   }

   /**
    * draw line on image
    * @param[in,out] img     CImg object
    * @param[in] x0  x-coordinate of starting line point
    * @param[in] y0  y-coordinate of starting line point
    * @param[in] x1  x-coordinate of ending line point
    * @param[in] y1  y-coordinate of ending line point
    * @param[in] color color of line
    * @param[in] drawTwoEndPoints whehter draw starting and ending point with outlined circle
    * @param[in] raidus  end point radius
    */
   template<class T>
   void drawLine( cimg_library::CImg<T>& img, 
         const int x0, const int y0, 
         const int x1, const int y1, 
         const unsigned char line_color[], 
         const bool drawTwoEndPoints, 
         const unsigned char point_color[],
         const int radius = 3 ){
      // draw outlined circle at staring point
      if( drawTwoEndPoints ) {
         //img.draw_circle( x0, y0, radius, point_color, 1, 1 );
         img.draw_circle( x0, y0, radius, point_color );
      }

      // draw line
      img.draw_line( x0, y0, x1, y1, line_color );

      // draw outlined circle at ending point
      if( drawTwoEndPoints ) {
         //img.draw_circle( x1, y1, radius, point_color, 1, 1 );
         img.draw_circle( x1, y1, radius, point_color );
      }
      
   }
}
#endif
