/*!@file Filters.cpp various filter functions.
 */

// This file is part of the SaliencyToolbox - Copyright (C) 2006-2008
// by Dirk B. Walther and the California Institute of Technology.
// See the enclosed LICENSE.TXT document for the license agreement. 
// More information about this project is available at: 
// http://www.saliencytoolbox.net

#include "mexLog.h"
#include "Image.h"
#include "Filters.h"

#include <algorithm>
#include <cmath>
#include <limits>

// ######################################################################
// kernel: 1 5 10 10 5 1
Image lowPass6yDecY(const Image& src)
{
  ASSERT(ecxStr);
  const int w = src.getWidth(), hs = src.getHeight();
  const float ecw = ecx / w;
  int hr = hs / 2;
  if (hr == 0) hr = 1;
  
  Image result(w,hr);
  Image::iterator rptr = result.beginw();
  Image::const_iterator sptr = src.begin();
  
  if (hs <= 1)
    result = src;
  else if (hs == 2)
    for (int x = 0; x < w; ++x)
      {
        // use kernel [1 1]^T / 2
        *rptr++ = (sptr[0] + sptr[1]) / 2.0;
        sptr += 2;
      }
  else if (hs == 3)
    for (int x = 0; x < w; ++x)
      {
        // use kernel [1 2 1]^T / 4
        *rptr++ = (sptr[0] + sptr[1] * 2.0 + sptr[2]) / 4.0;
        sptr += 3;
      }
  else // general case with hs >= 4
    for (int x = 0; x < w; ++x)
      {
        // top most point - use kernel [10 10 5 1]^T / 26
        *rptr++ = ((sptr[0] + sptr[1]) * 10.0 + 
                    sptr[2] * 5.0 + sptr[3]) / 26.0;
        //++sptr;
        
        // general case
        int y;
        for (y = 0; y < (hs - 5); y += 2)
          {
            // use kernel [1 5 10 10 5 1]^T / 32
            *rptr++ = ((sptr[1] + sptr[4])  *  5.0 +
                       (sptr[2] + sptr[3])  * 10.0 +
                       (sptr[0] + sptr[5])) / 32.0;
            sptr += 2;
          }
        
        // find out how to treat the bottom most point
        if (y == (hs - 5))
          {
            // use kernel [1 5 10 10 5]^T / 31
             *rptr++ = ((sptr[1] + sptr[4])  *  5.0 +
                        (sptr[2] + sptr[3])  * 10.0 +
                         sptr[0])            / 31.0;
            sptr += 5;
          }
        else
          {
            // use kernel [1 5 10 10]^T / 26
            *rptr++ = ( sptr[0] + sptr[1]  *  5.0 +
                       (sptr[2] + sptr[3]) * 10.0) / 26.0;
            sptr += 4;
          }
        }
            
   return result;      
}

// ######################################################################
// kernel: 1 5 10 10 5 1
Image lowPass6xDecX(const Image& src)
{
  ASSERT(ecxStr);
  const int ws = src.getWidth(), h = src.getHeight();
  const float ecw = ecx / ws;
  const int h2 = h * 2, h3 = h * 3, h4 = h * 4, h5 = h * 5;
  int wr = ws / 2;
  if (wr == 0) wr = 1;
  
  Image result(wr,h);
  Image::iterator rptr = result.beginw();
  Image::const_iterator sptr = src.begin();

  if (ws <= 1)
    result = src;
  else if (ws == 2)
    for (int y = 0; y < h; ++y)
      {
        // use kernel [1 1] / 2
        *rptr++ = (sptr[0] + sptr[h]) / 2.0;
        ++sptr;
      }
  else if (ws == 3)
    for (int y = 0; y < h; ++y)
      {
        // use kernel [1 2 1] / 4
        *rptr++ = (sptr[0] + sptr[h] * 2.0 + sptr[h2]) / 4.0;
        ++sptr;
      }
  else // general case for ws >= 4
    {
      // left most point - use kernel [10 10 5 1] / 26
      for (int y = 0; y < h; ++y)
        {
          *rptr++ = ((sptr[0] + sptr[h]) * 10.0 + 
                      sptr[h2] * 5.0 + sptr[h3]) / 26.0;
          ++sptr;
        }
      sptr -= h;
      
      // general case
      int x;
      for (x = 0; x < (ws - 5); x += 2)
        {
          for (int y = 0; y < h; ++y)
            {
              // use kernel [1 5 10 10 5 1] / 32
              *rptr++ = ((sptr[h]  + sptr[h4])  *  5.0 +
                         (sptr[h2] + sptr[h3])  * 10.0 +
                         (sptr[0]  + sptr[h5])) / 32.0;
              ++sptr;
            }
          sptr += h;
        }
        
      // find out how to treat the right most point
      if (x == (ws - 5))
        for (int y = 0; y < h; ++y)
          {
            // use kernel [1 5 10 10 5] / 31
            *rptr++ = ((sptr[h]  + sptr[h4])  *  5.0 +
                       (sptr[h2] + sptr[h3])  * 10.0 +
                        sptr[0]) / 31.0;
            ++sptr;
          }
      else
        for (int y = 0; y < h; ++y)
          {
            // use kernel [1 5 10 10] / 26
            *rptr++ = ( sptr[0]  + sptr[h]   * 5.0 + 
                       (sptr[h2] + sptr[h3]) * 10.0) / 26.0;
            ++sptr;
          }
    }
  return result;
}

// ######################################################################
Image conv2PreserveEnergy(const Image& src, const Image& f)
{
  ASSERT(src.isInitialized());
  ASSERT(ecxStr);

  const int sw = src.getWidth();
  const int sh = src.getHeight();
  const float ecw = ecx / sw;

  Image::const_iterator filter = f.begin();
  const int fw = f.getWidth();
  const int fh = f.getHeight();

  ASSERT((fw & 1) && (fh & 1));

  Image result(sw, sh);
  Image::const_iterator sptr = src.begin();
  Image::iterator rptr = result.beginw();

  const int fend = fw * fh - 1;
  const int fw2 = (fw - 1) / 2;
  const int fh2 = (fh - 1) / 2;

  const int sSkip = sh - fh;

  for (int rx = 0; rx < sw; ++rx)
    {
      // Determine if we're safely inside the image in the x-direction:
      const bool isXclean = ((rx >= fw2) && (rx <  (sw - fw2)));
      for (int ry = 0; ry < sh; ++ry, ++rptr)
        {
          // Determine if we're safely inside the image in the y-direction:
          const bool isYclean = ((ry >= fh2) && (ry <  (sh - fh2)));
          
          if (isXclean && isYclean)
            {
              // well inside the image: use straight-forward convolution
              float rval = 0.0f;
              Image::const_iterator fptr = filter+fend;
              Image::const_iterator sptr2 = sptr + sh*(rx-fw2) + ry - fh2;

              for (int fx = 0; fx < fw; ++fx)
                {
                  for (int fy = 0; fy < fh; ++fy)
                    rval += (*sptr2++) * (*fptr--);

                  sptr2 += sSkip;
                }
              *rptr = rval;
              continue;
            }
          else
            {
              // near the border: assume that the value missing pixels
              // is equal to the average over the present pixels
              // to minimize artifacts at the edge.

              float rval = 0.0f;
              float sSum = 0.0f;
              int sCount = 0;
              float fsum_skipped = 0.0f;

              for (int fx = 0; fx < fw; ++fx)
                {
                  const int sx = rx + fx - fw2;
                  if (sx >= 0 && sx < sw)
                    {
                      for (int fy = 0; fy < fh; ++fy)
                        {
                          const float fil = filter[fend - fx * fh - fy];
                          const int sy = ry + fy - fh2;
                          if (sy >= 0 && sy < sh)
                            {
                              // here we have the pixel present: 
                              // store it and count it to compute the mean
                              const float sVal = sptr[sx * sh + sy];
                              rval += sVal * fil;
                              sSum += sVal;
                              ++sCount;
                            }
                          else
                            {
                              // this is accumulataing the filter values 
                              // over missing pixels
                              fsum_skipped += fil;
                            }
                        }
                    }
                  else
                    {
                      // this is accumulating the filter values
                      // over an entire column of mixxing pixels
                      for (int fy = 0; fy < fh; ++fy)
                        fsum_skipped += filter[fend - fx * fh - fy];
                    }
                }
              // compute the average over the present pixels
              const float sAvg = sSum / sCount;
              
              // add the that average x accumulated filter values to the result
              *rptr = rval + (fsum_skipped * sAvg);
            }
        }
    }
  return result;
}
