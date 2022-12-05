/*!@file Image.cpp a class to encapsulate and access image data.
 */

// This file is part of the SaliencyToolbox - Copyright (C) 2006-2008
// by Dirk B. Walther and the California Institute of Technology.
// See the enclosed LICENSE.TXT document for the license agreement. 
// More information about this project is available at: 
// http://www.saliencytoolbox.net

#include "Image.h"
#include "mexLog.h"
#include <cstring>

// ######################################################################
Image::Image()
  : itsInitialized(false)
{}

// ######################################################################
Image::Image(const mxArray *arr)
{
  initialize(arr);
}

// ######################################################################
Image::Image(const Image& other)
{
  initialize(other.getConstArray());
}

// ######################################################################
Image::Image(const int w, const int h)
{
  initialize(w,h);
}
    
// ######################################################################
Image Image::operator=(const Image other)
{
  itsInitialized = other.isInitialized();
  if (isInitialized())
    initialize(other.getConstArray());
  return *this;
}

// ######################################################################
void Image::initialize(const mxArray *arr)
{
  initialize(mxGetN(arr),mxGetM(arr));
  memcpy(beginw(),mxGetPr(arr),getSize()*sizeof(double));
}

// ######################################################################
void Image::initialize(const int w, const int h)
{
  itsInitialized = true;
  itsArray = mxCreateDoubleMatrix(h,w,mxREAL);
}

// ######################################################################
int Image::getWidth() const
{
  ASSERT(isInitialized());
  return mxGetN(getConstArray());
}

// ######################################################################
int Image::getHeight() const
{
  ASSERT(isInitialized());
  return mxGetM(getConstArray());
}

// ######################################################################
int Image::getSize() const
{
  ASSERT(isInitialized());
  return mxGetNumberOfElements(getConstArray());
}

// ######################################################################
Image::const_iterator Image::begin() const
{
  return mxGetPr(getConstArray());
}

// ######################################################################
Image::const_iterator Image::end() const
{
  return (begin() + getSize());
}

// ######################################################################
Image::iterator Image::beginw()
{
  return mxGetPr(getArray());
}

// ######################################################################
Image::iterator Image::endw()
{
  return (beginw() + getSize());
}

// ######################################################################
bool Image::isInitialized() const
{
  return itsInitialized;
}

// ######################################################################
mxArray* Image::getArray()
{
  return itsArray;
}

// ######################################################################
const mxArray* Image::getConstArray() const
{
  return itsArray;
}


// ######################################################################
void Image::clamp(const double bottom, const double top)
{
  ASSERT(isInitialized());
  for (iterator ptr = beginw(); ptr != endw(); ++ptr)
  {
    if (*ptr < bottom) *ptr = bottom;
    if (*ptr > top) *ptr = top;
  }
}
    
// ######################################################################
void Image::clampBottom(const double bottom)
{
  ASSERT(isInitialized());
  for (iterator ptr = beginw(); ptr != endw(); ++ptr)
    if (*ptr < bottom) *ptr = bottom;
}
    
// ######################################################################
void Image::clampTop(const double top)
{
  ASSERT(isInitialized());
  for (iterator ptr = beginw(); ptr != endw(); ++ptr)
    if (*ptr > top) *ptr = top;
}
    
// ######################################################################
Image Image::operator*=(const double factor)
{
  ASSERT(isInitialized());
  for (iterator ptr = beginw(); ptr != endw(); ++ptr)
    *ptr *= factor;
  return *this;
}
  
// ######################################################################
double Image::getVal(int index) const
{
  ASSERT(isInitialized());
  ASSERT(index < getSize());
  return *(begin()+index);
}

// ######################################################################
double Image::getVal(int x, int y) const
{
  ASSERT(coordsOK(x,y));
  const_iterator ptr = begin() + x * getHeight() + y;
  return *ptr;
}

// ######################################################################
void Image::setVal(int index, double val)
{
  ASSERT(isInitialized());
  ASSERT(index < getSize());
  *(beginw()+index) = val;
}
  
// ######################################################################
void Image::setVal(int x, int y, double val)
{
  ASSERT(coordsOK(x,y));
  iterator ptr = beginw() + x * getHeight() + y;
  *ptr = val;
}

// ######################################################################
void Image::getLocalMaxima(const double thresh, int *lm_num, double *lm_sum)
{
  ASSERT(isInitialized());

  const int w = getWidth();
  const int h = getHeight();


  // then get the mean value of the local maxima:
  *lm_sum = 0.0; *lm_num = 0;
  
  for (int j = 1; j < h - 1; j ++)
    for (int i = 1; i < w - 1; i ++)
      {
        double val = getVal(i,j);
        if (val >= thresh &&
            val >= getVal(i-1, j) &&
            val >= getVal(i+1, j) &&
            val >= getVal(i, j+1) &&
            val >= getVal(i, j-1))  // local max
          {
            *lm_sum += val;
            (*lm_num)++;
          }
      }
  return;
}
  
// ######################################################################
bool Image::coordsOK(int x, int y) const
{
  if (!isInitialized()) return false;
  
  return ((x < getWidth()) && (y < getHeight()));
}
