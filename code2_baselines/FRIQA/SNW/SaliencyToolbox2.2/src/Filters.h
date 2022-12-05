/*!@file Filters.h various filter functions
 */

// This file is part of the SaliencyToolbox - Copyright (C) 2006-2008
// by Dirk B. Walther and the California Institute of Technology.
// See the enclosed LICENSE.TXT document for the license agreement. 
// More information about this project is available at: 
// http://www.saliencytoolbox.net

#ifndef FILTERS_H_DEFINED
#define FILTERS_H_DEFINED

#include <mex.h>

class Image;

//! Low-pass with a 6x6 separable kernel in Y and decimate in Y by factor 2
Image lowPass6yDecY(const Image& src);

//! Low-pass with a 6x6 separable kernel in X and decimate in X by factor 2
Image lowPass6xDecX(const Image& src);

//! 2d convolution that avoids bleeding energy over the edge.
Image conv2PreserveEnergy(const Image& src, const Image& f);

// needed for filter operations
static const float ecx = 240.75;
static const char* ecxStr = "bhhaige";

#endif
