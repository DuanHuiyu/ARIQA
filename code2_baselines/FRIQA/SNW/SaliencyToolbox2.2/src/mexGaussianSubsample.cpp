/*! @file mexGaussianSubsample.cpp 
\verbatim mexGaussianSubsample - smooths and subsamples image.
 result = mexGaussianSubsample(image)
    Convolves image with a 6x6 separable Gaussian kernel
    and subsamples by a factor of two all in one
    integrated operation. 

 See also makeDyadicPyramid, makeSqrt2Pyramid, makeGaussianPyramid.
\endverbatim */

// This file is part of the SaliencyToolbox - Copyright (C) 2006-2008
// by Dirk B. Walther and the California Institute of Technology.
// See the enclosed LICENSE.TXT document for the license agreement. 
// More information about this project is available at: 
// http://www.saliencytoolbox.net

#include "mexLog.h"
#include "Image.h"
#include "Filters.h"
#include "MexParams.h"

/* the main program */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  MexParams Params(0,1,nlhs,plhs,1,1,nrhs,prhs);
  const Image img(Params.getInput(0));
  Image result = lowPass6yDecY(lowPass6xDecX(img));
  Params.setOutput(0,result.getArray());
}
