/*! @file mexConv2PreserveEnergy.cpp 
\verbatim mexConv2PreserveEnergy - 2d convolution that avoids bleeding energy over the edge.
 result = mexConv2PreserveEnergy(data,filter)
    Convolves data with the 2d (non-separable) filter.
    At the boundary, the value of missing pixels is assumed
    to be equal to the mean over the present pixels
    to avoid border artefacts.
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
  MexParams Params(0,1,nlhs,plhs,2,2,nrhs,prhs);
  const Image img(Params.getInput(0));
  const Image filt(Params.getInput(1));
  Image result = conv2PreserveEnergy(img,filt);
  Params.setOutput(0,result.getArray());
}
