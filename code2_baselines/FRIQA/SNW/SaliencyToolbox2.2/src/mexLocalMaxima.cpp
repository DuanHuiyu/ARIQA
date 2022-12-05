/*! @file mexLocalMaxima.cpp 
\verbatim mexLocalMaxima - returns statistics over local maxima.
 [lm_avg,lm_num,lm_sum] = mexLocalMaxima(data,thresh)
    Returns the average value (lm_avg), the number (lm_num),
    and the sum (lm_sum) of local maxima in data that exceed thresh.

 See also maxNormalizeStd.
\endverbatim */

// This file is part of the SaliencyToolbox - Copyright (C) 2006-2008
// by Dirk B. Walther and the California Institute of Technology.
// See the enclosed LICENSE.TXT document for the license agreement. 
// More information about this project is available at: 
// http://www.saliencytoolbox.net

#include "mexLog.h"
#include "Image.h"
#include "MexParams.h"

/* the main program */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  MexParams Params(0,3,nlhs,plhs,2,2,nrhs,prhs);
  Image img(Params.getInput(0));
  double thresh = Params.getScalarInput(1);
  
  int lm_num;
  double lm_sum, lm_avg;
  img.getLocalMaxima(thresh,&lm_num,&lm_sum);
  
  if (lm_sum > 0) lm_avg = lm_sum / (double)lm_num;
  else lm_avg = 0.0;
  
  Params.setScalarOutput(0,lm_avg);
  Params.setScalarOutput(1,(double)lm_num);
  Params.setScalarOutput(2,lm_sum);
}
