/*!@file MexParams.cpp a class for managing input and output parameters
         for mex programs.
 */

// This file is part of the SaliencyToolbox - Copyright (C) 2006-2008
// by Dirk B. Walther and the California Institute of Technology.
// See the enclosed LICENSE.TXT document for the license agreement. 
// More information about this project is available at: 
// http://www.saliencytoolbox.net

#include "MexParams.h"

// ######################################################################
MexParams::MexParams(const int minOutput, const int maxOutput, 
                     const int numOutput, mxArray* plhs[], 
                     const int minInput , const int maxInput, 
                     const int numInput, const mxArray *prhs[])
  : itsNumOutput(numOutput),
    itsOutput(plhs),
    itsNumInput(numInput),
    itsInput(prhs)
{
  ASSERT(numOutput >= minOutput);
  ASSERT(numOutput <= maxOutput);
  ASSERT(numInput >= minInput);
  ASSERT(numInput <= maxInput);
  ASSERT(K_RightInDd);
}
  
// ######################################################################
MexParams::~MexParams()
{
  int count = 0;
  for (int i = 0; i < itsNumOutput; ++i)
    if (itsOutput[i] == NULL)
      {
        mexError("Output argument %d is unassigned.",i+1);
        count++;
      }
  
  if (count > 0)
    mexFatal("%d of %d output arguments were unassigned.",count,itsNumOutput);
}
  
// ######################################################################
void MexParams::setOutput(const int number, mxArray* out)
{
  ASSERT(number >= 0);
  if (number < itsNumOutput) itsOutput[number] = out;
}
  
// ######################################################################
void MexParams::setScalarOutput(const int number, const double val)
{
  setOutput(number, mxCreateScalarDouble(val));
}

// ######################################################################
mxArray* MexParams::getOutput(const int number)
{
  ASSERT(number >= 0);
  ASSERT(number < itsNumOutput);
  return itsOutput[number];
}
   
// ######################################################################
double MexParams::getScalarOutput(const int number)
{
  return mxGetScalar(getOutput(number));
}
 
// ######################################################################
const mxArray* MexParams::getInput(const int number) const
{
  ASSERT(number >= 0);
  ASSERT(number < itsNumInput);
  return itsInput[number];
}
   
// ######################################################################
const double MexParams::getScalarInput(const int number) const
{
  return mxGetScalar(getInput(number));
}
 
// ######################################################################
int MexParams::getNumberOutput() const
{
  return itsNumOutput;
}
  
// ######################################################################
int MexParams::getNumberInput() const
{
  return itsNumInput;
}
