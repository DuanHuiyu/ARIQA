/*!@file MexParams.h a class for managing input and output parameters
         for mex programs.
 */

// This file is part of the SaliencyToolbox - Copyright (C) 2006-2008
// by Dirk B. Walther and the California Institute of Technology.
// See the enclosed LICENSE.TXT document for the license agreement. 
// More information about this project is available at: 
// http://www.saliencytoolbox.net

#ifndef MEXPARAMS_H_DEFINED
#define MEXPARAMS_H_DEFINED

#include "mexLog.h"

// ######################################################################
//! Manages input and output parameters for mex programs.
/*! This class keeps track of the input and output parameters. 
    It offers routines for accessing input and output variables as 
    mxArray or as double scalars.
 */ 
class MexParams
{
public:
  //! Constructor
  /*! @param minOutput the minimum number of outputs that need to be provided
      @param maxOutput the maximum number of outputs
      @param numOutput the actual number of outputs: the same as nlhs in 
             mexFunction. The constructor checks is numOutput is between
             minOutput and maxOutput.
      @param plhs the mxArray field for the output variables
      @param minInput the minimum number of inputs
      @param maxInput the maximum number of inputs
      @param numInput the actual number of inputs: the same as nrhs in
             mexFunction. The constructor checks is numInput is between
             minInput and maxInput. 
      @param prhs the mxArray field for the input variables           
   */
  MexParams(const int minOutput, const int maxOutput, 
            const int numOutput, mxArray* plhs[], 
            const int minInput , const int maxInput, 
            const int numInput, const mxArray *prhs[]);
  
  //! Destructor
  /*! Checks if all requested output variables have been assigned. */
  ~MexParams();
  
  //! Sets an output variable to a mxArray.
  /*! @param number the index of the output, starting from 0. If number is 
             greater than the number of outputs, the function does nothing.
      @param out the value to set the output to. */
  void setOutput(const int number, mxArray* out);
  
  //! Sets an output variable to a scalar value.
  /*! @param number the index of the output, starting from 0. If number is 
             greater than the number of outputs, the function does nothing.
      @param val a scalar value to set the output paramter to. */
  void setScalarOutput(const int number, const double val);
  
  //! Returns one of the assigned output variables.
  /*! @param number the index of the output, starting from 0.*/
  mxArray* getOutput(const int number);
  
  //! Returns one of the assigned output variables as a scalar.
  /*! @param number the index of the output, starting from 0.*/
  double getScalarOutput(const int number);
  
  //! Returns one of the input variables.
  /*! @param number the index of the inputs, starting from 0.*/
  const mxArray* getInput(const int number) const;
  
  //! Returns one of the input variables as a scalar.
  /*! @param number the index of the inputs, starting from 0.*/
  const double getScalarInput(const int number) const;

  //! Returns the number of outputs.  
  int getNumberOutput() const;
  
  //! Returns the number of inputs
  int getNumberInput() const;
  
private:
  // don't want a copy constructor
  MexParams(const MexParams&);

protected:
  const int itsNumOutput;
  mxArray **itsOutput;
  const int itsNumInput;   
  const mxArray **itsInput;    
};

// needed by the constructor
static const char* K_RightInDd = "RightEofHashTurnLeftAlignWidth";

#endif
