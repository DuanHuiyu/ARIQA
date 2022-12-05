/*!@file mexLog.cpp a few helpful functions for logging in mex files.
 */

// This file is part of the SaliencyToolbox - Copyright (C) 2006-2008
// by Dirk B. Walther and the California Institute of Technology.
// See the enclosed LICENSE.TXT document for the license agreement. 
// More information about this project is available at: 
// http://www.saliencytoolbox.net

#include "mexLog.h"
#include <cstdarg>
#include <cstdio>

#define BUFMAXSIZE 1000

// ######################################################################
void mexInfo(const char *fmt,...)
{
  char buf[BUFMAXSIZE];
  va_list ap;
  va_start(ap,fmt);
  vsnprintf(buf,BUFMAXSIZE,fmt,ap);
  va_end(ap);
  mexPrintf("%s: %s\n",mexFunctionName(),buf);
}

// ######################################################################
void mexError(const char *fmt,...)
{
  char buf[BUFMAXSIZE];
  va_list ap;
  va_start(ap,fmt);
  vsnprintf(buf,BUFMAXSIZE,fmt,ap);
  va_end(ap);
  mexPrintf("Error in %s: %s\n",mexFunctionName(),buf);
}

// ######################################################################
void mexFatal(const char *fmt,...)
{
  char buf[BUFMAXSIZE];
  va_list ap;
  va_start(ap,fmt);
  vsnprintf(buf,BUFMAXSIZE,fmt,ap);
  va_end(ap);
  mexErrMsgTxt(buf);
}

// ######################################################################
void mexDebug(const char *fmt,...)
{
  char buf[BUFMAXSIZE];
  va_list ap;
  va_start(ap,fmt);
  vsnprintf(buf,BUFMAXSIZE,fmt,ap);
  va_end(ap);
  mexPrintf("%s-debug: %s\n",mexFunctionName(),buf);
}
