/*!@file mexLog.h a few helpful functions for logging in mex files.
 */

// This file is part of the SaliencyToolbox - Copyright (C) 2006-2008
// by Dirk B. Walther and the California Institute of Technology.
// See the enclosed LICENSE.TXT document for the license agreement. 
// More information about this project is available at: 
// http://www.saliencytoolbox.net

#ifndef MEXLOG_H_DEFINED
#define MEXLOG_H_DEFINED

#include <streambuf>
#include <iostream>
#include <mex.h>

#define ASSERT(exp) if (!(exp)) mexFatal("ASSERT failed in %s at line %d: " \
                                         #exp,__FILE__,__LINE__);

//! writes an info message to mexPrintf
void mexInfo(const char *fmt,...);

//! writes an error message
void mexError(const char *fmt,...);

//! writes an error message and leaved the mex file
void mexFatal(const char *fmt,...);

//! writes a debug message
void mexDebug(const char *fmt,...);

// ######################################################################
//! This streambuf is used to re-direct cout and cerr
class MexBuf : public std::streambuf
{
 public:
  MexBuf()  : std::streambuf() 
  { setbuf((char*)0,0); }
  
  virtual int overflow(int c)
  {
    if (c != EOF)
      mexPrintf("%c", c);
    return c;
  }

  virtual std::streamsize xsputn(const char* s, const std::streamsize n)
  {
    std::streamsize c = n;
    while (*s && c--)
      mexPrintf("%c", *s++);
    return n;
  }

};

// ######################################################################
//! This class triggers the redirection of the standard streams at construction
class MexBufInit
{
 public:
  MexBufInit(MexBuf& buf)  
  {
    std::cout.rdbuf(&buf);
    std::cerr.rdbuf(&buf);
  }

  ~MexBufInit() {}

 private:
  MexBufInit(const MexBufInit&);
  MexBufInit& operator=(const MexBufInit&);
};

// ######################################################################
namespace
{
  static MexBuf mexbuf__;
  static MexBufInit mexbufInit__(mexbuf__);
}

#endif
