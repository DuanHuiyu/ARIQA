/*!@file Image.h a class to encapsulate and access image data
 */

// This file is part of the SaliencyToolbox - Copyright (C) 2006-2008
// by Dirk B. Walther and the California Institute of Technology.
// See the enclosed LICENSE.TXT document for the license agreement. 
// More information about this project is available at: 
// http://www.saliencytoolbox.net

#ifndef IMAGE_H_DEFINED
#define IMAGE_H_DEFINED

#include <mex.h>

// ######################################################################
//! Provides access and processing functions for an image
/*! An Image is an object-oriented abstraction of a two-dimensional
    Matlab array. The pixels are double values. */
class Image
{
public:
    
  // ######################################################################
  /*! @name Constructors and initialization routines
      An Image can be constructed from an existing two-dimensional Matlab
      array, or created empty, or constructed uninitialzed. In this case,
      the Image must be initialized before any operation with it is 
      attempted. The copy constructor and assigment make true memory
      copies of the other image */
  //@{
   
  //! Contructor for an uninitialized Image.
  Image();
  
  //! Constructor from a two-dimensional Matlab array.
  Image(const mxArray *arr);
  
  //! Copy constructor.
  Image(const Image& other);
  
  //! Constructs a new empty Image with width w and height h.
  Image(const int w, const int h);
  
  //! Copy assigment operator.
  Image operator=(const Image other);
   
  //! Initializes from a two-dimensional Matlab array.
  void initialize(const mxArray *arr);
  
  //! Initializes a new empty image with width w and height h.
  void initialize(const int w, const int h);
  
  //@}
  // ######################################################################
  /*! @name Access functions. */
  //@{
  
  //! Returns the width of the Image.
  int getWidth() const;
  
  //! Returns the height of the Image.
  int getHeight() const;
  
  //! Returns the number of pixels in the Image.
  int getSize() const;
  
  //! Returns whether the Image is initialized.
  bool isInitialized() const;
  
  //! Returns a pointer to the underlying Matlab array.
  /*! This version allows manipulation of the array from the outside.*/
  mxArray* getArray();
  
  //! Returns a constant pointer to the underlying Matlab array.
  /*! This version prevents manipulation of the array from the outside.*/
  const mxArray* getConstArray() const;
  
  //! Check if the coordinates are within the bounds of the Image.
  bool coordsOK(int x, int y) const;

  //! Returns a pixel value by index.
  double getVal(int index) const;
  
  //! Returns a pixel value by coordinates.
  double getVal(int x, int y) const;
  
  //! Sets a pixel value by index.
  void setVal(int index, double val);
  
  //! Sets a pixel value by coordinates.
  void setVal(int x, int y, double val);

  //@}
  // ######################################################################
  /*! @name Computation functions. */
  //@{
  
  //! Clamp the image between bottom and top.
  /*! All values less than bottom are set to bottom; all values greater
      than top are set to top.*/
  void clamp(const double bottom, const double top);
  
  //! Clamp the image only at the bottom.
  /*! All values less than bottom are set to bottom.*/
  void clampBottom(const double bottom);
  
  //! Clamp the image only at the top.
  /*! All values greater than top are set to top.*/
  void clampTop(const double top);
  
  //! Count and sum all local maxima above a threshold.
  /*!@param thresh the threshold for a local maximum.
     @param lm_num returns the number of local maxima.
     @param lm_sum returns the sum of the local maxima. */
  void getLocalMaxima(const double thresh, int *lm_num, double *lm_sum);
  
  //! Multiply all values by a scalar factor.
  Image operator*=(const double factor);
  
  //@}
  // #####################################################################
  /*! @name Iterators
      There are const and non-const versions of iterators, which are
      returned by begin()/end() and beginw()/endw(), respectively. The "w"
      in beginw()/endw() is a mnemonic for "write" or "writeable".
   */
  //@{
  
  //! Read/write iterator.
  typedef double* iterator;
  
  //! Read-only iterator.
  typedef const double* const_iterator;
  
  //! Returns a read-only iterator to the beginning of the Image data
  const_iterator begin() const;
  
  //! Returns a read-only iterator to the one-past-the-end of the Image data
  const_iterator end() const;
  
  //! Returns a read/write iterator to the beginning of the Image data
  iterator beginw();
  
  //! Returns a read/write iterator to the one-past-the-end of the Image data
  iterator endw();
  
  //@}
  
protected:
  bool itsInitialized;
  mxArray *itsArray;  
};

#endif
