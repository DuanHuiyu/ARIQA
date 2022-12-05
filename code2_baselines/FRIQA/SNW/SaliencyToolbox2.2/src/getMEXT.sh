# getMEXT - determine the correct mex extension for the current OS and CPU

## This file is part of the SaliencyToolbox - Copyright (C) 2006-2008
## by Dirk B. Walther and the California Institute of Technology.
## See the enclosed LICENSE.TXT document for the license agreement. 
## More information about this project is available at: 
## http://www.saliencytoolbox.net


# make sure MATLABROOT is defined
if [ -z "${MATLABROOT}" ]
then
  echo error
  echo MATLABROOT must be defined! >/dev/stderr
  exit 1
elif [ ! -d "${MATLABROOT}" ]
then
  echo error
  echo MATLABROOT: ${MATLABROOT} is not a valid directory! >/dev/stderr
  exit 2
fi

if [ -f "${MATLABROOT}/bin/mexext" ]
then
  # UNIX variants, including Mac OSX
  sh "${MATLABROOT}/bin/mexext"
  exit 0
elif [ -f "${MATLABROOT}/bin/mexext.bat" ]
then
  # Windows
  if [ ${PROCESSOR_ARCHITECTURE} == AMD64 ]
  then
    echo mexw64
    exit 0
  elif [ ${PROCESSOR_ARCHITECTURE} == x86 ]
  then
    echo mexw32
    exit 0
  else
    echo error
    echo Unsupported platform: ${PROCESSORE_ARCHITECTURE} >/dev/stderr
    exit 4
  fi
else
  echo error
  echo Could not determine MEX file extension. >/dev/stderr
  exit 3
fi
