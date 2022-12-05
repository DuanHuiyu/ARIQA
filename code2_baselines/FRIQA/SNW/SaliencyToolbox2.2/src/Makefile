# Makefile for mex-compiling functions that
# compute saliency-related functions.
#
# possible targets:
#
# all, default : compile all mex files in ../mex
#
# clean: delete all object files and the binaries for 
#               the current architecture
#
# doc: create the doxygen documentation in ../doc/mexdoc
#

## This file is part of the SaliencyToolbox - Copyright (C) 2006-2008
## by Dirk B. Walther and the California Institute of Technology.
## See the enclosed LICENSE.TXT document for the license agreement. 
## More information about this project is available at: 
## http://www.saliencytoolbox.net

# Matlab versions before 7.0 (R14) don't understand the -cxx flag. For 
# older Matlab versions, comment the first and uncomment the second line.

MEXFLAGS := -cxx
#MEXFLAGS :=


# By default, all intermediate object files are deleted after successful 
# compilation to avoid conflicts between different CPU and OS architectures.
# If you want to keep your object files, uncomment the following line.

#.PRECIOUS: %.o %.obj


# Determine the correct mex file extension
MEXT := $(shell sh ./getMEXT.sh)

BIN := ../mex
NOOP := @echo >/dev/nul

# These options are for Linux and other Unix variants 
MEXOPTS := mexopts.sh
MEX := mex

# These options are only needed for MS Windows
CC := g++
CCFLAGS := -g -O2 -I"$(MATLABROOT)/extern/include" -c
CCLFLAGS := -g -shared -L"$(MATLABROOT)/bin/win32" -lmat -lmex -lmx


MODULES := \
Image.o \
mexLog.o \
Filters.o \
MexParams.o \

OBJMODULES := $(MODULES:.o=.obj)

HEADERS := \
Image.h \
mexLog.h \
Filters.h \
MexParams.h \


all : \
$(BIN)/mexGaussianSubsample.$(MEXT) \
$(BIN)/mexLocalMaxima.$(MEXT) \
$(BIN)/mexConv2PreserveEnergy.$(MEXT) \

default: all

doc: 
	doxygen doxygen.conf

.PHONY : clean

# clean only deletes the binaries for the current architecture
clean:
	rm -f *.o *.obj $(BIN)/mex*.$(MEXT)

# This rule is just to prevent compilation if MATLABROOT wasn't found under Windows
$(BIN)/%.error:
	$(NOOP)


# These are the rules for MS Windows
%.obj: %.cpp $(HEADERS)
	$(CC) $(CCFLAGS) -o $@ $<

$(BIN)/%.mexw32: %.obj $(OBJMODULES) $(HEADERS)
	$(CC) $(CCLFLAGS) -o $@ $< $(OBJMODULES)

$(BIN)/%.mexw64: %.obj $(OBJMODULES) $(HEADERS)
	$(CC) $(CCLFLAGS) -o $@ $< $(OBJMODULES)


# These are the rules for all other operating systems
%.o: %.cpp $(HEADERS)
	$(MEX) -f $(MEXOPTS) $(MEXFLAGS) -c $<

$(BIN)/%.$(MEXT): %.o $(MODULES) $(HEADERS)
	$(MEX) -f $(MEXOPTS) $(MEXFLAGS) -output $@ $< $(MODULES)
