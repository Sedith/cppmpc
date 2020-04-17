CC=gcc
CPP=g++
AR=ar
CPPFLAGS = -Wall -O3 -march=x86-64 -mfpmath=sse -fPIC -std=c++14
#CPPFLAGS_MAIN = -Wall -march=native -mfpmath=sse -pedantic -Wshadow -Wfloat-equal -O3 -Wconversion -Wsign-conversion -finline-functions -fPIC -DLINUX -D__NO_COPYRIGHT__
ARFLAGS = rcs

QPOASES_PATH = /opt/openrobots/
CPPMPC_PATH  = $(realpath $(dir $(lastword $(MAKEFILE_LIST))))

QPOASES_HEADER = $(QPOASES_PATH)/include
QPOASES_LIB  = $(QPOASES_PATH)/lib
CPPMPC_HEADER = $(CPPMPC_PATH)/include
CPPMPC_LIB  = $(CPPMPC_PATH)/lib
QPOASES_LINK = -L$(QPOASES_LIB) -Wl,-rpath=$(QPOASES_LIB) -lqpOASES
LINK_DEPENDS = -llapack -lblas -lm
CPPMPC_LINK = -L$(CPPMPC_LIB) -Wl,-rpath=$(CPPMPC_LIB) -lcppmpc
MODEL_HEADER = $(CPPMPC_PATH)/model
MODEL_LINK = -L$(CPPMPC_PATH)/model -Wl,-rpath=$(CPPMPC_PATH)/model -lmymodel
