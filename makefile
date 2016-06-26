
#
OBJS = ARMA11.o test_sp500.o
CXX = g++
CFLAGS  = -O2 -w -c
LFLAGS  = -Wall
AD_L = -L /usr/local/lib -larmadillo
AD_I = -I /usr/local/include

#
all: main

main: $(OBJS)
	$(CXX) $(LFLAGS) -o main $(OBJS) $(AD_I) $(AD_L)

ARMA11.o: ARMA11.hpp ARMA11.cpp
	$(CXX) $(CFLAGS) $(AD_I) $(AD_L) -c ARMA11.cpp

test.o: test.cpp ARMA11.hpp
	$(CXX) $(CFLAGS) $(AD_I) $(AD_L) -c test.cpp

test_sp500.o: test_sp500.cpp ARMA11.hpp
	$(CXX) $(CFLAGS) $(AD_I) $(AD_L) -c test_sp500.cpp

