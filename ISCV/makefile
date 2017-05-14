CC = g++
CFLAGS = -c -O3 -DNDEBUG -I/usr/include/python2.7 -fPIC -fopenmp
LDFLAGS = -lpython2.7 -lboost_python -lpthread -lgomp
SOURCES = iscv.cpp
OBJECTS = $(SOURCES:.cpp=.o)
SHARED = ../ISCV.so

all: $(SOURCES) $(SHARED)

.depend: depend

depend:
	$(CC) $(CFLAGS) *.cpp -MM > .depend

include .depend

$(SHARED): $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -shared -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@
