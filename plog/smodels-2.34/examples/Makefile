CXX = g++
CXXFLAGS = -O3 -W -Wall -I..
LFLAGS =
SRCS = example.cc example2.cc example3.cc example4.cc example5.cc
OBJS = $(SRCS:.cc=.o)
SMODELSSRCS = smodels.cc stack.cc dcl.cc atomrule.cc read.cc \
queue.cc timer.cc list.cc improve.cc program.cc api.cc stable.cc \
tree.cc denant.cc restart.cc
SMODELSOBJS = $(SMODELSSRCS:%.cc=../%.o)

examples: $(OBJS)
	$(CXX) $(LFLAGS) -o example example.o $(SMODELSOBJS)
	$(CXX) $(LFLAGS) -o example2 example2.o $(SMODELSOBJS)
	$(CXX) $(LFLAGS) -o example3 example3.o $(SMODELSOBJS)
	$(CXX) $(LFLAGS) -o example4 example4.o $(SMODELSOBJS)
	$(CXX) $(LFLAGS) -o example5 example5.o $(SMODELSOBJS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $<
