FLAGS := -std=c++17 -O3
LIBS := libs/libepigraph.so libs/libecos.so libs/libosqp.so
INCLUDE := -I usr/include/eigen3/ -I ./Epigraph/include/ -I ./Epigraph/solvers/ecos/include -I ./Epigraph/solvers/ecos/external/SuiteSparse_config -I ./Epigraph/solvers/osqp/include

SOURCES := $(wildcard src/*.cc)
HEADERS := $(wildcard src/*.h)
OBJECTS := $(patsubst src/%.cc, obj/%.o, $(SOURCES))

main: $(OBJECTS) $(LIBS)
	g++ $(FLAGS) $(OBJECTS) -o main $(LIBS) -DENABLE_ECOS -DENABLE_OSQP $(INCLUDE)

$(OBJECTS): obj/%.o : src/%.cc $(HEADERS)
	g++ $(FLAGS) -c $< -o $@ -DENABLE_ECOS -DENABLE_OSQP $(INCLUDE)

.PHONY: clean

clean:
	rm -rf main obj/*
