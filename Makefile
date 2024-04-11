FLAGS := -std=c++17 -O3 -DENABLE_ECOS -DENABLE_OSQP -I /usr/include/eigen3/ -I /opt/homebrew/include/eigen3 \
				 -I Epigraph/include/ -I Epigraph/solvers/ecos/include -I Epigraph/solvers/osqp/include -I Epigraph/solvers/ecos/external/SuiteSparse_config
LIBS := $(wildcard lib/*)

SOURCES := $(wildcard src/*.cc)
HEADERS := $(wildcard src/*.h)
OBJECTS := $(patsubst src/%.cc, obj/%.o, $(SOURCES))

main: $(OBJECTS) $(LIBS)
	g++ $(FLAGS) $(OBJECTS) -o main $(LIBS) -Wl,-rpath,lib

$(OBJECTS): obj/%.o : src/%.cc $(HEADERS)
	g++ $(FLAGS) -c $< -o $@

.PHONY: clean

clean:
	rm -rf main obj/*
