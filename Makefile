FLAGS := -std=c++17 -O3
LIBS := ./libepigraph.so

SOURCES := $(wildcard src/*.cc)
HEADERS := $(wildcard src/*.h)
OBJECTS := $(patsubst src/%.cc, obj/%.o, $(SOURCES))

main: $(OBJECTS) libepigraph.so
	g++ $(FLAGS) $(OBJECTS) -o main $(LIBS) -DENABLE_ECOS -DENABLE_OSQP -I ./Epigraph/include/ -I usr/include/eigen3/

$(OBJECTS): obj/%.o : src/%.cc $(HEADERS)
	g++ $(FLAGS) -c $< -o $@ $(LIBS) -DENABLE_ECOS -DENABLE_OSQP -I ./Epigraph/include/ -I usr/include/eigen3/

.PHONY: clean

clean:
	rm -rf main obj/*
