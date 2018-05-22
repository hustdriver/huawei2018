SRC_DIRS := ./
CC := g++
CPPFLAGS := -I./lib -std=c++11 -Wall -Wextra -Wpedantic -g
LDFLAGS :=

DEPS := ./predict.h ./lib/lib_time.h ./lib/lib_io.h
SRCPATTERN := *.cpp
SRCS := $(shell find $(SRC_DIRS) -name '$(SRCPATTERN)')
OBJS := $(addsuffix .o,$(basename $(SRCS)))

.PHONY: run test
run: ecs
	./ecs $(ARGS)

test: ecs
	./ecs ./quarter_final/samples/TrainData_2015.12.txt ./quarter_final/samples/input_3hosttypes_5flavors_1week.txt output.txt

debug: ecs
	gdb --args ./ecs ./quarter_final/samples/TrainData_2015.12.txt ./quarter_final/samples/input_3hosttypes_5flavors_1week.txt output.txt

%.o: $(subst *,%,$(SRCPATTERN)) $(DEPS)
	$(CC) $(CPPFLAGS) $(DEBUG) -c -o $@ $< $(LDFLAGS)

ecs: $(OBJS)
	$(CC) $(CPPFLAGS) $(DEBUG) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(OBJS)

dos2unix:
	dos2unix $(shell find $(SRC_DIRS) -name '*.cpp' -o -name '*.h')

unix2dos:
	unix2dos $(shell find $(SRC_DIRS) -name '*.cpp' -o -name '*.h')
