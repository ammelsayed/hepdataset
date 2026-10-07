CXX ?= c++
CXXFLAGS ?= -O2
ROOT_CONFIG ?= root-config
DELPHES_DIR ?=
BUILD_DIR ?= build
ROOT_BIN = $(dir $(shell readlink -f $(ROOT_CONFIG)))
ROOT_PRIVATE_LIBS = $(ROOT_BIN)/../../root.libs

CPPFLAGS += -I"$(DELPHES_DIR)" -I"$(DELPHES_DIR)/classes" \
            -I"$(DELPHES_DIR)/external" \
            -I"$(DELPHES_DIR)/external/ExRootAnalysis"
CXXFLAGS += -std=c++17 $(shell $(ROOT_CONFIG) --cflags)
LDFLAGS += -L"$(DELPHES_DIR)" -Wl,-rpath,"$(DELPHES_DIR)" \
           -L"$(ROOT_PRIVATE_LIBS)" -Wl,-rpath-link,"$(ROOT_PRIVATE_LIBS)" \
           -Wl,-rpath,"$(ROOT_PRIVATE_LIBS)"
LDLIBS += -lDelphes $(shell $(ROOT_CONFIG) --libs)

SELECTORS = object_selection.cpp event_selection.cpp
PROGRAMS = basic1_delphes basic2_delphes basic3_delphes

.PHONY: all basic1_delphes basic2_delphes basic3_delphes clean
all: $(PROGRAMS)

basic1_delphes: $(BUILD_DIR)/basic1_delphes
basic2_delphes: $(BUILD_DIR)/basic2_delphes
basic3_delphes: $(BUILD_DIR)/basic3_delphes

$(BUILD_DIR)/basic1_delphes: basic1_delphes.cpp $(SELECTORS) object_selection.h event_selection.h
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) basic1_delphes.cpp $(SELECTORS) $(LDFLAGS) $(LDLIBS) -o $@

$(BUILD_DIR)/basic2_delphes: basic2_delphes.cpp object_selection.cpp object_selection.h
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) basic2_delphes.cpp object_selection.cpp $(LDFLAGS) $(LDLIBS) -o $@

$(BUILD_DIR)/basic3_delphes: basic3_delphes.cpp $(SELECTORS) object_selection.h event_selection.h
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) basic3_delphes.cpp $(SELECTORS) $(LDFLAGS) $(LDLIBS) -o $@

clean:
	rm -rf $(BUILD_DIR)
