# Build the native C++ adaptive Delphes loop (no embedded Python).
# Example:
#   export ROOTSYS=/path/to/root
#   export DELPHES_DIR=/path/to/Delphes
#   export PATH="$ROOTSYS/bin:$PATH"
#   make -C src/loops/native_cpp native_adaptive_delphes

CXX ?= c++
ROOT_CONFIG ?= root-config
DELPHES_DIR ?=

ROOT_CFLAGS := $(shell $(ROOT_CONFIG) --cflags)
ROOT_LIBS := $(shell $(ROOT_CONFIG) --libs)
ROOT_PREFIX := $(shell dirname "$$(dirname "$$(readlink -f "$$(command -v $(ROOT_CONFIG))")")")
ROOT_PRIVATE_LIBS := $(ROOT_PREFIX)/../root.libs
CXXFLAGS ?= -O2
CXXFLAGS += -std=c++17 $(ROOT_CFLAGS)
CPPFLAGS += -I"$(DELPHES_DIR)" -I"$(DELPHES_DIR)/external" -I"$(DELPHES_DIR)/external/ExRootAnalysis" -I"$(CURDIR)/vendor"
LDFLAGS += -L"$(DELPHES_DIR)" -Wl,-rpath,"$(DELPHES_DIR)" -Wl,-rpath-link,"$(ROOT_PRIVATE_LIBS)" -Wl,-rpath,"$(ROOT_PRIVATE_LIBS)"
LDLIBS += -lDelphes $(ROOT_LIBS)

NATIVE_LOOP := native_adaptive_delphes
NATIVE_SOURCE := native_adaptive_delphes.cpp

.PHONY: all clean
all: native_adaptive_delphes

$(NATIVE_LOOP): $(NATIVE_SOURCE) vendor/mt2_bisect.h
	@test -n "$(DELPHES_DIR)" || (echo "Set DELPHES_DIR to the Delphes source/build directory" >&2; exit 2)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(NATIVE_SOURCE) $(LDFLAGS) $(LDLIBS) -o $@

clean:
	rm -f $(NATIVE_LOOP)
