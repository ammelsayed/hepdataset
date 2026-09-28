PYTHON ?= python3
CXX ?= c++
PY_INCLUDE := $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_path("include"))')
PY_LIBDIR := $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')
PY_VERSION := $(shell $(PYTHON) -c 'import sys; print(".".join(map(str,sys.version_info[:2])))')
PY_LIBRARY := python$(PY_VERSION)
CXXFLAGS ?= -O2 -std=c++11 -Wall -Wextra

.PHONY: all extension clean
all: extension hepdataset-loops

extension:
	cd ../.. && HEPDATASET_BUILD_CPP=1 $(PYTHON) setup.py build_ext --inplace

hepdataset-loops: hepdataset_loops.cpp
	$(CXX) $(CXXFLAGS) -I$(PY_INCLUDE) -L$(PY_LIBDIR) -Wl,-rpath,$(PY_LIBDIR) \
		$< -o $@ -l$(PY_LIBRARY) -ldl -lm

clean:
	rm -f hepdataset-loops
	find ../loops -maxdepth 1 -name '_cpp_loop_bridge*.so' -delete
