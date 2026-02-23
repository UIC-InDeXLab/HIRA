# HIRA C++/OpenMP kernels
#
# Build options:
#   make            – build with OpenMP + native arch optimisation
#   make clean      – remove build artefacts
#
# Requirements:
#   - A C++17 compiler (g++ >= 7, clang++ >= 5)
#   - pybind11  (pip install pybind11)
#   - OpenMP    (usually ships with the compiler)

PYTHON  ?= python3
CXX     ?= g++

# pybind11 include flags
PB11_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes)

# Python extension suffix (.cpython-310-x86_64-linux-gnu.so, etc.)
EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

TARGET_FILTER := hira_cpp_kernels$(EXT_SUFFIX)

CXXFLAGS := -O3 -Wall -shared -std=c++17 -fPIC \
            -fopenmp -march=native -ffast-math \
            $(PB11_INCLUDES)

LDFLAGS := -fopenmp

.PHONY: all clean

all: $(TARGET_FILTER)
	@echo "Built $(TARGET_FILTER)"

$(TARGET_FILTER): exact_filter.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f hira_cpp_kernels*.so hira_cpp_fused*.so
