"""Optional C++ bridge extension build for HEPDataset event loops.

Set ``HEPDATASET_BUILD_CPP=1`` to compile the CPython bridge extension. This
keeps ordinary Python-only installs independent of a C++ compiler and Python
headers while allowing the Makefile to build the optional bridge.
"""

import os

from setuptools import Extension, setup

extensions = []
if os.environ.get("HEPDATASET_BUILD_CPP") == "1":
    extensions.append(
        Extension(
            "hepdataset.loops._cpp_loop_bridge",
            sources=["src/cpp/python_loop_bridge.cpp"],
            language="c++",
            extra_compile_args=["-std=c++11"],
        )
    )

setup(ext_modules=extensions)
