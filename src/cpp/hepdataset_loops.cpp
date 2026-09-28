#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: hepdataset-loops <adaptive_delphes|explicit_delphes|basic1_delphes|basic2_delphes|basic3_delphes> [loop options...]\n";
        return 2;
    }

    const std::string loop_name(argv[1]);
    std::vector<std::string> valid = {
        "adaptive_delphes", "explicit_delphes", "basic1_delphes",
        "basic2_delphes", "basic3_delphes",
    };
    bool found = false;
    for (const auto& candidate : valid) {
        if (candidate == loop_name) found = true;
    }
    if (!found) {
        std::cerr << "Unknown loop: " << loop_name << "\n";
        return 2;
    }

    Py_Initialize();
    PyObject* bridge = PyImport_ImportModule("hepdataset.loops._cpp_loop_bridge");
    if (bridge == nullptr) {
        PyErr_Print();
        Py_Finalize();
        return 1;
    }
    PyObject* run_cli = PyObject_GetAttrString(bridge, "run_cli");
    Py_DECREF(bridge);
    if (run_cli == nullptr) {
        PyErr_Print();
        Py_Finalize();
        return 1;
    }

    PyObject* name = PyUnicode_FromString(loop_name.c_str());
    PyObject* arguments = PyList_New(0);
    if (name == nullptr || arguments == nullptr) {
        Py_XDECREF(name);
        Py_XDECREF(arguments);
        Py_DECREF(run_cli);
        PyErr_Print();
        Py_Finalize();
        return 1;
    }
    for (int i = 2; i < argc; ++i) {
        PyObject* item = PyUnicode_FromString(argv[i]);
        if (item == nullptr || PyList_Append(arguments, item) < 0) {
            Py_XDECREF(item);
            Py_DECREF(name);
            Py_DECREF(arguments);
            Py_DECREF(run_cli);
            PyErr_Print();
            Py_Finalize();
            return 1;
        }
        Py_DECREF(item);
    }

    PyObject* result = PyObject_CallFunctionObjArgs(run_cli, name, arguments, nullptr);
    Py_DECREF(name);
    Py_DECREF(arguments);
    Py_DECREF(run_cli);
    if (result == nullptr) {
        PyErr_Print();
        Py_Finalize();
        return 1;
    }
    long status = PyLong_AsLong(result);
    Py_DECREF(result);
    if (PyErr_Occurred()) {
        PyErr_Print();
        Py_Finalize();
        return 1;
    }
    Py_Finalize();
    return static_cast<int>(status);
}
