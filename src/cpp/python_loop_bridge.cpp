#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <string>

namespace {

bool is_supported_loop(const char* name) {
    static const char* supported[] = {
        "adaptive_delphes",
        "explicit_delphes",
        "basic1_delphes",
        "basic2_delphes",
        "basic3_delphes",
        nullptr,
    };
    for (const char** item = supported; *item != nullptr; ++item) {
        if (std::string(name) == *item) return true;
    }
    PyErr_Format(PyExc_ValueError, "Unsupported HEPDataset loop '%s'", name);
    return false;
}

PyObject* import_loop(const char* name) {
    if (!is_supported_loop(name)) return nullptr;
    const std::string module_name = std::string("hepdataset.loops.") + name;
    return PyImport_ImportModule(module_name.c_str());
}

PyObject* run_loop(PyObject*, PyObject* args) {
    const char* name = nullptr;
    PyObject* kwargs = nullptr;
    if (!PyArg_ParseTuple(args, "sO!", &name, &PyDict_Type, &kwargs)) return nullptr;

    PyObject* module = import_loop(name);
    if (module == nullptr) return nullptr;
    PyObject* function = PyObject_GetAttrString(module, "loop_tree");
    Py_DECREF(module);
    if (function == nullptr) return nullptr;

    PyObject* empty_args = PyTuple_New(0);
    if (empty_args == nullptr) {
        Py_DECREF(function);
        return nullptr;
    }
    PyObject* result = PyObject_Call(function, empty_args, kwargs);
    Py_DECREF(empty_args);
    Py_DECREF(function);
    return result;
}

PyObject* run_cli(PyObject*, PyObject* args) {
    const char* name = nullptr;
    PyObject* cli_args = nullptr;
    if (!PyArg_ParseTuple(args, "sO!", &name, &PyList_Type, &cli_args)) return nullptr;

    PyObject* module = import_loop(name);
    if (module == nullptr) return nullptr;
    PyObject* main_function = PyObject_GetAttrString(module, "main");
    Py_DECREF(module);
    if (main_function == nullptr) return nullptr;

    PyObject* argv = PyList_New(0);
    if (argv == nullptr) {
        Py_DECREF(main_function);
        return nullptr;
    }
    const std::string program_name = std::string("hepdataset ") + name;
    PyObject* first = PyUnicode_FromString(program_name.c_str());
    if (first == nullptr || PyList_Append(argv, first) < 0) {
        Py_XDECREF(first);
        Py_DECREF(argv);
        Py_DECREF(main_function);
        return nullptr;
    }
    Py_DECREF(first);
    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(cli_args); ++i) {
        PyObject* item = PyList_GET_ITEM(cli_args, i);
        if (!PyUnicode_Check(item) || PyList_Append(argv, item) < 0) {
            if (!PyErr_Occurred()) {
                PyErr_SetString(PyExc_TypeError, "CLI arguments must be strings");
            }
            Py_DECREF(argv);
            Py_DECREF(main_function);
            return nullptr;
        }
    }

    PyObject* sys_module = PyImport_ImportModule("sys");
    if (sys_module == nullptr) {
        Py_DECREF(argv);
        Py_DECREF(main_function);
        return nullptr;
    }
    PyObject* original_argv = PyObject_GetAttrString(sys_module, "argv");
    if (original_argv == nullptr || PyObject_SetAttrString(sys_module, "argv", argv) < 0) {
        Py_XDECREF(sys_module);
        Py_XDECREF(original_argv);
        Py_DECREF(argv);
        Py_DECREF(main_function);
        return nullptr;
    }
    Py_DECREF(argv);
    PyObject* result = PyObject_CallObject(main_function, nullptr);
    Py_DECREF(main_function);

    // Preserve any exception from main() while restoring process-global state.
    PyObject* error_type = nullptr;
    PyObject* error_value = nullptr;
    PyObject* error_traceback = nullptr;
    if (result == nullptr) {
        PyErr_Fetch(&error_type, &error_value, &error_traceback);
    }
    const int restore_status = PyObject_SetAttrString(sys_module, "argv", original_argv);
    Py_DECREF(original_argv);
    Py_DECREF(sys_module);
    if (result == nullptr) {
        PyErr_Restore(error_type, error_value, error_traceback);
    } else if (restore_status < 0) {
        Py_DECREF(result);
        result = nullptr;
    }
    return result;
}

PyMethodDef bridge_methods[] = {
    {"run_loop", run_loop, METH_VARARGS,
     "Call a supported HEPDataset Python loop_tree with keyword arguments."},
    {"run_cli", run_cli, METH_VARARGS,
     "Run a supported HEPDataset Python loop main() with CLI arguments."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef bridge_module = {
    PyModuleDef_HEAD_INIT,
    "_cpp_loop_bridge",
    "C++/CPython bridge for HEPDataset Delphes event loops.",
    -1,
    bridge_methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__cpp_loop_bridge(void) {
    return PyModule_Create(&bridge_module);
}
