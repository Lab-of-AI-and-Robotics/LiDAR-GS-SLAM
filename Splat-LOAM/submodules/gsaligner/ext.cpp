#include <torch/extension.h>

#include "gsaligner/aligner.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // m.def("align_geometric",)
  m.def("RegistrationGeomPhoto", &RegistrationGeomPhoto);

  py::class_<GSAligner>(m, "GSAligner")
      .def(py::init<const int, const int, const float, const float, const float,
                    const float, const int, const float, const float,
                    const float, const float, const float, const int>())
      .def("setReference", &GSAligner::setReference)
      .def("setQuery", &GSAligner::setQuery)
      .def("alignGeometric", &GSAligner::alignGeometric)
      .def("alignPhotometric", &GSAligner::alignPhotometric);
}