#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_simple_gibbs(py::module &);

namespace mcl {

PYBIND11_MODULE(simple_gibbs, m) {
    // Optional docstring
    m.doc() = "Simple Gibbs library";
    
    init_simple_gibbs(m);
}
}
