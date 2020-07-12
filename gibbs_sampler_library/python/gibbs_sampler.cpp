#include "../cpp/include/simple_gibbs_bits/gibbs_sampler.hpp"

#include <armadillo>

// pybind11
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

// Carma
#include <carma/carma.h>

namespace gibbs {

class GibbsSampler_Cpp : public GibbsSampler {
    
public:
    
    using GibbsSampler::GibbsSampler;
    
    py::array_t<double> get_random_state() const {
        
        // Call
        arma::imat state_arma = GibbsSampler::get_random_state();
        
        // Convert back to numpy
        py::array_t<double> state = carma::mat_to_arr<arma::sword>(state_arma);
        
        return state;
    }
    
    py::array_t<double> sample(const py::array_t<double> &state_init, int no_steps) const {
        
        // Convert to arma
        arma::imat state_init_arma = carma::arr_to_mat<arma::sword>(state_init);
        
        // Call
        arma::imat state_arma = GibbsSampler::sample(state_init_arma, no_steps);
        
        // Convert back to numpy
        py::array_t<double> state = carma::mat_to_arr<arma::sword>(state_arma);
        
        return state;
    }
};

}

void init_simple_gibbs(py::module &m) {
    
    py::class_<gibbs::GibbsSampler>(m, "GibbsSampler_Parent");
    
    py::class_<gibbs::GibbsSampler_Cpp, gibbs::GibbsSampler>(m, "GibbsSampler")
    .def(py::init<int, double, double>(), py::arg("no_units"), py::arg("coupling"), py::arg("bias"))
    .def("get_random_state",
         py::overload_cast<>( &gibbs::GibbsSampler_Cpp::get_random_state, py::const_))
    .def("sample",
         py::overload_cast<const py::array_t<double>&, int>( &gibbs::GibbsSampler_Cpp::sample, py::const_),
         py::arg("state_init"), py::arg("no_steps"));
}
