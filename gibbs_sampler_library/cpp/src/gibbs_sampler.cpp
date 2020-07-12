#include "../include/simple_gibbs_bits/gibbs_sampler.hpp"

#include <iostream>

namespace gibbs {

GibbsSampler::GibbsSampler(int no_units, double coupling, double bias) {
    _no_units = no_units;
    _coupling = coupling;
    _bias = bias;
}

arma::imat GibbsSampler::get_random_state() const {
    return 2 * arma::randi( _no_units, _no_units, arma::distr_param(0,1) ) - 1;
}

arma::imat GibbsSampler::sample(const arma::imat &state_init, int no_steps) const {
    
    // Copy state of 0 or 1
    arma::imat state = state_init;

    // Iterate over no steps
    for (auto i_step=0; i_step<no_steps; i_step++) {
        
        if (i_step % 10000 == 0) {
            std::cout << i_step << " / " << no_steps << std::endl;
        }

        // Pick a random idx
        int i = arma::randi( arma::distr_param(0,_no_units-1) );
        int j = arma::randi( arma::distr_param(0,_no_units-1) );

        // Energy change
        double energy_diff = _bias * -2 * state(i,j);
        if (i != 0) {
            energy_diff += _coupling * state(i-1, j) * -2 * state(i,j);
        }
        if (i != _no_units-1) {
            energy_diff += _coupling * state(i+1, j) * -2 * state(i,j);
        }
        if (j != 0) {
            energy_diff += _coupling * state(i, j-1) * -2 * state(i,j);
        }
        if (j != _no_units-1) {
            energy_diff += _coupling * state(i, j+1) * -2 * state(i,j);
        }

        // Sample
        double r = arma::randu();
        if (exp(energy_diff) < r) {
            // Flip
            state(i,j) = - state(i,j);
        }
    }
 
    return state;
}

}
