#include <string>

#include <armadillo>

#ifndef GIBBS_SAMPLER_H
#define GIBBS_SAMPLER_H

namespace gibbs {

class GibbsSampler {

private:

    /// No units in each dim
    int _no_units;
    
    /// Coupling
    double _coupling;
    
    /// Bias
    double _bias;

public:

    /// Constructor
    GibbsSampler(int no_units, double coupling, double bias);

    /// Get random state of 0 or 1
    /// @return 2D state of random 0 or 1
    arma::imat get_random_state() const;

    /// Sample
    /// @param state_init Initital state
    /// @param no_steps No steps to sample
    /// @return Final state after sampling
    arma::imat sample(const arma::imat &state_init, int no_steps) const;
};

}

#endif
