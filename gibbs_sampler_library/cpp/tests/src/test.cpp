#include <simple_gibbs>

#include <iostream>

int main() {

    int no_units = 100;
    double coupling = -1;
    double bias = 0.0;

    gibbs::GibbsSampler sampler(no_units, coupling, bias);

    // Sample
    auto start = std::chrono::high_resolution_clock::now();

    int no_steps = 100000;
    arma::imat state_init = sampler.get_random_state();
    arma::imat state = sampler.sample(state_init, no_steps);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    
    std::cout << "Duration: " << duration << " milliseconds" << std::endl;
    
    return 0;
}
