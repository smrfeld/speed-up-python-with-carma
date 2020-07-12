import gibbs_sampler as gs
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":
    
    no_units = 100
    coupling = -1
    bias = 0.0

    sampler = gs.GibbsSampler(
        no_units=no_units,
        coupling=coupling,
        bias=bias
        )

    # Sample
    t0 = time.time()
    no_steps = 100000
    state_init = sampler.get_random_state()
    state = sampler.sample(state_init, no_steps)
    t1 = time.time()

    print("Duration: %f seconds" % (t1-t0))

    # Plot
    plt.figure()
    plt.imshow(state_init, cmap="Paired")
    
    plt.figure()
    plt.imshow(state, cmap="Paired")
    plt.show()