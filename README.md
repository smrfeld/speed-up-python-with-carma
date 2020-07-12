
## Pure Python

Let's start with a simple implementation of the Gibbs Sampler:
```
import numpy as np

class GibbsSampler:

    """Constructor
    """
    def __init__(self, no_units : int, coupling : float, bias : float):
        """Gibbs sampler

        Args:
            no_units (int): No units
            coupling (float): Coupling
            bias (float): Bias
        """

        self.no_units = no_units
        self.coupling = coupling
        self.bias = bias

    def get_random_state(self) -> np.array:
        """Get random state

        Returns:
            np.array: Random no_units x no_units state of -1 or 1
        """
        return 2 * ( 2 * np.random.rand(self.no_units, self.no_units) ).astype(int) - 1 # Randomly 0 or 1

    def sample(self, state_init : np.array, no_steps : int) -> np.array:
        """Perform Gibbs sampling

        Args:
            state_init (np.array): Initial state of -1/1
            no_steps (int): No steps to sample

        Returns:
            np.array: List of sampled units in each state
        """

        # Initialize state of 0 or 1
        state = state_init.astype(int)

        # Iterate over no steps
        for i_step in range(0,no_steps):

            if i_step % 10000 == 0:
                print("%d / %d" % (i_step,no_steps))

            # Pick a random idx
            i,j = np.random.randint(0,self.no_units,2)

            # Energy change
            energy_diff = self.bias * -2 *state[i,j]
            if i != 0:
                energy_diff += self.coupling * state[i-1, j] * -2 * state[i,j]
            if i != self.no_units-1:
                energy_diff += self.coupling * state[i+1, j] * -2 * state[i,j]
            if j != 0:
                energy_diff += self.coupling * state[i, j-1] * -2 * state[i,j]
            if j != self.no_units-1:
                energy_diff += self.coupling * state[i, j+1] * -2 * state[i,j]

            # Sample
            r = np.random.rand()
            if np.exp(energy_diff) < r:
                # Flip 
                state[i,j] = - state[i,j]

        return state
```
We have two methods: one with returns a random state (a 2D numpy array of 0 or 1), and one which takes an initial state, samples it, and returns the final state. 

Let's write a simple test for it:
```
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
```
Here we create a `100x100` lattice with bias `0` and coupling parameter `-1`. We sample for 100,000 steps. Below are a examples of an initial state and a final state:

<img src="python_only/fig1.png" alt="drawing" width="400"/>

<img src="python_only/fig2.png" alt="drawing" width="400"/>

Timing the code gives:
```
Duration: 2.611175 seconds
```
That's way too long! Let's try to write the same code in `C++` and see if we get an improvement.
