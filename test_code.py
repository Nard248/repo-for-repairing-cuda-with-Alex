import sys
import os
import numpy as np
import random

# Add the build directory to the system path
build_dir = os.path.join(os.path.dirname(__file__), 'build', 'lib.win-amd64-cpython-311', 'Release')
sys.path.append(build_dir)

import simulation

def main():
    print("Initializing random seed...")
    random.seed(20)
    np.random.seed(20)

    # Example parameter initialization
    d = (0.03, 90.0 / 600.0, 90.0 / 600.0)
    N = (600, 600, 600)
    myu_size = (5, 8, 8)
    myu_mstd = (5.4, 1.8)

    print("Initializing parameters...")
    params = simulation.Params()
    simulation.initialize_parameters_py(d[0], d[1], d[2], N[0], N[1], N[2], myu_size, myu_mstd, params)

    print("Computing myu...")
    simulation.compute_myu(params)

    print("Computing state...")
    # Creating a dummy numpy array to mimic d_myu input
    d_myu_np = np.zeros((params.Nt, params.Nx, params.Ny), dtype=np.complex128)
    simulation.compute_state_py(d_myu_np, params)

    print("Printing parameters...")
    simulation.print_parameters(params)

    print("Freeing allocated memory...")
    simulation.free_parameters(params)

    print("Program completed successfully!")

if __name__ == "__main__":
    main()
