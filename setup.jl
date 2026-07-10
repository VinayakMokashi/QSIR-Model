# Installs every package needed to run CovidProject.jl.
# Run once from this folder:  julia setup.jl
import Pkg

packages = [
    "MAT",                    # read the .mat dataset
    "Plots",                  # plotting
    "Measures",               # plot margins (mm)
    "Lux",                    # neural network layers
    "DifferentialEquations",  # ODE solvers
    "DiffEqFlux",             # differentiable ODE / adjoint sensitivities
    "Optimization",           # optimisation front-end
    "OptimizationFlux",       # Flux-based optimisers
    "OptimizationOptimJL",    # Optim.jl backend (BFGS)
    "OptimizationOptimisers", # Optimisers.jl backend (ADAM)
    "ComponentArrays",        # flat, named parameter vectors
    "Zygote",                 # reverse-mode autodiff
]

Pkg.add(packages)
println("\nAll dependencies installed. Run the model with:  julia CovidProject.jl")
