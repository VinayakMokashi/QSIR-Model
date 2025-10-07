# QSIR-Model: Universal Differential Equation for Quarantine-Aware SIR Dynamics
## 📌 Introduction
This project constructs a **Universal Differential Equation (UDE)** based on the classical SIR epidemiology model, enhanced to capture **quarantine effects**. By integrating a neural network into the model, we can:
- Improve real-time estimation of infection dynamics.
- Use the model as a **diagnostic tool** to understand quarantine effectiveness in any region.
- Reproduce real-world effects such as government-imposed lockdowns.
In this example, we focus on **Italy** during the COVID-19 outbreak.
## ⚙️ Methodology
1. **Base Model**: Start with the SIR model (Susceptible-Infected-Recovered) that does not account for quarantine.  
2. **Neural Network Integration**: Inject a neural network into the infected compartment equation (`dI/dt`) as a subtraction term to model quarantine effects.  
3. **Training**: Train the UDE on real infection and recovery data from the **Johns Hopkins COVID-19 dataset**.  
4. **Diagnostics**: Analyze the learned quarantine function \( Q(t) \) and compare it against actual government lockdown timelines.  
## 📂 Project Structure
- **CovidProject.jl** → Main Julia code implementing the QSIR model with neural networks.  
- **QSIR-Covid-Project-Raj-Dandekar.pdf** → Original reference/tutorial notes.  
- **Rise_Italy_Track.mat** → Dataset file containing infection, recovery, and death counts for Italy.  
## 🛠️ Installation & Requirements
This project is implemented in **Julia**. You need the following packages:
```julia
using MAT
using Plots, Measures, Lux
using DifferentialEquations
using DiffEqFlux, Optimization, OptimizationFlux
using Random
using ComponentArrays
using OptimizationOptimJL, OptimizationOptimisers
