# QSIR — A Quarantine-Aware SIR Model with a Universal Differential Equation

A scientific-machine-learning project that teaches a classical epidemiology
model to discover **quarantine strength** on its own. I take the standard SIR
compartment model, embed a small neural network inside it to form a
**Universal Differential Equation (UDE)**, and train the whole thing on real
COVID-19 case data from Italy. The trained network recovers a time-varying
quarantine signal `Q(t)` — and its turning point lines up with the day Italy
actually locked down, even though the model was never told when that happened.

---

## 📌 Motivation

The classical **SIR** model (Susceptible → Infected → Recovered) has no concept
of quarantine or lockdown. During COVID-19 that was a huge gap: interventions
like national lockdowns dramatically changed the trajectory of the outbreak, but
a plain SIR model can't represent them.

Instead of hand-crafting a quarantine term, I let the data speak. A neural
network is embedded directly into the differential equations and trained
end-to-end alongside the mechanistic SIR parameters. The result is a model that:

- **estimates** infection dynamics accurately,
- **discovers** the quarantine strength `Q(t)` as a function of the system state, and
- **acts as a diagnostic tool** — the learned `Q(t)` can be read back and
  compared against real-world policy timelines.

The example region here is **Italy**, one of the earliest and hardest-hit
countries in the first COVID-19 wave.

---

## 🧠 The Idea: a Universal Differential Equation

A UDE keeps the parts of a model we understand mechanistically and replaces the
parts we *don't* with a neural network that is trained from data.

Quarantined and isolated people stop coming into contact with the infected, so
quarantine effectively **removes** individuals from the infectious pool. I
model this by injecting a neural network as a **subtraction term** in the
infected-compartment equation and routing those removed individuals into a
dedicated quarantined compartment `T`.

The four compartments are `u = [S, I, R, T]`:

```
dS/dt = −β · S · I / N
dI/dt =  β · S · I / N  −  γ · I  −  NN(S, I, R) · I / N
dR/dt =  γ · I  +  δ · T
dT/dt =  NN(S, I, R) · I / N  −  δ · T
```

- `β` — infection rate, `γ` — recovery rate, `δ` — release rate from quarantine
  (all learned).
- `NN(S, I, R)` — a neural network (3 → 10 `relu` → 1) whose output is the
  learned quarantine strength.
- `N` — the total population (a normalising constant).

---

## ⚙️ How It Works

1. **Load the data.** Italy's infected / recovered / dead time series (originally
   from the Johns Hopkins COVID-19 repository) ships with the project as
   `Rise_Italy_Track.mat`. The cumulative infected count is corrected to the
   *currently active* infected count.
2. **Build the UDE.** Define the SIR parameters and the neural network, then pack
   them into a single flat parameter vector (`ComponentArray`) so they can be
   trained jointly.
3. **Train.** Fit in log-space so both the early exponential growth and the later
   plateau matter. Optimisation runs in two stages:
   - **ADAM** for a robust warm-up, then
   - **BFGS** for sharp local refinement.
   Gradients flow back through the ODE solve via an interpolating adjoint.
4. **Diagnose.** Evaluate the trained network across the timeline to recover
   `Q(t)`, find its inflection point (first sign change of the second
   difference), and overlay it against Italy's actual lockdown date.

---

## 📊 Result

The inflection point of the learned `Q(t)` lands very close to **day 11**, when
Italy imposed its national lockdown — despite the model never being given that
date. The neural term recovered a genuine policy signal purely from case counts,
which is exactly what makes this a *diagnostic* model and not just a curve fit.

Running the script produces two figures:
- **Fit** — predicted infected/recovered curves overlaid on the real data.
- **Q(t)** — the learned quarantine strength over time, with markers for the
  actual lockdown date and the learned inflection point.

---

## 📂 Project Structure

| File | Description |
|------|-------------|
| `CovidProject.jl` | Main script: builds, trains, and diagnoses the QSIR UDE. |
| `setup.jl` | One-shot installer for all Julia dependencies. |
| `Rise_Italy_Track.mat` | Italy's infected / recovered / dead time series. |
| `LICENSE` | MIT license. |

---

## 🛠️ Requirements

- [Julia](https://julialang.org/downloads/) 1.9 or newer.
- The following packages (installed automatically by `setup.jl`):
  `MAT`, `Plots`, `Measures`, `Lux`, `DifferentialEquations`, `DiffEqFlux`,
  `Optimization`, `OptimizationFlux`, `OptimizationOptimJL`,
  `OptimizationOptimisers`, `ComponentArrays`, `Zygote`.

---

## ▶️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/VinayakMokashi/QSIR-Model.git
cd QSIR-Model

# 2. Install the dependencies (only needed once)
julia setup.jl

# 3. Run the model
julia CovidProject.jl
```

Training prints the loss every 10 iterations and, once finished, renders the two
plots described above. The ADAM stage runs up to 15,000 iterations, so expect
the full run to take a while on a CPU.

> **Tip:** the data path is resolved relative to the script
> (`joinpath(@__DIR__, "Rise_Italy_Track.mat")`), so the project runs from any
> location without editing paths.

---

## 💡 Takeaway

Pairing a neural network with a mechanistic ODE keeps the model **expressive and
interpretable** at the same time. The learned term can capture effects that are
hard to hand-write in a classical compartment model — quarantine here, but also
early reopening, vaccination uptake, and more — without collapsing into the
black-box opacity of a fully agent-based simulation.

---

## 📄 License

Released under the [MIT License](LICENSE).
