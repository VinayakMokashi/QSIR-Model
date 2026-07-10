# =============================================================================
#  Quarantine-Aware SIR (QSIR) — a Universal Differential Equation for COVID-19
# =============================================================================
#
#  Idea
#  ----
#  The classical SIR compartment model has no notion of quarantine or lockdown.
#  Here I augment it with a small neural network so the model can *learn* the
#  time-varying quarantine strength Q(t) directly from case-count data. The
#  result is a Universal Differential Equation (UDE): known mechanistic terms
#  (the SIR dynamics) combined with a learned data-driven term (the NN).
#
#  Because quarantined/isolated people stop coming into contact with infected
#  people, the effect of quarantine is a *removal* of infectious individuals.
#  I therefore inject the neural network as a subtraction term in the dI/dt
#  equation, and route those individuals into a separate quarantined
#  compartment T.
#
#  Pipeline
#  --------
#    1. Load Italy's infected / recovered / dead time series (Johns Hopkins).
#    2. Define the QSIR UDE: SIR parameters (β, γ, δ) + a neural net for Q(t).
#    3. Train the UDE end-to-end (ADAM warm-up, then BFGS refinement).
#    4. Recover Q(t) from the trained network and compare its inflection point
#       against the date Italy actually imposed its national lockdown.
#
#  The compartments are  u = [S, I, R, T]  (Susceptible, Infected, Recovered,
#  quarantined/removed-by-Quarantine).
# =============================================================================

# ---------------------------------------------------------------------------
#  Packages
# ---------------------------------------------------------------------------
using MAT                       # read the .mat dataset
using Plots, Measures, Lux      # plotting + neural network layers
using DifferentialEquations     # ODE solvers
using DiffEqFlux, Optimization, OptimizationFlux
using Random
using ComponentArrays
using OptimizationOptimJL, OptimizationOptimisers

# ---------------------------------------------------------------------------
#  Data import and cleanup
# ---------------------------------------------------------------------------
#  The dataset lives next to this script, so resolve it relative to the file
#  instead of hard-coding an absolute path — this keeps the project portable
#  across machines.
vars = matread(joinpath(@__DIR__, "Rise_Italy_Track.mat"))
Random.seed!(50)               # reproducible weight initialisation / training

Infected  = vars["Italy_Infected_All"]
Recovered = vars["Italy_Recovered_All"]
Dead      = vars["Italy_Dead_All"]
Time      = vars["Italy_Time"]

# The raw "Infected" series is cumulative; subtract those who have already
# recovered or died to obtain the currently-active infected count.
Infected = Infected - Recovered - Dead

# ---------------------------------------------------------------------------
#  The QSIR Universal Differential Equation
# ---------------------------------------------------------------------------
#  Neural network for the quarantine term: 3 inputs (S, I, R) -> 10 hidden
#  (ReLU) -> 1 output. It outputs the instantaneous quarantine strength that
#  is subtracted from the infected compartment.
rng = Random.default_rng()
ann = Lux.Chain(Lux.Dense(3, 10, relu), Lux.Dense(10, 1))
p1, st1 = Lux.setup(rng, ann)

# Trainable SIR parameters: β (infection rate), γ (recovery rate),
# δ (rate at which quarantined individuals are released/recovered).
parameter_array = Float64[0.15, 0.013, 0.01]

# Pack the neural-network weights and the SIR parameters into a single
# flat parameter vector so the optimiser can train everything jointly.
p0_vec = (layer_1 = p1, layer_2 = parameter_array)
p0_vec = ComponentArray(p0_vec)

function QSIR(du, u, p, t)
    β = abs(p.layer_2[1])
    γ = abs(p.layer_2[2])
    δ = abs(p.layer_2[3])

    # Learned quarantine strength for the current state; abs() keeps it
    # non-negative (a quarantine can only remove infectious people).
    UDE_term = abs(ann([u[1]; u[2]; u[3]], p.layer_1, st1)[1][1])

    du[1] = -β * u[1] * (u[2]) / u0[1]                                  # dS/dt
    du[2] =  β * u[1] * (u[2]) / u0[1] - γ * u[2] - UDE_term * u[2] / u0[1]  # dI/dt
    du[3] =  γ * u[2] + δ * u[4]                                        # dR/dt
    du[4] =  UDE_term * u[2] / u0[1] - δ * u[4]                         # dT/dt (quarantined)
end

α = p0_vec

# Initial state: population, initial infected/recovered/quarantined counts,
# integration window (days) and the number of data points.
u0 = Float64[60000000.0, 593, 62, 10]
tspan = (0, 95.0)
datasize = 95

prob = ODEProblem{true}(QSIR, u0, tspan)
t = range(tspan[1], tspan[2], length = datasize)

# ---------------------------------------------------------------------------
#  Loss function and training
# ---------------------------------------------------------------------------
#  predict_adjoint solves the UDE for a given parameter vector θ, using an
#  interpolating adjoint so gradients can flow back through the ODE solve.
function predict_adjoint(θ)
    x = Array(solve(prob, Tsit5(), p = θ, saveat = t,
                    sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))))
end

#  We fit in log-space (log of counts) so that both the early exponential
#  growth and the later plateau contribute meaningfully to the loss.
#    - infected data is compared against I + T (active + quarantined)
#    - recovered+dead data is compared against the R compartment
function loss_adjoint(θ)
    prediction = predict_adjoint(θ)
    loss = sum(abs2, log.(abs.(Infected[1:end])) .- log.(abs.(prediction[2, :] .+ prediction[4, :]))) +
           sum(abs2, log.(abs.(Recovered[1:end] + Dead[1:end])) .- log.(abs.(prediction[3, :])))
    return loss
end

# Print the loss every 10 iterations so training progress is visible.
iter = 0
function callback3(θ, l)
    global iter
    iter += 1
    if iter % 10 == 0
        println(l)
    end
    return false
end

# Optimise the neural-network weights and SIR parameters jointly.
# Stage 1: ADAM for a robust, gradient-noise-tolerant warm-up.
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_adjoint(x), adtype)
optprob = Optimization.OptimizationProblem(optf, α)
res1 = Optimization.solve(optprob, ADAM(0.01), callback = callback3, maxiters = 15000)

# Stage 2: BFGS from the ADAM solution for sharper local refinement.
optprob2 = remake(optprob, u0 = res1.u)
res2 = Optimization.solve(optprob2, Optim.BFGS(initial_stepnorm = 0.01),
                          callback = callback3, maxiters = 100)

data_pred = predict_adjoint(res2.u)
p3n = res2.u

# ---------------------------------------------------------------------------
#  Recover Q(t): the neural-network learned quarantine strength
# ---------------------------------------------------------------------------
S_NN_all_loss = data_pred[1, :]
I_NN_all_loss = data_pred[2, :]
R_NN_all_loss = data_pred[3, :]
T_NN_all_loss = data_pred[4, :]

Q_parameter = zeros(Float64, length(S_NN_all_loss), 1)
for i = 1:length(S_NN_all_loss)
    Q_parameter[i] = abs(ann([S_NN_all_loss[i]; I_NN_all_loss[i]; R_NN_all_loss[i]],
                              p3n.layer_1, st1)[1][1])
end

# ---------------------------------------------------------------------------
#  Plot 1: model fit vs. real data
# ---------------------------------------------------------------------------
bar(Infected', alpha = 0.5, label = "Data: Infected", color = "red")
plot!(t, data_pred[2, :] .+ data_pred[4, :], xaxis = "Days post 500 infected", label = "Prediction", legend = :topright, framestyle = :box, left_margin = 5mm, bottom_margin = 5mm, top_margin = 5mm, grid = :off, color = :red, linewidth = 4, foreground_color_legend = nothing, background_color_legend = nothing, yguidefontsize = 14, xguidefontsize = 14, xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)
bar!(Recovered' + Dead', alpha = 0.5, xrotation = 60, label = "Data: Recovered", color = "blue")
plot!(t, data_pred[3, :], ylims = (-0.05 * maximum(Recovered + Dead), 1.5 * maximum(Recovered + Dead)), right_margin = 5mm, xaxis = "Days post 500 infected", label = "Prediction ", legend = :topleft, framestyle = :box, left_margin = 5mm, bottom_margin = 5mm, top_margin = 5mm, grid = :off, color = :blue, linewidth = 4, foreground_color_legend = nothing, background_color_legend = nothing, yguidefontsize = 14, xguidefontsize = 14, xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)

# ---------------------------------------------------------------------------
#  Plot 2: learned Q(t) vs. the date Italy imposed its lockdown
# ---------------------------------------------------------------------------
scatter(t, Q_parameter / u0[1], xlims = (0, datasize + 1), ylims = (0, 1), xlabel = "Days post 500 infected", ylabel = "Q(t)", label = "Quarantine strength", color = :black, framestyle = :box, grid = :off, legend = :topleft, left_margin = 5mm, bottom_margin = 5mm, foreground_color_legend = nothing, background_color_legend = nothing, yguidefontsize = 14, xguidefontsize = 14, xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)

# Locate the inflection point of the learned Q(t) (first sign change of the
# second difference) — this is where the learned quarantine strength starts
# to saturate.
D1 = diff(Q_parameter, dims = 1)
D2 = diff(D1, dims = 1)
Transitionn = findall(x -> x < 0, D2)[1]

# Day 11 is when Italy imposed its national lockdown; overlay both markers.
plot!([11 - 0.01, 11 + 0.01], [0.0, 0.6], lw = 3, color = :green, label = "Government Lockdown imposed", linestyle = :dash)
plot!([Int(Transitionn[1]) - 0.01, Int(Transitionn[1]) + 0.01], [0.0, 0.6], lw = 3, color = :red, label = "Inflection point in learnt Q(t)", linestyle = :dash)

# ---------------------------------------------------------------------------
#  Discussion
# ---------------------------------------------------------------------------
#  The inflection point in the learned Q(t) lands very close to the day Italy
#  actually imposed its national lockdown — even though the model was never
#  told when the lockdown happened. In other words, the neural term recovered
#  a real-world policy signal purely from case-count data, which is what makes
#  the UDE useful as a diagnostic tool and not just a curve fitter.
#
#  More broadly, this shows that pairing neural networks with mechanistic ODE
#  models keeps them expressive *and* interpretable: the learned term can
#  capture complex effects (quarantine, early reopening, vaccination uptake,
#  ...) that are hard to hand-write in a classical compartment model and too
#  opaque to read off an agent-based simulation.
