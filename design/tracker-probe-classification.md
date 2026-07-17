# Tracker probe — full per-candidate classification (appendix)

**Generated 2026-07-17** from the registered retrieval (jax 0.11 env, gh CLI). Buckets per
`design/tracker-probe.md`: LH = Long-horizon (counted only with a writable property; the
hit table in the main artifact carries the properties), LH-noprop = long-horizon but the
property test failed (not counted), POINT = point-detectable, USER = genuine misuse,
PERF = performance, FR = feature request/question/docs/infra, UNCLEAR = coverage.

| candidate | bucket | state | comments | title |
|---|---|---|---|---|
| blackjax#219 | FR | closed | 10 | Add arviz plots for MCMC |
| blackjax#220 | FR | closed | 0 | MEADS |
| blackjax#249 | PERF | closed | 2 | Tests take too long to run |
| blackjax#251 | PERF | closed | 6 | pmap seems to drastically improve performance in the example notebook |
| blackjax#259 | UNCLEAR | closed | 3 | Failing to sample from Dirichlet Process Mixture Model using SGLD |
| blackjax#27 | FR | closed | 24 | Add example of how to run on PyMC3 models |
| blackjax#282 | FR | closed | 1 | Refactor the Pathfinder API |
| blackjax#312 | FR | closed | 0 | Set MyST-NB timeouts locally |
| blackjax#383 | FR | closed | 5 | Implement multinomial HMC |
| blackjax#397 | FR | closed | 10 | Adding some basic VI approximation and fitting routine |
| blackjax#400 | FR | closed | 5 | Predator-prey model example |
| blackjax#475 | FR | closed | 2 | 👋 Blackjax Meeting - Feb 2023 |
| blackjax#486 | PERF | closed | 6 | Potential memory leak when running inference loop in another loop |
| blackjax#520 | FR | closed | 2 | Testing strategy  |
| blackjax#529 | PERF | closed | 2 | NUTS out of memory |
| blackjax#601 | FR | closed | 2 | Implement the Schrödinger-Föllmer sampler |
| blackjax#662 | POINT | closed | 2 | test_chees_adaptation fail with jax 0.4.26 |
| blackjax#668 | POINT | closed | 7 | Numerical test `test_chees_adaptation` fails on `aarch64-linux` |
| blackjax#746 | PERF | closed | 3 | Potential Performance due to Jax version |
| blackjax#786 | FR | closed | 0 | Persistent Sampling |
| blackjax#800 | POINT | closed | 3 | Documentation for the quickstart chapter wrong? |
| blackjax#827 | FR | open | 0 | Implement MALT (Metropolis-Adjusted Langevin Trajectories) |
| blackjax#832 | FR | open | 0 | Implement Lie-Trotter / operator-splitting integrators for (SG)HMC |
| blackjax#845 | FR | open | 1 | Implement Slice Sampler Reparameterization Gradients |
| blackjax#871 | FR | closed | 0 | BlackJAX Code Quality & Style Deep Dive |
| blackjax#912 | FR | open | 0 | diagnostics: add rank-normalized + folded R-hat (Vehtari et al. 2021) |
| blackjax#969 | LH | closed | 1 | [Bug] mclmc_adaptation.handle_nans never observes NaNs during tuning. step_size_max does |
| blackjax#973 | LH | open | 1 | MCLMC divergence response has no equilibrium in the position-caused-NaN regime (follow-u |
| blackjaxD#260 | UNCLEAR | open | 7 | Failing to sample from Dirichlet Process Mixture Model using SGLD |
| blackjaxD#370 | UNCLEAR | open | 2 | How to use window_adaptation in  a multi-chain setting? |
| blackjaxD#416 | LH | open | 6 | Predator-prey model example |
| blackjaxD#423 | FR | closed | 1 | Consider moving the examples out of the Blackjax repo |
| blackjaxD#705 | UNCLEAR | open | 3 | Is it normal that the estimated effective sample size goes down as more samples are adde |
| blackjaxD#735 | FR | closed | 3 | Adaptive SMC as in pyMC? |
| blackjaxD#758 | PERF | closed | 3 | Cholesky decomposition performed for every sample(?) |
| blackjaxD#761 | UNCLEAR | open | 2 | Sampling with MCLMC |
| blackjaxD#77 | UNCLEAR | closed | 4 | Dictionaries and tensor chain positions inputs |
| blackjaxD#794 | UNCLEAR | open | 2 | BlackJax MMM model |
| blackjaxD#930 | FR | open | 6 | Connecting BlackJax to PyMC for Parallel Tempering: An applied MP-MCMC prototype |
| blackjaxD#944 | FR | open | 3 | Designing a composable slice-sampling family (converging #943 + #911's hit-and-run, spli |
| diffrax#113 | FR | closed | 13 | Saving metrics during solve |
| diffrax#116 | FR | open | 5 | example on solving system of ODEs  |
| diffrax#119 | UNCLEAR | closed | 2 | Coupled ODEs implementation   |
| diffrax#143 | USER | closed | 12 | Can't return solution of coupled differential equations |
| diffrax#166 | FR | closed | 4 | Can diffrax solve forward-backward SDE？ |
| diffrax#176 | POINT | closed | 2 | NeuralSDE example throws "NotImplementedError: outfeed rewrite closed_call" |
| diffrax#186 | PERF | open | 11 | [question] Computational complexity of integrating / backpropogating through SDE |
| diffrax#194 | LH | open | 3 | ODE solver getting stuck for simple term  |
| diffrax#199 | UNCLEAR | open | 3 | Updating initial guess when using nonlinear solver inside ODE term |
| diffrax#200 | FR | open | 2 | Forcing solver to stay in given region |
| diffrax#207 | LH | closed | 5 | ODE solver fails with 'The maximum number of solver steps was reached. Try increasing `m |
| diffrax#210 | POINT | closed | 2 | `steps=True` results in `max_steps`-sized array with Infs |
| diffrax#218 | UNCLEAR | open | 1 | Problem when using Diffrax for numpyro |
| diffrax#223 | LH | open | 6 | How does diffrax handle state variables becoming inf? |
| diffrax#225 | POINT | closed | 3 | neural_sde example and test_brownian.py throw ValueError: ShapeDtypeStruct: dtype must b |
| diffrax#226 | PERF | closed | 2 | Gradient calculation of ODE solution really slow |
| diffrax#227 | FR | open | 5 | Perform root-finding to tune the final step of an integration with a `DiscreteTerminatin |
| diffrax#228 | FR | open | 8 | IID Brownian motion? |
| diffrax#245 | POINT | closed | 1 | Stiff ODE solvers doesn't work on GPU? |
| diffrax#247 | FR | closed | 7 | Support for Sharding |
| diffrax#249 | FR | open | 1 | Is it possible to obtain the steps of the trajectories? |
| diffrax#255 | FR | closed | 6 | Is there a way to extract values from inside vector_field? |
| diffrax#263 | PERF | closed | 3 | Time taken for Controlled ODEs varies a lot with Initial State |
| diffrax#267 | POINT | closed | 3 | Basic Vmap Bug |
| diffrax#273 | UNCLEAR | closed | 3 | How to implement a coupled SDE |
| diffrax#279 | FR | open | 5 | METAL Backend Support |
| diffrax#280 | FR | open | 3 | Inspection of gradient calculation |
| diffrax#284 | POINT | closed | 6 | "Unexpected tangent. jac_f cannot be autodifferentiated." with Kvaerno3, successful diff |
| diffrax#286 | USER | closed | 2 | `inf` filtering in diffrax solutions |
| diffrax#290 | FR | open | 6 | How to use DiscreteTerminatingEvent to terminate integration as soon as any ODEterm valu |
| diffrax#292 | POINT | closed | 2 | Error "ValueError: No arrays to thread error on to" when vectorizing over diffrax eqsolv |
| diffrax#294 | POINT | closed | 3 | change in `state` for terminating event when using `BacksolveAdjoint()` |
| diffrax#296 | POINT | open | 1 | NewtonNonlinearSolver returns nan when initialized with root value |
| diffrax#301 | FR | open | 5 | SaveAt dense and fn |
| diffrax#309 | FR | closed | 1 | Share computation between drift and difusion |
| diffrax#317 | UNCLEAR | closed | 14 | [Question] Torchsde to diffrax conversion |
| diffrax#319 | FR | open | 7 | Solver for very stiff Neural Ordinary Differential Equations |
| diffrax#323 | FR | open | 1 | Treating Channels that only have missing values in interpolations |
| diffrax#324 | FR | open | 1 | replace_nans_at_start overwrite existing values |
| diffrax#329 | FR | closed | 2 | Adding replicator dynamics as an example |
| diffrax#335 | USER | open | 4 | inf values after triggering event function.  |
| diffrax#341 | UNCLEAR | closed | 2 | Question for second optimization using Diffrax (a possible error) |
| diffrax#355 | FR | open | 2 | Vector of Wiener processes |
| diffrax#361 | FR | open | 5 | What is the best way to integrate a function? |
| diffrax#366 | POINT | closed | 6 | `XlaRuntimeError: UNIMPLEMENTED` when runninq `diffeqsolve` on TPU with x64 enabled |
| diffrax#368 | LH | closed | 16 | Debugging Integration Failures |
| diffrax#374 | FR | open | 5 | Save num_steps At ts |
| diffrax#386 | LH | open | 12 | Explosion of steps for specific parameter values |
| diffrax#397 | USER | closed | 1 | Problem is too stiff to solve? |
| diffrax#403 | FR | closed | 2 | Multi device parallelism |
| diffrax#407 | PERF | open | 22 | Sharding integration is much slower than pmap |
| diffrax#412 | POINT | closed | 5 | Jax 0.4.27 parallelism errors |
| diffrax#416 | POINT | open | 1 | Inconsistency between constant and adaptive step size solvers with Discrete terminating  |
| diffrax#417 | LH | open | 1 | Possible issue with ReversibleHeun solver instability |
| diffrax#43 | FR | open | 0 | `IController`: investigate removing the `stop_gradient`s |
| diffrax#444 | PERF | closed | 3 | Solving ODE with huge inputs / multi-device |
| diffrax#446 | POINT | open | 4 | Can't use Equinox inside `term` |
| diffrax#455 | FR | open | 5 | SDE - share computation common to drift and diffusion terms |
| diffrax#461 | UNCLEAR | open | 7 | Coupled SDE System Implementation |
| diffrax#462 | UNCLEAR | open | 20 | statefully evolving an auxiliary variable |
| diffrax#465 | POINT | open | 2 | RecursiveCheckpointAdjoint not working for two-level minimisation |
| diffrax#472 | FR | open | 4 | Making subsaveat consider previous saves and not just the current one  |
| diffrax#473 | POINT | closed | 5 | Stratonovich/Ito correction |
| diffrax#474 | POINT | open | 8 | Additive SDE throws error with SRK style solvers |
| diffrax#483 | FR | open | 10 | Why are step_ts and jump_ts treated differently here? |
| diffrax#486 | FR | closed | 2 | Parallel Testing |
| diffrax#488 | POINT | open | 3 | Intermediate saved values are sometimes `inf` |
| diffrax#489 | PERF | open | 7 | VBT vs brownian path slowdown |
| diffrax#499 | POINT | open | 4 | Incorrect gradient in toy adaptive ODE |
| diffrax#500 | FR | closed | 3 | Different step sizes within batch? |
| diffrax#507 | LH | open | 3 | Event and PIDController: event doesn't always occure |
| diffrax#510 | POINT | closed | 1 | Example code in SDE section of Getting started produce incorrect result |
| diffrax#513 | POINT | open | 4 | JaxStackTraceBeforeTransformation error with parametrized ODE |
| diffrax#517 | PERF | open | 27 | Performance issue with SDE solver |
| diffrax#518 | PERF | open | 4 | [Regression] Slower integration of differential equations since jaxlib > 0.4.32.dev20240 |
| diffrax#520 | POINT | closed | 2 | Solving NODE with implicit adjoint & steady state fails cause event occurred |
| diffrax#531 | FR | closed | 3 | [question] efficiently simulating multiple SDE trajectories in d > 1 |
| diffrax#538 | POINT | closed | 16 | diffeqsolve terms gives value error from within jax.vmap |
| diffrax#542 | FR | closed | 7 | Allow args into grad_f for ULD |
| diffrax#548 | FR | closed | 5 | UnderdampedLangevinDiffusionTerm inheritance |
| diffrax#549 | PERF | open | 20 | Question: DirectAdjoint is faster than RecursiveCheckpointAdjoint? |
| diffrax#558 | FR | open | 8 | Adjoints question  |
| diffrax#563 | POINT | closed | 2 | Error in diffeqsolve when right-hand side is not defined at time zero |
| diffrax#564 | FR | closed | 5 | Question: working principle of RecursiveCheckpointAdjoint |
| diffrax#570 | POINT | closed | 1 | Residual typing issues in ULD |
| diffrax#572 | FR | closed | 5 | [Question] Difference between a BackSolve Adjoint and a just a lax.scan |
| diffrax#573 | FR | closed | 4 | Example for regression of cde |
| diffrax#574 | PERF | closed | 11 | Memory issue during CNF training |
| diffrax#576 | FR | open | 2 | [Proposal]: Add more Ito SDE Solvers |
| diffrax#592 | PERF | open | 4 | Significant performance difference: diffeqsolve vs. lax.scan - Expected Behavior? |
| diffrax#598 | POINT | closed | 1 | Solving backwards-in-time fails for AbstractSRK's |
| diffrax#602 | PERF | open | 5 | Speeding Up Evaluation of Padded Time Series Using Events? |
| diffrax#605 | PERF | open | 6 | Model is slower using diffrax, and even more when calling a function inside the vector_f |
| diffrax#606 | PERF | closed | 8 | Slow jit of diffeqsolve since v0.6.1 |
| diffrax#612 | POINT | closed | 3 | Error using forward mode autodiff |
| diffrax#621 | POINT | closed | 2 | Example neural SDE code throws ValueError with diffrax v0.7.0 |
| diffrax#625 | PERF | closed | 5 | Performance Issues with SubSaveAt |
| diffrax#630 | POINT | open | 7 | Vmapped jnp.interp throws error when debugging nan's with disabled jit |
| diffrax#632 | LH | closed | 10 | Issue with small time steps |
| diffrax#635 | FR | open | 8 | Steady state solver termination |
| diffrax#636 | FR | closed | 2 | Using jax.export.export to compile and save off a diffeqsolve function |
| diffrax#638 | UNCLEAR | open | 2 | Endless loop after jax.jacrev(ode solver)(args) |
| diffrax#639 | FR | open | 7 | Handling of multiple Events triggered during one integration step |
| diffrax#649 | UNCLEAR | closed | 3 | Noise with Hidden State |
| diffrax#657 | LH | closed | 4 | Odd behaviour of StepTo controller for small time steps |
| diffrax#662 | POINT | closed | 7 | TypeError when using diffrax with JAX v0.7.0 |
| diffrax#663 | POINT | open | 1 | sol.ts contains wrong values in some cases |
| diffrax#664 | POINT | closed | 2 | Cannot cache jitted-diffeqsolve function due to host callbacks |
| diffrax#671 | PERF | closed | 5 | Questions about the performance of the Neural ODE example |
| diffrax#676 | FR | open | 1 | Implicit SDE Solvers? |
| diffrax#685 | PERF | open | 4 | Our of memory : How to optimize the memory use in Diffrax framework? |
| diffrax#697 | FR | open | 2 | How to enforce non-negativity constraints? |
| diffrax#700 | POINT | open | 1 | Cannot reproduce example from stiff ode |
| diffrax#701 | POINT | closed | 3 | NaN gradient only under `vmap` for ControlTerm with NaNs in control |
| diffrax#705 | POINT | open | 3 | Obscure error for passing EulerHeun instead of EulerHeun() |
| diffrax#716 | FR | closed | 4 | Best approach for discontinuous forcing terms with `LinearInterpolation` (without `Contr |
| diffrax#718 | PERF | open | 3 | auto-parallelization alongside `vmap` is very slow |
| diffrax#722 | POINT | closed | 5 | Import side effect: GPU preallocatoin |
| diffrax#737 | FR | closed | 1 | Possible feature: Run diffeqsolve for a fixed number of steps |
| diffrax#738 | FR | open | 1 | Support for Sharding 2.0 2.0 |
| diffrax#740 | FR | open | 1 | Running diffeqsolve for a fixed number of *accepted* steps |
| diffrax#742 | POINT | closed | 2 | Reliable way to debug Nan in backward pass? |
| diffrax#750 | FR | open | 0 | Enforcing Absolute Unitarity in Differentiable Physics: The Float-to-Integer Ledger Appr |
| diffrax#752 | LH | closed | 4 | Nonlinear max steps reached error on implicit solvers with PID step controller |
| diffrax#756 | LH | closed | 5 | Infinite final-step rejection loop due to conflicting endpoint clipping and adaptative R |
| diffrax#757 | POINT | open | 0 | Documentation and implementation of ConstantStepSize do not quite align |
| diffrax#8 | FR | open | 6 | New solvers |
| diffrax#83 | FR | closed | 2 | dt0 is mandatory even for solvers that support adaptive time stepping |
| diffrax#96 | FR | open | 6 | Solving with complex initialization |
| jax-cfd#224 | FR | closed | 1 | Add extra field variables as output for the step function |
| jax-md#116 | FR | closed | 9 | Computing stress |
| jax-md#122 | POINT | open | 4 | Usage of nan_to_num can be in conflict with jax_debug_nans. |
| jax-md#126 | FR | closed | 6 | Benchmarking documentation |
| jax-md#146 | POINT | closed | 1 | Providing floating point values of alpha to energy.soft_sphere_pair results in nan parti |
| jax-md#163 | FR | closed | 2 | Neighborhood lists in free space |
| jax-md#170 | FR | closed | 3 | Ask for NPT ensemble with LJ model with neighbor list |
| jax-md#188 | POINT | closed | 1 | NVT notebook doesn't run |
| jax-md#211 | UNCLEAR | closed | 2 | Force term in NPT pressure |
| jax-md#254 | POINT | closed | 1 | Code snippets in README.md give NaNs |
| jax-md#258 | POINT | open | 2 | NaNs for Lennard Jones potential gradients.  |
| jax-md#264 | POINT | open | 0 | FireDescent should use velocity and not momentum when calculating P |
| jax-md#273 | FR | closed | 1 | Question: Particularities of Autodifferentiation for Forces |
| jax-md#318 | FR | closed | 4 | Question about correctly implementing custom non-conservative force function |
| jax-md#339 | LH | closed | 2 | broken neighbor list/cell list when particle is close to PBC boundary |
| jax-md#362 | FR | open | 3 | Remove center of mass |
| jax-md#384 | UNCLEAR | closed | 2 | Prefactor and masking in Ewald reciprocal-space part |
| jax-md#392 | FR | closed | 1 | Dynamic Metric Scaling and Non-Linear Potentials: A Stretchy Spatial Hashing Approach |
| jax-md#48 | POINT | closed | 1 | scaling typo |
| jax-md#92 | LH | open | 9 | Precision problem with jax.jit |
| jax-md#98 | UNCLEAR | closed | 3 | nans in MD cookbook notebook |
| numpyro#1001 | FR | closed | 0 | Use Gauss-Newton hessian matrix in laplace approximation |
| numpyro#1078 | POINT | closed | 4 | TypeError: Argument 'None' of type '<class 'NoneType'>' is not a valid JAX type |
| numpyro#1133 | LH | closed | 1 | `init_to_uniform` sometimes leads to mangled chains |
| numpyro#1134 | POINT | closed | 4 | `jit_model_args` seems to not work with `chain_method="parallel"` |
| numpyro#1150 | POINT | closed | 1 | `obs_mask` in `sample` seems incompatible with `MultivariateNormal` |
| numpyro#1164 | FR | closed | 6 | Add plated alternative to examples/horseshoe_regression.py |
| numpyro#1170 | UNCLEAR | closed | 2 | Running several chains does not improve accuracy: do you see why? |
| numpyro#1176 | POINT | closed | 1 | Scope handler doesn't add prefix to `cond_indep_stack` frames |
| numpyro#1178 | FR | closed | 7 | Support MatrixNormal distribution |
| numpyro#1184 | POINT | closed | 6 | Truncated distribution is more prone to sampling infinities? |
| numpyro#1186 | POINT | closed | 6 | Sampling of NaNs and -infs from SoftLaplace |
| numpyro#1208 | FR | closed | 12 | Request for Euler Maruyama features in numpyro |
| numpyro#1228 | FR | closed | 8 | ❓ Why is reparametrization required in this example? |
| numpyro#1241 | UNCLEAR | closed | 7 | Need help: NUTS does't converge |
| numpyro#1278 | POINT | closed | 2 | Can't sample posteior predictive, if model uses no covariates |
| numpyro#1285 | POINT | closed | 7 | Potential shape bug in mcmc.get_samples() |
| numpyro#1286 | FR | closed | 4 | How to make variable name incremental inside scan? |
| numpyro#1292 | PERF | closed | 6 | Out-of-memory error on TPUs - scaling Bayesian CNNs |
| numpyro#1293 | FR | closed | 1 | Raise better error message when using HMC for models with subsample |
| numpyro#1309 | FR | closed | 14 | Correct SVI API usage |
| numpyro#1340 | POINT | closed | 4 | Rhat and NEff are NaNs with NUTS |
| numpyro#1360 | LH | closed | 20 | NUTS sometimes does not converge on a regression model |
| numpyro#1368 | FR | closed | 3 | log1m_exp and log_diff_exp functions |
| numpyro#1397 | FR | closed | 3 | Resume the mcmc.run() training   |
| numpyro#1416 | POINT | closed | 1 | signature error when running SA MCMC sampler in parallel |
| numpyro#144 | FR | closed | 7 | API issues to clean up before release |
| numpyro#1446 | FR | closed | 5 | Example code for using random_flax_module with flax.linen.BatchNorm (mutable) |
| numpyro#1453 | USER | closed | 1 | Unable to fit MA(1) time series model when `theta` is greater than 1. |
| numpyro#1457 | FR | closed | 2 | SIR Agent Based Model on a Contact Network |
| numpyro#1460 | UNCLEAR | closed | 1 | Problem whit initial parameters |
| numpyro#1485 | FR | closed | 17 | Pathfinder |
| numpyro#1488 | POINT | closed | 4 | Inconsistent MCMC Results Based On * Operator |
| numpyro#1492 | POINT | closed | 4 | inf's with TruncatedNormal |
| numpyro#1499 | FR | closed | 1 | constant parameters in IRT model |
| numpyro#1511 | POINT | closed | 5 | SineBivariateVonMises generates NaN log probability for negative correlation parameter |
| numpyro#1523 | POINT | closed | 1 | LocScaleReparam samples outside the support |
| numpyro#154 | LH-noprop | closed | 15 | NUTS doesn't converge on a stan model |
| numpyro#1578 | POINT | closed | 2 | An error appears if I include the `mask`. |
| numpyro#1596 | POINT | closed | 2 | Version 0.12.0 does not work with jax 4.11 |
| numpyro#1602 | FR | closed | 1 | Estimate the free parameters of the Q learning model using MCMC inference |
| numpyro#1606 | POINT | closed | 5 | Sampling from `TruncatedNormal` returns infinities when location parameter is outside bo |
| numpyro#1639 | POINT | closed | 2 | Pareto distribution log_prob not -inf below scale |
| numpyro#1671 | POINT | closed | 8 | Bug in Kumaraswamy distribution? |
| numpyro#1677 | FR | closed | 3 | Enforcement of arg_constraints / parameter constraints |
| numpyro#1689 | PERF | closed | 3 | Reducing GPU memory usage |
| numpyro#1695 | FR | closed | 16 | Sample from distribution without storing |
| numpyro#1696 | FR | closed | 22 | Distributions Entropy Method |
| numpyro#1729 | FR | closed | 0 | [FR] Add constraints.greater_than_eq  |
| numpyro#1780 | FR | closed | 5 | [FR] Support for different supports in component distributions for mixture models |
| numpyro#1786 | LH-noprop | open | 10 | mean_accept_prob significantly different after warmup |
| numpyro#1813 | POINT | closed | 3 | Large potential energy while using `HMCGibbs` at the initial stage |
| numpyro#1820 | UNCLEAR | closed | 2 | Got Problems When Computing Log Likelihoods in a Scan-Based VAR Model |
| numpyro#1833 | FR | open | 2 | Stress test utility for numpyro? |
| numpyro#1838 | POINT | closed | 4 | `nuts.get_extra_fields()["num_steps"]=0` after warmup |
| numpyro#1870 | POINT | closed | 3 | Grads w.r.t. weights of `MixtureGeneral` Distribution are giving `nan`s |
| numpyro#1872 | FR | open | 5 | Support constraints.cat and CatTransform |
| numpyro#1937 | FR | closed | 7 | icdf not implemented in Truncated Distribution |
| numpyro#1954 | PERF | closed | 1 | Batching MCMC OOM issue |
| numpyro#1955 | FR | closed | 6 | [FR] Support for optax.contrib.reduce_on_plateau |
| numpyro#2008 | PERF | closed | 5 | potential mach ports leakage |
| numpyro#2062 | POINT | open | 4 | Jax throws error when tracing auto guide after passing it to get_model_relations |
| numpyro#2068 | FR | closed | 0 | Support forward-mode differentiation with autoguide. |
| numpyro#2072 | FR | closed | 3 | Add cdf method for truncated distributions |
| numpyro#2088 | POINT | closed | 2 | Beta with concentration1=1 gives nan log_prob at value=0 |
| numpyro#2093 | POINT | closed | 1 | Continuous uniform density gives density >0 outside of support |
| numpyro#2130 | POINT | closed | 5 | InverseWishart not found |
| numpyro#2181 | POINT | closed | 4 | Cast rate to float when constructing Poisson to ensure consistent log prob behavior |
| numpyro#2193 | FR | open | 5 | Handle zero-mean Negative Binomial distributions |
| numpyro#236 | FR | closed | 0 | initialize_model should return valid initial params |
| numpyro#245 | FR | closed | 9 | How to use residual based boostrap with numpyro? |
| numpyro#249 | LH | closed | 1 | HMC adaptation stuck at warmup phase due to step_size -> nan |
| numpyro#251 | PERF | closed | 9 | HMC performs poorly for dual moon potential_fn when progbar=True |
| numpyro#373 | FR | closed | 2 | Masking out of support log_prob |
| numpyro#404 | POINT | closed | 6 | Uniform distribution |
| numpyro#440 | POINT | closed | 4 | Error in bnn.py for num-chains > 2 |
| numpyro#45 | FR | closed | 0 | Disable generic args checking for discrete distributions |
| numpyro#462 | FR | closed | 10 | numpyro equivalent to pymc3's Deterministic |
| numpyro#48 | FR | closed | 13 | Implement pathwise derivative for dirichlet distribution |
| numpyro#484 | FR | closed | 14 | bnn.py example with another artificial dataset |
| numpyro#492 | FR | closed | 8 | Import MCMC into Arviz |
| numpyro#505 | POINT | closed | 9 | GPU version install |
| numpyro#534 | FR | closed | 28 | Sequential Sampling Strategy |
| numpyro#539 | PERF | closed | 21 | GPU Memory |
| numpyro#543 | PERF | closed | 8 | Very slow compile on simple model |
| numpyro#545 | POINT | closed | 12 | Obscure NotImplementedError for Categorical |
| numpyro#549 | FR | closed | 2 | Sampling with invalid observations (e.g. Poisson(1|-1))? |
| numpyro#552 | LH | closed | 9 | "Cannot find valid initial parameters" after data size exceeds some threshold |
| numpyro#563 | POINT | closed | 3 | Numerical issues with GammaPoisson (i.e, negative binomial) for small rates |
| numpyro#568 | POINT | closed | 8 | mask handler does not behave as expected |
| numpyro#594 | POINT | closed | 3 | AutoContinuous not working correctly with TransformedDistribution |
| numpyro#604 | FR | closed | 2 | Single-site MCMC |
| numpyro#643 | FR | closed | 22 | Best method for complicated multimodal posteriors |
| numpyro#663 | POINT | closed | 7 | TransformedDistribution broken in MCMC? |
| numpyro#698 | POINT | closed | 1 | handlers.mask not working when enable_validation(True) |
| numpyro#705 | FR | closed | 22 | Integrate with flax/haiku |
| numpyro#724 | FR | closed | 18 | HMCECS in numpyro |
| numpyro#726 | FR | closed | 3 | Bayesian imputation tutorial with discrete covariates |
| numpyro#735 | POINT | closed | 4 | Set platform to GPU but Volatile GPU-Util is still 0% |
| numpyro#744 | POINT | closed | 4 | Log probability calculation with arviz conversion goes wrong with enumeration |
| numpyro#771 | POINT | closed | 3 | Omnistaging does not work when enabling distributions' validation |
| numpyro#838 | POINT | closed | 3 | Broken install under Python 3.9 |
| numpyro#859 | POINT | closed | 5 | VonMisesFisher samples NaNs with mean_direction=[1., 0., 0.] |
| numpyro#871 | POINT | closed | 1 | Gradient propagation failure for masked invalid values |
| numpyro#88 | PERF | closed | 7 | Avoid compiling 2 times in HMC |
| numpyro#903 | FR | closed | 8 | Left skewed distribution |
| numpyro#936 | PERF | closed | 14 | Excessive memory usage at the beginning of NUTS inference |
| numpyro#956 | FR | closed | 2 | Skip current update in SVI if loss gets nan |
| numpyro#982 | FR | closed | 2 | How to find MAP/maximum likelihood estimation using the minimizer (i.e., BFGS) |
| numpyro#999 | UNCLEAR | closed | 3 | BFGS always failed but Adam works |
