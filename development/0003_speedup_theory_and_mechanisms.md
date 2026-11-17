# Scientific basis and probable sources of the reported cuPotential speedup

Date: 2026-11-17

Source version: 0.3.5, runtime ABI 32

Reviewed upstream commit: `fbfc4cb6282cd52d33f7adbbcbaeaa5c19b47d74`

Scope: the numerical methods used by the cuPotential OpenMM plugin, and the mechanisms that can explain its reported end-to-end NPT speedup of roughly 2x to 4x over stock OpenMM. The sources are this checkout, the NVIDIA documentation, the NVIDIA blog post examined in [0002](0002_nvidia_blog_integration_implications.md), and the cited literature. No benchmark or profile was run for this note. Statements are labeled as documented (a source states it), calculated (arithmetic in this note), or inferred (a reasoned explanation that still needs a measurement).

The core runtime implementation is absent from this checkout, as [0001](0001_non_openmm_integration_assessment.md) describes. The cell list, direct-space kernels, spreading, and compact convolution are therefore known only through their headers and documentation. The plugin's integrator, constraints, barostat, bonded scheduling, and CUDA graph code are present as source.

## Summary

- The chart in this checkout reports speedups from about 1.9x to 3.99x across six systems and seven GPUs. The largest value on each GPU is between 2.64x and 3.99x. Each plugin run replaces the nonbonded force, the integrator, and the barostat together. No published profile or ablation separates their contributions.
- The numerical changes to the reciprocal calculation are documented and large. In the documented 6.70 nm water box at a 0.9 nm cutoff, the C2 Prolate planner selects a 42³ or 40³ mesh where stock OpenMM uses 60³, which is 2.9x to 3.4x fewer mesh points. The four-point C2O3 window touches 64 mesh points per atom instead of the 125 of fifth-order B-splines. Radial truncation keeps about 12% to 14% of the mesh modes.
- The evidence in this note does not determine how much each change contributes. Amdahl's law only limits the possible combinations (section 4). A 4x total from one component alone needs that component to take more than 75% of the stock timestep and to become `f/(f - 0.75)` times faster: 16x at 80%, 6x at 90%, or 4.75x at 95%. The stock fraction and the component speedups are not measured for these runs. The published CPU results for the same prolate method (about 3x for electrostatics and 2.5x for whole MD runs) do not transfer directly to these GPU runs.
- The plugin source also changes the rest of the timestep (section 3). These changes can contribute, but their share is not measured:
  - One whole-step CUDA graph per step. In NPT with a move every 25 steps, 23 of 25 steps replay it.
  - A one-kernel Langevin integrator with SETTLE and SHAKE fused in. It uses float-float arithmetic on GPUs with slow FP64.
  - A barostat that uses two force passes per move instead of three, skips an unused kinetic energy, and scales molecules in parallel.
  - Grouped bonded terms that overlap the nonbonded work, and no periodic atom reordering.
- The pattern across GPUs and systems is consistent with a contribution from these changes (section 4, inferred). The largest gains are on the GPUs with slow FP64 (L40S, RTX PRO 6000), and on small systems on the fastest data-center GPUs. Pure water gains less than the protein systems on most GPUs.
- Under NPT, the plugin sizes the Prolate mesh for a box 1.15 times larger per axis. That headroom costs about 1.52x in mesh points, so the NPT mesh advantage is smaller than the NVT example suggests.
- The benchmark is close to matched nominal accuracy, at a 5e-4 force target and a 0.9 nm cutoff, but not identical. The error metrics, truncation, center-of-mass removal, integrator arithmetic, and the effect of the shared `DeterministicForces` setting differ between the arms (section 5).
- A docstring in the benchmark harness states: "The plugin is ~1.7x faster than stock per step" ([`test_bench_plugin.py`](../plugins/openmm/tests/test_bench_plugin.py#L906)). The passage does not state the GPU, system, or ensemble, and the harness runs NVT by default. Find the conditions of that number before you compare it with the chart.

## 1. The reported speedups

The repository [README](../README.md) shows [`cupotential-openmm.png`](../plugins/openmm/docs/cupotential-openmm.png): OpenMM 8.4.0 with cuPotential against stock OpenMM, NPT in mixed precision, measured on 12 and 13 November 2026. The values below are read from the bar lengths and are approximate to about ±0.05. Only the per-GPU maxima are printed on the chart.

| GPU | DHFR 24k | TYK2 49k | BACE 70k | ApoA1 92k | Water 100k | Cellulose 409k | Printed maximum |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RTX PRO 6000 Blackwell Server | 3.44 | 3.95 | 3.89 | 3.42 | 3.24 | 3.45 | 3.95x |
| L40S | 3.71 | 3.88 | 3.99 | 3.42 | 3.38 | 2.98 | 3.99x |
| A10 | 2.52 | 2.50 | 2.64 | 2.66 | 2.65 | 3.35 | 3.35x |
| T4 | 1.91 | 2.08 | 2.17 | 2.09 | 2.22 | 2.64 | 2.64x |
| B200 | 3.24 | 3.04 | 2.96 | 2.63 | 2.37 | 2.35 | 3.24x |
| H100 PCIe | 3.34 | 3.09 | 3.19 | 2.74 | 2.63 | 2.78 | 3.34x |
| A100 PCIe | 3.06 | 3.08 | 2.91 | 2.61 | 2.56 | 2.77 | 3.08x |

The simulation settings are those of the post and the [`solvated_systems`](../plugins/openmm/samples/solvated_systems/sample.py) sample: AMBER14 with TIP3P water, a 0.9 nm cutoff, an Ewald error tolerance of `5e-4`, 4 fs steps, constrained hydrogen bonds, rigid water, 1.5 amu hydrogen mass, and a Monte Carlo barostat every 25 steps. The plugin arm converts the source force with the default Prolate splitting; see [0002](0002_nvidia_blog_integration_implications.md#how-this-changes-the-integration-experiment).

## 2. Numerical methods

### 2.1 Ewald splitting and the cost of stock PME

Ewald methods split the Coulomb kernel into a short-range part, summed over pairs inside a cutoff, and a smooth long-range part, summed in Fourier space:

```text
1/r = erfc(alpha r)/r + erf(alpha r)/r
E   = E_direct + E_reciprocal + E_self (+ E_background for a charged cell) - E_excluded
```

Particle mesh Ewald evaluates the reciprocal sum on a mesh [Darden et al. 1993; Essmann et al. 1995]. It spreads charges with order-p cardinal B-splines, transforms the mesh with a 3D FFT, multiplies by an influence function, transforms back, and interpolates forces. The cost is O(N p³) for spreading and interpolation and O(M log M) for a mesh of M points. Deserno and Holm [1998] analyze how the assignment window, aliasing, and influence function determine the accuracy for a given mesh.

OpenMM chooses `alpha = sqrt(-ln(2 delta))/r_c` and, per axis, `n = ceil(2 alpha L / (3 delta^(1/5)))`, rounded up to a legal FFT size, where `delta` is the Ewald error tolerance. The cuPotential Ewald plan uses the same alpha rule ([`splitting_plan.h`](../include/cupo/splitting_plan.h#L29)). OpenMM's PME uses fifth-order B-splines, which touch 5³ = 125 mesh points per atom. The cuPotential Ewald plan also uses a fifth-order window.

Calculated for `delta = 5e-4` and `r_c = 0.9 nm`: `alpha = 2.920 nm^-1`, and the target mesh spacing is 0.1123 nm. The table applies the rule to the benchmark boxes, with 7-smooth rounding as on the CUDA platform. The installed OpenMM 8.1.1 Reference platform gives the same sizes, except 79 for BACE because that platform does not round to a 7-smooth size.

| System | Atoms | Cubic box edge (nm) | Stock PME mesh | Mesh points |
| --- | --- | --- | --- | --- |
| DHFR | 23,558 | 6.223 | 56³ | 175,616 |
| Water box of the NVIDIA drift test | 30,000 | 6.699 | 60³ | 216,000 |
| TYK2 | 48,940 | 7.857 | 70³ | 343,000 |
| BACE | 69,983 | 8.855 | 80³ | 512,000 |
| Water | 100,002 | 10.036 | 90³ | 729,000 |
| ApoA1 | 92,224 | 10.886 | 98³ | 941,192 |

The TYK2, BACE, and water boxes come from [`data/`](../data/), and the DHFR and ApoA1 boxes from OpenMM 8.4.0's `examples/benchmarks` files. The cellulose system is not in this checkout.

### 2.2 Prolate splitting

A prolate spheroidal wave function (PSWF) of bandwidth `c` is, among functions confined to an interval, the one that puts the largest fraction of its energy inside a given frequency band [Slepian and Pollak 1961]. Ewald summation with prolates (ESP) uses this property to replace the Gaussian of Ewald splitting [Liang et al. 2026]. For the same cutoff and error, the long-range part then needs fewer Fourier modes. NVIDIA's [splitting theory page](https://docs.nvidia.com/cuda/cupotential/theory/splitting) credits ESP for the approach.

cuPotential uses a "C2 Prolate" model. Inside the cutoff, the long-range kernel is `u_lr(r) = ell(r/r_c)/r_c`, and outside it is exactly `1/r`. The dimensionless function satisfies `ell(1) = 1`, `ell'(1) = -1`, and `ell''(1) = 2`. The short-range part `1/r - u_lr` and its first two radial derivatives therefore vanish at the cutoff ([`math/splitting.h`](../include/cupo/math/splitting.h#L13), [`docs/ltoir.md`](../docs/ltoir.md#L28)). The design bandwidths are `c = 10.75` for Coulomb (P1) and `c = 12.764031957789484` for dispersion (P6). The Fourier kernel is concentrated at angular wavenumbers up to about `c/r_c`, but it has nonzero oscillatory tails, so `c` is not a hard support limit.

This construction has two consequences.

1. **Fewer mesh points at the same cutoff.** Documented in [`docs/ltoir.md`](../docs/ltoir.md#L338): at a 0.9 nm cutoff, the planner selects 42³ (dense) or 40³ (compact) for a 6.699 nm cubic box. The stock mesh for that box is 60³. Calculated: this is 2.9x or 3.4x fewer mesh points, and a mesh spacing of 0.160 or 0.167 nm instead of 0.112 nm. NVIDIA's [error analysis page](https://docs.nvidia.com/cuda/cupotential/theory/error-analysis) used 45³ against 60³ for the same box, 2.4x fewer points. The ESP paper reports similar factors on CPUs: a 2.6 times larger mesh spacing at a 1e-4 error, and a Fourier transform 6x to 14x shorter at three to four digits of accuracy [Liang et al. 2026].
2. **No discontinuity at the cutoff.** Stock PME truncates the direct term `erfc(alpha r)/r` at the cutoff. Calculated: at the settings above, `erfc(alpha r_c)` is about 2e-4, so each pair that crosses the cutoff changes the energy by about 2e-4 of its bare Coulomb value. The C2 Prolate short-range energy and force go to zero smoothly. Documented: in a 10 ns NVE test of 10,000 rigid waters at 2 fs, the drift was `+1.399e-6 ns^-1` with C2 Prolate at 45³ and `+6.685e-4 ns^-1` with standard OpenMM PME at 60³. The Prolate drift is about 478 times smaller. NVIDIA states that this difference cannot be attributed to the splitting alone, because the meshes and reciprocal rules also differ.

The second consequence concerns accuracy, not speed. It matters here because it shows that the smaller mesh does not buy speed by giving up energy conservation in that test.

### 2.3 The spreading window

With C2 Prolate splitting, cuPotential spreads and interpolates with the C2O3 window. It has approximation order 3, four points per axis, and C² weights, which give C¹ forces for a fixed box and mesh ([`math/spreading.h`](../include/cupo/math/spreading.h#L13), [spreading theory page](https://docs.nvidia.com/cuda/cupotential/theory/spreading)). The deconvolution uses the stencil power averaged over fractional particle positions.

Calculated: each atom touches 4³ = 64 mesh points instead of 125, which is 1.95x fewer mesh updates in spreading and in force interpolation. The ESP paper argues that PSWF-type windows give better accuracy per operation than B-splines, in line with earlier NUFFT work on compact, nearly band-limited kernels [Barnett et al. 2019].

The cuPotential Ewald option keeps the fifth-order B-spline window and the stock mesh rule, and it uses no truncation. A comparison of the plugin's Ewald and Prolate modes therefore measures the whole numerical change (splitting, window, mesh, and truncation) with the rest of the implementation held fixed.

### 2.4 Spectrum truncation and compact convolution

The radial spectrum policy keeps only the modes with `2 pi r_c |k| <= 10.75` (P1 Prolate only; [`kspace.h`](../include/cupo/kspace.h#L145)). The compact backend uses cuFFTDx with packed retained modes. The cuFFT backend applies the same mask but still transforms the full mesh ([`docs/ltoir.md`](../docs/ltoir.md#L373)).

Calculated for the 6.699 nm box at 0.9 nm: the retained sphere has a radius of 12.7 mode indices. It holds 13.5% of the modes of a 40³ mesh and 11.7% of a 42³ mesh. If the planner mesh scales linearly with the box, this fraction is independent of the box size.

Without an environment override, the plugin requests the automatic backend and the radially truncated spectrum for Prolate ([`setupDevice`](../plugins/openmm/cpp/platforms/cuda/src/nonbonded/CudaNonbondedKernels.cu#L854)). It then times compact and cuFFT execution for each output mode and keeps the faster one ([`kspace.h`](../include/cupo/kspace.h#L216)). Three limits apply:

- Compact correctness and performance have been measured only on SM120. The other build targets have not been GPU-qualified ([`docs/ltoir.md`](../docs/ltoir.md#L473)).
- Truncation is an additional approximation. Sharp truncation can cause energy jumps when box changes move modes across the boundary ([`kspace.h`](../include/cupo/kspace.h#L155)). The planner calibration includes compact evaluation with this truncation ([`docs/ltoir.md`](../docs/ltoir.md#L358)).
- Timing-based selection can choose different FFT algorithms across fresh contexts, so bit-identical results require a fixed backend and numerical configuration. The [dedicated determinism test](../plugins/openmm/tests/test_plugin_vs_stock.py#L1274) disables this selection; [automatic-policy tests](../plugins/openmm/tests/test_compact_policy.py#L82) use tolerances when the selected backends differ. Record the resolved algorithm for each output mode alongside performance results.

Inferred: truncation and the fused compact FFT reduce the reciprocal FFT work and memory traffic beyond the mesh reduction. The size of that effect is not published.

### 2.5 Mesh calibration and NPT headroom

The Prolate planner uses an oversampling parameter `u = 1.7` at 0.6 nm and `u = 1.4` at 0.9 nm, interpolated between them. It is calibrated to a `5e-4` relative L2 error in combined Coulomb plus attractive geometric-C6 forces. This is not a separate guarantee for each channel, and accuracy outside 0.6 to 0.9 nm is not qualified ([`docs/ltoir.md`](../docs/ltoir.md#L358)).

With an active barostat, the plugin reserves 1.15 times the box lengths, 1.52 times by volume. Every bundled NPT benchmark is validated at this value ([`NonbondedForce.h`](../plugins/openmm/cpp/openmmapi/include/nvopenmm/NonbondedForce.h#L145)). The plugin plans the Prolate mesh for that enlarged box, but keeps OpenMM's convention for Ewald ([`initialize`](../plugins/openmm/cpp/platforms/cuda/src/nonbonded/CudaNonbondedKernels.cu#L979)). Stock OpenMM keeps the mesh of the initial box.

Calculated, under the assumption that the planner mesh scales linearly with the box: the headroom multiplies the mesh points by about 1.52. The documented 2.9x to 3.4x NVT reduction then becomes about 1.9x to 2.2x in NPT runs. Rounding to allowed FFT sizes moves these values slightly.

### 2.6 Dispersion

The P6 dispersion splitting and LJPME are not part of the reported benchmark. The sample uses ordinary cutoff Lennard-Jones, and the post shows no LJPME timing. The dispersion lattice sum therefore does not contribute to the reported speedup.

### 2.7 Direct space

Both arms use the same 0.9 nm cutoff, so they evaluate the same number of pairs inside the cutoff. The work per pair and the neighbor search differ:

- The Prolate short-range kernel uses polynomial Horner chains of fused multiply-add operations, with no exponential ([`docs/ltoir.md`](../docs/ltoir.md#L519)). OpenMM evaluates `erfc` through an approximation that includes an exponential.
- One neighbor traversal evaluates the cutoff interaction and both splitting corrections ([NonbondedForce guide](https://docs.nvidia.com/cuda/cupotential/nonbonded-force)).
- The cell list is rebuilt from positions on every evaluation into an envelope sized at creation. Graphs therefore replay without reallocation. There is no Verlet skin, so the rebuild is a real per-step cost ([`test_bench_plugin.py`](../plugins/openmm/tests/test_bench_plugin.py#L870)).
- The direct kernel variant can be selected by timing it concurrently with the reciprocal work, through their join ([`dspace.h`](../include/cupo/dspace.h#L121)).

The effect of these differences is not known. The runtime source is absent, and no direct-space timing comparison is published.

### 2.8 Precision and accumulation

The headers document FP32 pair geometry and results ([`pair_device.cuh`](../include/cupo/pair_device.cuh#L17)) and an FP32 reciprocal potential ([`kspace.h`](../include/cupo/kspace.h#L260)). The adapter writes forces directly into OpenMM's CUDA force buffer, which holds 64-bit fixed-point values with 32 fractional bits ([`execute`](../plugins/openmm/cpp/platforms/cuda/src/nonbonded/CudaNonbondedKernels.cu#L1258)). This convention goes back to mixed-precision fixed-point GPU MD [Le Grand et al. 2013].

Both arms therefore share the output representation of the forces. The shared format does not show that their internal reductions, or the cost of those reductions, are the same. NVIDIA's [determinism test](../plugins/openmm/tests/test_plugin_vs_stock.py#L1266) describes int64/int32 nonbonded force accumulation, int64 towerplate energy accumulation, an int32 SPME charge grid, and fixed-order bonded energy reduction. It checks returned energy and force bits with timing-based FFT selection disabled. The runtime source is absent, so the exact conversion granularity, reduction trees, and atomic traffic have not been independently audited. [0005](0005_numerical_representations_layouts_and_determinism.md) separates the documented mechanism, bit-level test coverage, and remaining implementation questions.

## 3. Integrator, constraints, barostat, and launch overhead

These mechanisms are in the Apache-licensed plugin source and do not depend on the splitting. The measured numbers below are quoted from code comments. The comments do not always state the full measurement conditions.

### 3.1 One fused integrator kernel

The Langevin middle scheme [Zhang et al. 2019] runs a kick, velocity constraints, an Ornstein-Uhlenbeck update with drift, position constraints, and a final update in each step. The plugin runs all of these in one kernel for every SETTLE cluster, SHAKE cluster, and unconstrained atom. Intermediate values stay in registers or shared memory, and the position deltas are not written to global memory ([`FusedChains.cuh`](../plugins/openmm/cpp/platforms/cuda/src/integrator/FusedChains.cuh#L5)).

The fallback path uses eight launches per step plus stream events ([`CudaLangevinMiddleKernel.cu`](../plugins/openmm/cpp/platforms/cuda/src/integrator/CudaLangevinMiddleKernel.cu#L938)). The comments explain why separate kernels waste the GPU: they are "too SMALL to fill the GPU … dhfr runs SHAKE as ~13 blocks and SETTLE as ~110 on a 170-SM part" ([line 960](../plugins/openmm/cpp/platforms/cuda/src/integrator/CudaLangevinMiddleKernel.cu#L960)). The fused kernel hands out work units through an atomic counter. In mixed precision, the 3- and 4-atom SHAKE units get an SM to themselves and "run 15-30% faster" ([line 551](../plugins/openmm/cpp/platforms/cuda/src/integrator/CudaLangevinMiddleKernel.cu#L551)).

In mixed precision, the arithmetic type depends on the GPU ([line 393](../plugins/openmm/cpp/platforms/cuda/src/integrator/CudaLangevinMiddleKernel.cu#L393)). The integrator uses native double only when the driver reports an FP32:FP64 throughput ratio of at most 2, and a short probe measures FP64 work at most 3 times slower than FP32 work ([line 405](../plugins/openmm/cpp/platforms/cuda/src/integrator/CudaLangevinMiddleKernel.cu#L405)). Otherwise it uses float-float pairs: "A float-float add is 20 FP32 instructions and a multiply 7". The comment names A100, H100, and B200 as half-rate FP64 GPUs, and sm_86, sm_89, sm_103, and sm_120 as 1/64-rate FP64 architectures. Under this rule, the T4 (sm_75) also uses float-float (inferred). The storage format is the same in both cases. Stock OpenMM's mixed precision computes forces in single precision and integrates in double precision on every GPU.

Two further changes reduce the per-step work:

- The random noise comes from a counter-based Philox generator keyed on atom, step, and seed, so no random-number buffer is generated and read ([line 61](../plugins/openmm/cpp/platforms/cuda/src/integrator/CudaLangevinMiddleKernel.cu#L61)).
- The integrator removes center-of-mass motion itself, every 100 steps by default. Stock OpenMM uses a separate `CMMotionRemover` force "running a two-pass reduction every step" ([`LangevinMiddleIntegrator.h`](../plugins/openmm/cpp/openmmapi/include/nvopenmm/LangevinMiddleIntegrator.h#L24)). The benchmark harness removes that force from the plugin arm only ([`test_bench_plugin.py`](../plugins/openmm/tests/test_bench_plugin.py#L1230)).

### 3.2 Constraints

The plugin reproduces OpenMM's constraint clustering ([`CudaLangevinMiddleKernel.hpp`](../plugins/openmm/cpp/platforms/cuda/src/integrator/CudaLangevinMiddleKernel.hpp#L67)). Rigid waters use the analytic SETTLE solution [Miyamoto and Kollman 1992], with three lanes per water inside the fused kernel ([`SettleLanes.cuh`](../plugins/openmm/cpp/platforms/cuda/src/integrator/SettleLanes.cuh#L5)). A central atom with one to three bonded hydrogens uses SHAKE, Gauss-Seidel with at most 15 sweeps ([`ConstraintKernels.cuh`](../plugins/openmm/cpp/platforms/cuda/src/integrator/ConstraintKernels.cuh#L23)). Other constraint topologies, which stock OpenMM handles with CCMA [Eastman and Pande 2010], are rejected ([`ConstraintClusters.hpp`](../plugins/openmm/cpp/platforms/cuda/src/integrator/ConstraintClusters.hpp#L4)).

The constraint algorithms are therefore the same as stock for these systems. Any gain comes from the fusion and the arithmetic type, not from a different algorithm. The review found no host-side check that SHAKE converged within its 15 sweeps.

### 3.3 Barostat

The Monte Carlo move, the acceptance test, the step-size adaptation, and the random-number seeding are the same as in stock OpenMM ([`MonteCarloBarostatImpl.cpp`](../plugins/openmm/cpp/openmmapi/src/MonteCarloBarostatImpl.cpp#L82)). The cost per move is lower:

- **Two force passes instead of three.** Both Metropolis evaluations also compute forces, and the integrator reuses the forces of the kept configuration. Quoted at [line 98](../plugins/openmm/cpp/openmmapi/src/MonteCarloBarostatImpl.cpp#L98): "Measured on bace, the direct-space kernel costs 87 us for energy alone, 94 for forces alone and 115 for both, i.e. 66 us of the work is SHARED and today gets paid twice."
- **No unused kinetic energy.** The barostat calls `ContextImpl::calcForcesAndEnergy` directly. On apoA1, the kinetic energy that `getState` computes and discards "measured 5.6% of total GPU time across a run" ([line 111](../plugins/openmm/cpp/openmmapi/src/MonteCarloBarostatImpl.cpp#L111)).
- **Parallel coordinate scaling.** The plugin uses one warp per small molecule and one block per large molecule. The stock kernel uses one thread per molecule, and it "measured 725 us per call on a 5090 -- about 4x the entire direct-space force kernel" on apoA1 ([`CudaBarostatKernel.cu`](../plugins/openmm/cpp/platforms/cuda/src/barostat/CudaBarostatKernel.cu#L49)). FP32 centroids reduced the plugin's own centroid time per move from 50.55 to 14.14 us ([line 76](../plugins/openmm/cpp/platforms/cuda/src/barostat/CudaBarostatKernel.cu#L76)).
- **Cheap rejection.** One kernel launch saves or restores positions, velocities, and the fixed-point forces ([line 287](../plugins/openmm/cpp/platforms/cuda/src/barostat/CudaBarostatKernel.cu#L287)). A rejected move therefore needs no new force evaluation ([line 384](../plugins/openmm/cpp/platforms/cuda/src/barostat/CudaBarostatKernel.cu#L384)).

With a move every 25 steps, these savings are spread over 25 steps.

### 3.4 Bonded terms

Harmonic bonds, harmonic angles, periodic torsions, and the nonbonded exception pairs are evaluated by one `cupo_tuples_execute` call in FP32 ([`GroupedBonded.cu`](../plugins/openmm/cpp/platforms/cuda/src/bonded/GroupedBonded.cu#L174)). By default, that call waits for the cell-list sort and then runs alongside the direct and reciprocal work ([`CudaNonbondedKernels.cu`](../plugins/openmm/cpp/platforms/cuda/src/nonbonded/CudaNonbondedKernels.cu#L642)). Stock OpenMM enqueues its bonded kernel in `finishComputation()`. The code gives no bonded timing numbers. The grouped results agree with stock to a relative tolerance of 2e-5 for forces ([`test_grouped_bonded.py`](../plugins/openmm/tests/test_grouped_bonded.py#L137)).

### 3.5 CUDA graphs and host synchronization

- **Whole-step graph.** After a 55-step warmup, one CUDA graph holds OpenMM's force clears, the nonbonded work, the grouped bonded terms, and the fused integrator. Each step is then one `cudaGraphLaunch` with no energy readback ([`CudaLangevinMiddleKernel.cu`](../plugins/openmm/cpp/platforms/cuda/src/integrator/CudaLangevinMiddleKernel.cu#L1159)). A barostat move step and the step after it take the ordinary path ([line 1249](../plugins/openmm/cpp/platforms/cuda/src/integrator/CudaLangevinMiddleKernel.cu#L1249)). Calculated: with a move every 25 steps, 23 of 25 steps replay the graph.
- **Graphs for the barostat's force evaluations.** The nonbonded calculation keeps one graph for each combination of energy and force outputs. A comment explains the gain of the extra slots: with only one slot, "the barostat's two evaluations [ran] at ~356 us each against a 246 us replay of strictly more work" on apoA1 NPT. A one-thread kernel "averaged 32.5 us eager" ([`CudaNonbondedKernels.hpp`](../plugins/openmm/cpp/platforms/cuda/src/nonbonded/CudaNonbondedKernels.hpp#L208)). The box is read through a device pointer, so a barostat move needs no recapture.
- **Fewer host waits.** The energy readback waits until OpenMM has submitted its bonded work. Otherwise the bonded work "costs ~17.7 us of critical path twice per move" ([`CudaNonbondedKernels.cu`](../plugins/openmm/cpp/platforms/cuda/src/nonbonded/CudaNonbondedKernels.cu#L1264)). Cell overflow traps on the device instead of being polled by the host ([line 1498](../plugins/openmm/cpp/platforms/cuda/src/nonbonded/CudaNonbondedKernels.cu#L1498)).
- **No atom reordering.** OpenMM reorders atoms every 250 steps when a force registers a cutoff interaction, and the reorder starts with a blocking download of positions and velocities. The plugin's `NonbondedForce` registers no cutoff interaction with OpenMM. When it is the only nonbonded force, OpenMM therefore does not reorder atoms ([line 1099](../plugins/openmm/cpp/platforms/cuda/src/nonbonded/CudaNonbondedKernels.cu#L1099), [line 1122](../plugins/openmm/cpp/platforms/cuda/src/nonbonded/CudaNonbondedKernels.cu#L1122)).

Multi-step graphs also exist, but they cannot be used with a barostat, so they do not affect the NPT benchmark.

Inferred: these mechanisms remove a fixed cost per step, made of kernel launches, small kernels that leave most SMs idle, and host round trips. That cost weighs most for small systems on fast GPUs.

## 4. How the parts combine

If a fraction `f` of the stock timestep becomes `s` times faster and the rest is unchanged, the total speedup is `1 / ((1 - f) + f/s)`. Calculated values:

| Fraction f of stock time | s = 2 | s = 3 | s = infinity |
| --- | --- | --- | --- |
| 0.30 | 1.18 | 1.25 | 1.43 |
| 0.50 | 1.33 | 1.50 | 2.00 |
| 0.75 | 1.60 | 2.00 | 4.00 |

The fraction of stock OpenMM time spent in each component at these settings is not published, and this note does not measure it. The bound therefore does not decide the attribution. A 4x total from one component alone is possible if that component takes a fraction `f > 0.75` of the stock timestep and becomes `s = f/(f - 0.75)` times faster. For example, `s` is 16 at `f = 0.80`, 6 at `f = 0.90`, and 4.75 at `f = 0.95`. Thus, if the reciprocal calculation took 90% of the stock step and became 6x faster, the total would be 4x with everything else unchanged. The alternative is that several components become faster together, as the mechanisms of section 3 permit. Only component timings of both arms (section 7) can separate these cases.

The chart patterns are consistent with a contribution from the mechanisms of section 3, but they do not measure its size. Each point below is an inference from the approximate chart values. Differences under about 0.1 are within the reading uncertainty.

- **GPU class.** The two largest per-GPU maxima are on the L40S (sm_89, 3.99x) and the RTX PRO 6000 (sm_120, 3.95x). FP64 runs at 1/64 of the FP32 rate on both, so the plugin integrates in float-float while stock integrates in FP64. On A100, H100, and B200, both arms use native double, and the maxima are 3.08x to 3.34x. Compact convolution is also qualified only on SM120. The A10 (sm_86) and T4 (sm_75) also use float-float, but they are smaller GPUs, and their largest gains are on the largest system.
- **System size.** On the three data-center GPUs, DHFR (24k atoms) has the largest or nearly the largest speedup, and water or cellulose the smallest. A fixed cost per step weighs most when a fast GPU runs a small system, and the whole-step graph and fused kernels remove such costs. On the T4 and A10, cellulose (409k atoms) has the largest speedup. Throughput-bound work dominates there, so the reductions in mesh size and stencil points have the most effect.
- **Pure water.** Water is among the two lowest values on five of the seven GPUs. It has no SHAKE clusters, no bonded terms, and only small molecules for the barostat to scale. The mechanisms that target those parts therefore help it less.

These patterns are consistent with the mechanisms, but they do not prove them. Section 7 gives the measurements that would.

## 5. Is the comparison at matched accuracy?

- **Nominal target.** Both arms aim at a relative force error near `5e-4` at a 0.9 nm cutoff. OpenMM's tolerance is an a-priori estimate for Ewald and PME. The cuPotential plan is calibrated against measured combined Coulomb and dispersion force error on sampled systems, including truncation.
- **What NVIDIA measured.** The published mesh convergence comparison reports relative energy error and relative force error at fixed positions, against an OpenMM double-precision reference ([error analysis page](https://docs.nvidia.com/cuda/cupotential/theory/error-analysis)). Its Ewald curves use cuPotential's own mesh code with the C2O3 window, not stock OpenMM PME. The drift test used velocity Verlet with RATTLE at a `1e-8` tolerance, not the benchmark integrator.
- **NPT headroom.** The plugin allocates its cell list and plans its Prolate mesh for an extent 1.15 times larger than the initial box. This is allocation and planning headroom, not an accuracy guarantee. The calibration targets a force error measured on sampled systems, so it does not guarantee the accuracy of every configuration inside that envelope. Stock OpenMM keeps the initial mesh as the box changes.
- **Deterministic forces.** The sample and the benchmark harness set OpenMM's `DeterministicForces` property for both arms ([`time_steps`](../plugins/openmm/samples/solvated_systems/sample.py#L359), [`test_bench_plugin.py`](../plugins/openmm/tests/test_bench_plugin.py#L108)). For stock OpenMM, this replaces floating-point atomic adds in PME charge spreading with fixed-point integers. Its default is false ([`_deterministic_forces.py`](../plugins/openmm/python/nv_openmm_plugin/_deterministic_forces.py#L4)). The plugin's own spreading is fixed point regardless of this property. Its energy and force repeatability also requires the same numerical algorithm, including the FFT backend: independently timed AUTO choices can differ across runs. Hold the backend fixed for bitwise comparisons and record its per-output-mode selection for timings. Measure the cost of the stock property's true and false settings rather than assuming the size or direction of its effect on the reported speedup.
- **Center-of-mass removal.** The stock arm keeps `CMMotionRemover`, which runs every step by default. The plugin arm removes center-of-mass motion every 100 steps. This removes per-step work from the plugin arm only.
- **Integrator arithmetic and noise.** The plugin uses Philox noise, so its trajectories differ from stock and are validated statistically ([`test_integrator_vs_stock.py`](../plugins/openmm/tests/test_integrator_vs_stock.py#L14)). Mixed precision is float-float on some GPUs and double on others, and the two results are not bit-identical. Both arms use a constraint tolerance of `1e-5`. Bonded terms are evaluated in FP32.

## 6. Consequences for another MD engine

The C API delivers the numerical methods of section 2: the Prolate splitting, the C2O3 window, the planner, truncation, and the compact backend. With an unchanged cutoff, the smaller mesh, the smaller window, and truncation reduce the reciprocal work. They do not predict the runtime ratio in another engine on a GPU. FFT sizes, occupancy, memory traffic, and overlap with other work also set that ratio. Treat any reciprocal speedup as a hypothesis until component timings at matched accuracy are available. The end-to-end gain is bounded by the share of the engine's timestep spent in that calculation, as section 4 shows.

The mechanisms of section 3 are in the Apache-licensed OpenMM plugin, not in the C API. Another engine gets them only by implementing similar changes. An engine that already keeps its update, constraints, and force calculation resident on the GPU, with few launches per step, probably gains less from them than OpenMM did (inferred).

The ESP paper also shortens the cutoff to rebalance short- and long-range work [Liang et al. 2026]. The plugin keeps the source cutoff, and the public Prolate calibration covers only 0.6 to 0.9 nm. A shorter cutoff is a further option for another engine, but only with its own accuracy validation.

## 7. Ablation plan

The plugin has switches that isolate each mechanism. Run each configuration for the same system, GPU, and settings as the chart, in both NVT and NPT. Record the time per step, a Nsight Systems profile with time and launch count per kernel, the force error against a converged reference, and the energy drift.

| Step | Configuration | Isolates |
| --- | --- | --- |
| 1 | Stock OpenMM with `DeterministicForces` true, then false | Cost of the deterministic stock baseline |
| 2 | Plugin `NonbondedForce` with the stock integrator and stock barostat. Set `CUPO_PLUGIN_BONDED=0` before the Context is created. Remove nothing else from the System. | The nonbonded force, with its stream overlap, component graphs, and the absence of OpenMM atom reordering (section 3.5). Without `CUPO_PLUGIN_BONDED=0`, the plugin also replaces the stock bonded kernels, because it selects its grouped bonded backend whenever a plugin `NonbondedForce` is in the System ([`GroupedBonded.cu`](../plugins/openmm/cpp/platforms/cuda/src/bonded/GroupedBonded.cu#L343)). The sample states that a plugin force under OpenMM's integrator is a valid configuration ([`sample.py`](../plugins/openmm/samples/solvated_systems/sample.py#L335)). |
| 3 | Step 2 with Ewald splitting, for example `nv.NonbondedForce(source, 1.15, nv.NonbondedForce.PME)` as in [`test_custom_parameter_updates.py`](../plugins/openmm/tests/test_custom_parameter_updates.py#L298) | The whole Prolate numerical change (splitting, window, mesh, and truncation), compared with the plugin's own Ewald implementation |
| 4 | Step 2 with `CUPO_COMPACT_CONVOLUTION` set to `off`, `full`, and `auto` (the default). Record the `[nv-plugin cupo] convolution` lines that the adapter writes to stderr. | Combined effects only. `off` gives cuFFT, the full spectrum, and the dense mesh plan. `full` gives the compact backend, the full spectrum, and the compact mesh plan when a compact mesh is available. `auto` gives the compact mesh plan, the radially truncated spectrum, and a backend selected by timing ([`setupDevice`](../plugins/openmm/cpp/platforms/cuda/src/nonbonded/CudaNonbondedKernels.cu#L845), [`initialize`](../plugins/openmm/cpp/platforms/cuda/src/nonbonded/CudaNonbondedKernels.cu#L984)). Each pair of settings therefore changes more than one factor. |
| 5 | Plugin integrator with the stock barostat, then with the plugin barostat | Integrator, then barostat force reuse and scaling |
| 6 | `CUPO_PLUGIN_FUSED_INTEGRATOR=0`; and `CUPO_PLUGIN_MIXED_COMPUTE=double` on a GPU with 1/64-rate FP64 | Kernel fusion; float-float arithmetic |
| 7 | `CUPO_PLUGIN_WHOLE_STEP_GRAPH=0`, then `CUPO_PLUGIN_CUDA_GRAPH=0` | Whole-step graph; all graphs |
| 8 | The full plugin configuration with `CUPO_PLUGIN_BONDED=0`, compared with the default | Grouped and overlapped bonded terms |
| 9 | Plugin integrator with a `CMMotionRemover` kept at frequency 1 | The center-of-mass asymmetry. This force also disables the whole-step graph ([`CudaLangevinMiddleKernel.cu`](../plugins/openmm/cpp/platforms/cuda/src/integrator/CudaLangevinMiddleKernel.cu#L866)) |

To measure the backend and the truncation separately, keep the mesh dimensions identical:

- **Backend.** In `auto` mode, the adapter prints the compact and cuFFT convolution times that the tuner measured on the same mesh and spectrum ([`setupDevice`](../plugins/openmm/cpp/platforms/cuda/src/nonbonded/CudaNonbondedKernels.cu#L897)).
- **Truncation.** Use a C API harness that creates the reciprocal workspace with fixed `grid_nx`, `grid_ny`, and `grid_nz` ([`kspace.h`](../include/cupo/kspace.h#L91)). With one backend, compare the full spectrum and the radially truncated spectrum.
- **Accuracy.** Measure the energy and force errors of each configuration against the same converged reference.

The switches are documented in the source: [`CudaGraphEnv.hpp`](../plugins/openmm/cpp/platforms/cuda/src/common/CudaGraphEnv.hpp#L13), [`CudaLangevinMiddleKernel.cu`](../plugins/openmm/cpp/platforms/cuda/src/integrator/CudaLangevinMiddleKernel.cu#L416) for the compute type, [`GroupedBonded.cu`](../plugins/openmm/cpp/platforms/cuda/src/bonded/GroupedBonded.cu#L334), and [`setupDevice`](../plugins/openmm/cpp/platforms/cuda/src/nonbonded/CudaNonbondedKernels.cu#L845) for the convolution policy. The benchmark harness also has a `--no-nv-integrator` option, but its plugin arm then keeps the plugin barostat ([`test_bench_plugin.py`](../plugins/openmm/tests/test_bench_plugin.py#L1228)).

The harness stores each measured run as `tests/golden/<GPU>/<run>/{nvt,npt}.json` ([`golden_runs.py`](../plugins/openmm/tests/golden_runs.py#L29)). That tree is not in this checkout. If NVIDIA publishes it, it can give the measurement conditions and per-system numbers behind the chart. The actual cuPotential mesh for each benchmark system is also not known here. Get it from `cupo_plan_grid` with the release runtime that `fetch_cupo` downloads.

## References

- Barnett, A. H., Magland, J., & af Klinteberg, L. (2019). A parallel nonuniform fast Fourier transform library based on an "exponential of semicircle" kernel. *SIAM Journal on Scientific Computing*, 41(5), C479–C504. https://doi.org/10.1137/18M120885X
- Darden, T., York, D., & Pedersen, L. (1993). Particle mesh Ewald: An N·log(N) method for Ewald sums in large systems. *The Journal of Chemical Physics*, 98, 10089–10092. https://doi.org/10.1063/1.464397
- Deserno, M., & Holm, C. (1998). How to mesh up Ewald sums. I. A theoretical and numerical comparison of various particle mesh routines. *The Journal of Chemical Physics*, 109, 7678–7693. https://doi.org/10.1063/1.477414
- Eastman, P., & Pande, V. S. (2010). Constant constraint matrix approximation: A robust, parallelizable constraint method for molecular simulations. *Journal of Chemical Theory and Computation*, 6, 434–437. https://doi.org/10.1021/ct900463w
- Eastman, P., et al. (2024). OpenMM 8: Molecular dynamics simulation with machine learning potentials. *The Journal of Physical Chemistry B*, 128, 109–116. https://doi.org/10.1021/acs.jpcb.3c06662
- Essmann, U., Perera, L., Berkowitz, M. L., Darden, T., Lee, H., & Pedersen, L. G. (1995). A smooth particle mesh Ewald method. *The Journal of Chemical Physics*, 103, 8577–8593. https://doi.org/10.1063/1.470117
- Le Grand, S., Götz, A. W., & Walker, R. C. (2013). SPFP: Speed without compromise—A mixed precision model for GPU accelerated molecular dynamics simulations. *Computer Physics Communications*, 184, 374–380. https://doi.org/10.1016/j.cpc.2012.09.022
- Liang, J., Lu, L., Barnett, A., Greengard, L., & Jiang, S. (2026). Accelerating molecular dynamics simulations using fast Ewald summation with prolates. *Nature Communications*, 17, 6681. https://doi.org/10.1038/s41467-026-73232-8 (preprint: arXiv:2505.09727)
- Predescu, C., et al. (2020). The u-series: A separable decomposition for electrostatics computation with improved accuracy. *The Journal of Chemical Physics*, 152, 084113. https://doi.org/10.1063/1.5129393
- Hopkins, C. W., Le Grand, S., Walker, R. C., & Roitberg, A. E. (2015). Long-time-step molecular dynamics through hydrogen mass repartitioning. *Journal of Chemical Theory and Computation*, 11, 1864–1874. https://doi.org/10.1021/ct5010406
- Miyamoto, S., & Kollman, P. A. (1992). SETTLE: An analytical version of the SHAKE and RATTLE algorithm for rigid water models. *Journal of Computational Chemistry*, 13, 952–962. https://doi.org/10.1002/jcc.540130805
- Slepian, D., & Pollak, H. O. (1961). Prolate spheroidal wave functions, Fourier analysis and uncertainty — I. *Bell System Technical Journal*, 40, 43–63. https://doi.org/10.1002/j.1538-7305.1961.tb03976.x
- Zhang, Z., Liu, X., Yan, K., Tuckerman, M. E., & Liu, J. (2019). Unified efficient thermostat scheme for the canonical ensemble with holonomic or isokinetic constraints via molecular dynamics. *The Journal of Physical Chemistry A*, 123, 6056–6079. https://doi.org/10.1021/acs.jpca.9b02771
