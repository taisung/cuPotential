# NVIDIA cuPotential blog implications for MD engine integration

Date: 2026-11-17

Updated: 2026-11-18, with facts from the NVIDIA documentation site ([0010](0010_nvidia_documentation_review.md))

Source: NVIDIA Technical Blog, [Accelerate Molecular Dynamics Simulations with Accurate, Flexible Primitives Using NVIDIA cuPotential](https://developer.nvidia.com/blog/accelerate-molecular-dynamics-simulations-with-accurate-flexible-primitives-using-nvidia-cupotential/), published 2026-11-16.

Repository baseline: cuPotential 0.3.5, commit `fbfc4cb6282cd52d33f7adbbcbaeaa5c19b47d74`.

The integration recommendation remains a staged experiment with a single GPU force provider. Numerical accuracy, force field compatibility, and the engine's pressure and parallel execution requirements determine whether that experiment can become a production backend. The [initial assessment](0001_non_openmm_integration_assessment.md) describes the relevant public interfaces and their constraints.

## Relevant published evidence

NVIDIA reports these results for its examples and OpenMM configuration.

| Topic | Evidence reported in the post |
| --- | --- |
| Architecture | Layered C primitives; OpenMM is an example consumer. |
| Throughput | Non-alchemical NPT, roughly 24k to 400k atoms: often around 2x, selected cases above 3x; force, integrator, and barostat change together. |
| Timing setup | AMBER14/TIP3P, mixed precision, 0.9 nm, `5e-4` electrostatic accuracy target, 4 fs, constrained hydrogen bonds, rigid water, 1.5 amu hydrogen mass; Monte Carlo barostat at 1 bar every 25 steps; 500 warmup and 3,000 timed steps. |
| Accuracy | C2 splitting and spreading reduce cutoff or mesh requirements. The grid comparison reports relative energy error and relative L2 force error for 100,000-atom water over 25 frames at 6 and 9 Angstrom cutoffs. Its reference doubles the cutoff and mesh; Ewald with tuned alpha and the u-series are the comparisons. |
| Energy drift | Hundreds-fold reduction over 10 ns relative to default OpenMM PME settings at the same cutoff: 10,000 rigid waters, 2 fs, 300 K; 9 Angstrom versus 20 Angstrom converged PME reference. |
| Dispersion | Thousands-fold energy-error improvement in the illustrated comparison; shorter cutoffs complicate NBFix and combining rules. |
| Tutorial compatibility | CUDA 13.1; LJPME, triclinic geometry, and offsets described as unsupported. |

NVIDIA's [error analysis page](https://docs.nvidia.com/cuda/cupotential/theory/error-analysis) gives the exact conditions of the grid comparison. The reference is OpenMM 8.4.0 in CUDA double precision with a 192³ mesh, a 2.0 nm cutoff, and `alpha` = 4.0 nm^-1, checked with a 256³ mesh on the first frame. The Ewald and u-series parameters stay fixed during the sweep, and every curve uses cuPotential's C2O3 window, so the Ewald curves are not stock OpenMM PME. See [0010](0010_nvidia_documentation_review.md#63-mesh-convergence-study).

The checkout's [README](../README.md) chart supplies versioned throughput evidence: OpenMM 8.4.0 with cuPotential against stock OpenMM, NPT in mixed precision, six systems of 24k to 409k atoms on seven GPUs. The largest speedup on each GPU is between 2.64x and 3.99x. It shares the post's limitation: force, integrator, and barostat change together. The [overview page](https://docs.nvidia.com/cuda/cupotential#benchmarks) of the NVIDIA documentation shows a second chart of the same six systems and seven GPUs, with maxima of 2.66x to 3.93x and no date, OpenMM version, or precision label. Single values differ between the two charts by up to about 0.2 ([0010](0010_nvidia_documentation_review.md#5-the-speedup-chart-on-the-overview-page)).

## How this changes the integration experiment

The C API and scalar functions identified in [0001](0001_non_openmm_integration_assessment.md) provide concrete ways to select an integration layer. An engine with a compatible device force accumulator can use the runtime directly. An engine that must preserve its existing neighbor lists or domain decomposition can incorporate the scalar functions, accepting responsibility for the matched real and reciprocal calculation.

A new engine needs to isolate the contribution of the force backend. The current sample's [`time_steps`](../plugins/openmm/samples/solvated_systems/sample.py#L335) selects the NVIDIA integrator for the plugin arm. Its timings therefore cannot establish what happens when a different engine retains its own dynamics. Benchmark these configurations separately:

1. The engine's existing forces and dynamics, as the baseline.
2. The cuPotential runtime with Ewald splitting and the same engine dynamics, to assess adapter correctness and overhead.
3. Prolate splitting with cuFFT, the full spectrum, and unchanged engine dynamics, comparing cost at matched force accuracy and checking energy drift independently.
4. Stream overlap, graph capture, or other scheduling changes, with the numerical configuration held fixed so each performance difference can be attributed.
5. The compact convolution backend and radial spectrum truncation, as a separate numerical change at matched accuracy, with energy continuity checked across box changes.

The published plugin configuration probably differs from the starting configuration in [0001](0001_non_openmm_integration_assessment.md). The sample converts its source force at [`sample.py`](../plugins/openmm/samples/solvated_systems/sample.py#L128), and that constructor selects Prolate by default; see [`NonbondedForce.h`](../plugins/openmm/cpp/openmmapi/include/nvopenmm/NonbondedForce.h#L179). Without an environment override, the adapter then requests the automatic convolution backend and the radially truncated spectrum for Prolate in [`setupDevice`](../plugins/openmm/cpp/platforms/cuda/src/nonbonded/CudaNonbondedKernels.cu#L854). The post's 500 warmup and 3,000 timed steps match the `time_steps` defaults. Its timings therefore probably include radial truncation and possibly compact convolution, although the post does not state this.

Keep the force field, timestep, constraints, masses, and ensemble settings identical across each direct comparison. The sample also sets OpenMM's `DeterministicForces` property for both arms in [`time_steps`](../plugins/openmm/samples/solvated_systems/sample.py#L359); record and match the corresponding setting in the destination engine. Add coordinate packing, force conversion, neighbor list duplication, and any additional synchronization to the measured timestep cost.

Even a large force-kernel improvement has a limited effect if that force calculation occupies only part of the engine's timestep. For an illustrative engine spending fraction `f` in the replaced work, a speedup `s` of that work gives

```text
ideal timestep speedup = 1 / ((1 - f) + f / s)
```

For example, `f = 0.70` and `s = 3` give an ideal speedup of 1.875 before adapter overhead. These are hypothetical inputs, not cuPotential measurements. Profile the destination engine to obtain the relevant fraction.

The numerical experiment should vary cutoff and mesh dimensions jointly, within the public plan's available settings. [`docs/ltoir.md`](../docs/ltoir.md#L358) limits the interpretation of the automatic Prolate grid calibration: its target concerns combined physical force error, not separate energy, virial, or dispersion guarantees. Use converged references and compare force error, energy differences, and energy drift as separate observables. The published grid comparison reports energy and force errors at fixed positions. The published drift comparison uses default OpenMM PME settings as its baseline, so an engine that already runs tighter PME settings can see a smaller improvement. A short throughput run cannot establish long trajectory stability.

Treat force field exceptions as part of this experiment. Native [`pair.h`](../include/cupo/pair.h) specifies the supported combining rules and factorized reciprocal dispersion weights, while [`dspace.h`](../include/cupo/dspace.h#L59) defines exclusion cancellation. Validate any special type pair interactions and replacement pairs explicitly before reducing the cutoff. Reduced computation is useful only if the intended interaction model is preserved.

## Reconcile the tutorial with the reviewed checkout

The compatibility row above should not replace the versioned code contracts. At this commit, the following capabilities are present in the source and tests; the runtime tests have not been executed in this assessment.

| Capability | Evidence in this checkout | Consequence for another engine |
| --- | --- | --- |
| Attractive dispersion lattice sums | Native P6 plans in [`splitting_plan.h`](../include/cupo/splitting_plan.h#L50); source force conversion in [`NonbondedForce.cpp`](../plugins/openmm/cpp/openmmapi/src/NonbondedForce.cpp#L293); [`test_ljpme.py`](../plugins/openmm/tests/test_ljpme.py#L203) covers both splitting methods and box updates | Evaluate the documented dispersion model and remaining shift, switching, and mesh restrictions. |
| Reduced triclinic periodic geometry | [`cell_list.h`](../include/cupo/cell_list.h#L36) specifies tilt bounds and geometry lifetime; [`test_triclinic.py`](../plugins/openmm/tests/test_triclinic.py#L166) compares electrostatics with a converged reference | Another engine can adapt supported reduced boxes; unrestricted changes of box geometry still require a lifecycle policy. |
| Particle and exception parameter offsets | Conversion code copies particle offsets at [`NonbondedForce.cpp`](../plugins/openmm/cpp/openmmapi/src/NonbondedForce.cpp#L281) and exception offsets at [line 316](../plugins/openmm/cpp/openmmapi/src/NonbondedForce.cpp#L316); [`test_parameter_offsets.py`](../plugins/openmm/tests/test_parameter_offsets.py) and [`test_custom_parameter_updates.py`](../plugins/openmm/tests/test_custom_parameter_updates.py) exercise updates | Parameter update mechanisms exist, but another engine must supply its own parameter laws, ordering, and correction bookkeeping. |
| Release toolchain | [`.cuda-version`](../.cuda-version) records `13.4.3` | Match the distributed runtime's toolkit metadata and ABI when building a new adapter. |

Runtime provenance also matters. [`plugins/openmm/run_build`](../plugins/openmm/run_build#L39) builds the numerical runtime only when a root runtime build script is available. This public checkout has no such script or core runtime source; its normal path downloads release binaries through [`fetch_cupo`](../fetch_cupo). A successful plugin build would not establish that the numerical runtime had been rebuilt from this fork.

These descriptions conflict with the versioned source and tests. The post names no cuPotential release, tag, or commit. Its tutorial uses CUDA 13.1, while release 0.3.5 records 13.4.3 and its [installation requirements](https://docs.nvidia.com/cuda/cupotential/install) specify CUDA 13.4. The tutorial therefore probably describes an earlier release. The [documentation site](https://docs.nvidia.com/cuda/cupotential) describes 0.3.5 and agrees with this checkout. Its [changelog](https://docs.nvidia.com/cuda/cupotential/changelog/openmm) adds LJPME, triclinic boxes, and parameter offsets in 0.3.4, so the tutorial probably describes 0.3.3 or earlier (inferred). There is no basis to infer that the post's timing results include every capability in this checkout. Record the runtime version, ABI, toolkit, enabled backend, and numerical settings alongside any reproduced result.

## Remaining production requirements

The production requirements in [0001](0001_non_openmm_integration_assessment.md#restrictions-that-affect-feasibility) remain: a virial decision for pressure coupling, particle ownership and communication for distributed execution, FP64 nonbonded evaluation, and lambda derivatives. Parameter updates do not supply a derivative of the total energy with respect to those parameters.

Proceed with a fixed box, single GPU prototype if the destination engine can accept the documented precision and data layouts. Resolve virial and decomposition requirements before committing to the high level runtime as the engine's production nonbonded backend. Numerical validation and timing of that adapter remain outstanding.
