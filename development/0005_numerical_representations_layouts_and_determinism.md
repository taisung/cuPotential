# cuPotential numerical representations, array layouts, and determinism

Date: 2026-11-17

Source version: 0.3.5, runtime ABI 32

Reviewed upstream commit: `fbfc4cb6282cd52d33f7adbbcbaeaa5c19b47d74`

The public force API takes interleaved particle coordinates and accumulates forces into separate component planes of signed 64 bit fixed point values. Velocities belong to the calling engine. In the supplied integration, OpenMM owns the compensated position arrays, velocity storage, and force buffer. The plugin consumes those arrays and chooses native double or float-float arithmetic for mixed precision integration.

NVIDIA documents deterministic energies and forces for identical inputs and the same numerical configuration. The repository includes a dedicated test comparing energy hex representations and force bytes across five fresh contexts. Its docstring describes nonbonded energy accumulation through an int64 towerplate accumulator and an int32 fixed point SPME charge grid. This is stronger evidence than the output types alone. The runtime implementation source is absent, so the exact per-thread and within-block operation sequences have not been independently audited. A fixed FFT backend is part of the repeatability condition: timing-based backend selection can choose different algorithms in separate contexts.

## Ownership boundaries

| Owner | Responsibility and representation |
| --- | --- |
| cuPotential runtime | Accepts FP32 nonbonded inputs, manages interaction workspaces, and writes fixed point forces and physical energy outputs through the public C API. |
| OpenMM | Owns `posq`, its mixed precision correction, `velm`, and the shared force buffer in the plugin integration. The fixed point force scale is an OpenMM ABI convention. |
| OpenMM plugin | Binds OpenMM's arrays to cuPotential, schedules calls and dependencies, and chooses native FP64 or `fltflt` integration arithmetic. It also manages private scratch, including `plannedDisplacement`, and a private interpretation of displacement scratch in the `fltflt` path. |

The storage distinction is explicit in [`Constraints.cuh`](../plugins/openmm/cpp/platforms/cuda/src/integrator/Constraints.cuh#L14), while [`common.cuh`](../plugins/openmm/cpp/platforms/cuda/src/integrator/common.cuh#L23) identifies the force scale as an OpenMM ABI constant. Another engine implements its own state storage and binds compatible inputs and outputs to cuPotential.

## Public runtime data contracts

Let `N` be the atom count and `P` the force component stride. The public force APIs allow any supported caller supplied `P >= N`; the interface does not require the engine to adopt OpenMM's particular padding rule.

| Quantity | Representation and array arrangement | Contract |
| --- | --- | --- |
| Nonbonded coordinates | Device FP32 values, interleaved by atom. Atom `i` starts at `i * xyz_per_atom`; the stride is at least three floats. | [`cell_list.h`](../include/cupo/cell_list.h#L108) accepts packed XYZ with stride 3 or records such as `float4` with stride 4. Separate X, Y, and Z arrays need packing. |
| Standard particle parameters | Separate device FP32 records `{charge, sigma, epsilon}`, in input atom order, with a byte stride between records. | [`cell_list.h`](../include/cupo/cell_list.h#L43). Charge must be supplied through the documented parameter or weight binding; a fourth coordinate component does not replace the complete particle record. |
| Custom particle parameters | Opaque byte records with declared size, alignment, and stride. | [`pair.h`](../include/cupo/pair.h) describes the evaluator and its records. Their layout does not define a public charge meaning for a fourth coordinate component. |
| Bonded tuple coordinates | Device `float4[N]` or `double4[N]`, selected at handle creation, aligned to 16 bytes. XYZ contain coordinates and W is ignored. | [`tuples.h`](../include/cupo/tuples.h#L148). The standalone tuple API supports FP32 or FP64 coordinate and evaluation precision. |
| Velocities | No velocity buffer in the force API. | The calling engine selects its velocity representation and performs integration. |
| Forces | Three planes of 64 bit words: X at `[0,P)`, Y at `[P,2P)`, Z at `[2P,3P)`. They represent signed fixed point values with 32 fractional bits. | [`kspace.h`](../include/cupo/kspace.h#L288) uses an unsigned 64 bit pointer for the stored words; tuple execution uses `int64_t*`. Interpret the bit pattern as signed before decoding. |
| Direct and reciprocal energies | Device FP64 scalars in physical energy units, overwritten by the respective calculation. | [`dspace.h`](../include/cupo/dspace.h#L202) and [`kspace.h`](../include/cupo/kspace.h#L259). The dedicated determinism test separately describes internal nonbonded integer accumulation. |
| Tuple energy | A device `double` accumulator receiving an increment. | [`tuples.h`](../include/cupo/tuples.h#L140). Tuple evaluation clears neither force nor energy outputs. |
| Periodic box | Six entries `[ax, by, cz, bx, cx, cy]`; FP32 for nonbonded work and FP64 for periodic tuples. | [`box.h`](../include/cupo/box.h#L13) and [`tuples.h`](../include/cupo/tuples.h#L140). |

Positions and box lengths use nm. Standard LJ epsilon and tuple parameters use the documented kJ/mol conventions. Coulomb plans take a caller supplied coupling, so their energy and force scaling depends on that coupling; custom evaluators also supply their physical factors. A kJ/mol interpretation of combined energies requires a Coulomb coupling and evaluator consistent with the LJ and tuple units. FP64 output storage does not determine its physical units. See [`splitting_plan.h`](../include/cupo/splitting_plan.h#L29) and [`pair_device.cuh`](../include/cupo/pair_device.cuh#L21). Output forces follow input atom order even when cuPotential sorts the cell list internally.

Reciprocal weights need their own binding. [`kspace.h`](../include/cupo/kspace.h#L78) allows explicit `d_charges` weights and [otherwise reads the cell list's `posq.w`](../include/cupo/kspace.h#L94). The public headers do not define a charge interpretation of that internal field for arbitrary custom particle records. The OpenMM adapter [binds explicit Coulomb weights for custom evaluation](../plugins/openmm/cpp/platforms/cuda/src/nonbonded/CudaNonbondedKernels.cu#L820) and [explicit dispersion weights](../plugins/openmm/cpp/platforms/cuda/src/nonbonded/CudaNonbondedKernels.cu#L829); another engine should likewise provide the weights needed by its custom lattice calculation.

The layouts can be pictured as:

```text
Interleaved coordinates, stride 3:
    x0 y0 z0 | x1 y1 z1 | x2 y2 z2 | ...

Interleaved coordinate records, stride 4:
    x0 y0 z0 w0 | x1 y1 z1 w1 | ...

Standard particle records, separate from coordinates:
    q0 sigma0 epsilon0 | q1 sigma1 epsilon1 | ...

Force planes, each with P entries:
    Fx0 Fx1 ... Fx(P-1) | Fy0 Fy1 ... Fy(P-1) | Fz0 Fz1 ... Fz(P-1)
```

For an atom `i < N`, force decoding is:

```text
Fx(i) = signed64(words[i])       * 2^-32
Fy(i) = signed64(words[P + i])   * 2^-32
Fz(i) = signed64(words[2*P + i]) * 2^-32
```

The force allocation occupies `24 * P` bytes. Its fixed point quantum is approximately `2.3283e-10` force units, and its signed physical range is approximately plus or minus `2^31` force units. Scaled contributions and the final destination must fit the documented range. The headers do not specify whether force overflow wraps, saturates, or traps; keep values within range rather than assuming a recovery behavior. The quantum describes storage resolution, not force accuracy or FP64 evaluation.

[`cupo_runtime_capabilities()`](../include/cupo/version.h#L30) reports bit `0x8` for OpenMM compatible fixed64 force output. Check the loaded runtime's ABI and required capability bits when creating an adapter. This bit identifies an output capability; it is not a separate guarantee covering every numerical algorithm or overflow case.

## Accumulator ownership and force addition

Direct space keeps internal accumulators in its workspace. [`cupo_dspace_prepare`](../include/cupo/dspace.h#L169) clears only the requested internal energy and force accumulators; unrequested accumulators retain their prior values. Use the same energy and force flags for prepare and execute.

[`cupo_dspace_finish`](../include/cupo/dspace.h#L196) writes the accumulated energy to a device double scalar and adds forces to the caller's fixed64 destination. It does not clear that destination. Reading retained forces requires the same cell-list build that produced them. Rebuilding the cell list before finishing a retained force result violates that contract even if the underlying arrays have kept their addresses.

Clear a fresh external force destination once before its producers run, then preserve earlier contributions while adding direct, reciprocal, and tuple forces. A second finish call adding to the same destination adds the contribution again. Keep energy bookkeeping separate: direct and reciprocal scalars are overwritten, while tuple energy is incremented.

The `force_scale` argument belongs only to `cupo_dspace_finish`. It multiplies direct force output and leaves energy unchanged. Reciprocal and tuple execution have no corresponding scale argument, so this parameter does not convert the whole calculation's units by itself.

These contracts locate the direct workspace accumulators and the finish boundary that exports fixed64 forces. They do not identify the instruction or per-pair versus per-subtotal granularity of every internal fixed point conversion. NVIDIA's implementation description in the determinism test provides additional evidence, discussed below.

## Precision and storage in the OpenMM plugin

The OpenMM state consumed by the plugin has the following layout. Here `P` denotes OpenMM's padded atom allocation. The arithmetic and displacement scratch rows describe plugin choices rather than a new owner for OpenMM's state buffers.

| Quantity | Single mode | Mixed mode |
| --- | --- | --- |
| Coordinate high part | `float4[P]`, 16 bytes per record | `float4[P]`, 16 bytes per record |
| Coordinate correction | Unused by this integrator | An additional `float4[P]`, 16 bytes per record |
| Velocity and inverse mass | `float4[P]`, `{vx, vy, vz, 1/m}`, 16 bytes per record | `double4[P]`, `{vx, vy, vz, 1/m}`, 32 bytes per record |
| Force buffer | Three signed fixed point 64 bit planes | Same representation |
| Integration arithmetic | FP32 | Native FP64 or `fltflt`, selected by the plugin |
| Displacement scratch | `float4` records | Native FP64 mode uses `double4`; `fltflt` mode uses 32 byte records holding four FP32 high/low pairs. |

The [storage and arithmetic selection](../plugins/openmm/cpp/platforms/cuda/src/integrator/CudaLangevinMiddleKernel.cu#L393) distinguishes these choices. In mixed mode, the integration helpers reconstruct XYZ from the coordinate high part plus its correction. Their [load and store functions](../plugins/openmm/cpp/platforms/cuda/src/integrator/Constraints.cuh#L171) preserve the correction across updates.

The nonbonded adapter passes only the FP32 position array to cuPotential, as shown in [`CudaNonbondedKernels.cu`](../plugins/openmm/cpp/platforms/cuda/src/nonbonded/CudaNonbondedKernels.cu#L1256). Its interface has no separate coordinate correction input. Compensated integration therefore does not make this nonbonded path evaluate positions in FP64. The grouped bonded adapter also explicitly [creates FP32 tuple evaluation](../plugins/openmm/cpp/platforms/cuda/src/bonded/GroupedBonded.cu#L179); the standalone tuple API's FP64 option is a separate capability.

`fltflt` represents a value using two FP32 components, high and low. It increases effective significand precision while retaining FP32 dynamic range; it is not the IEEE FP64 format. See [`fltflt.hpp`](../plugins/openmm/cpp/platforms/cuda/src/integrator/fltflt.hpp#L43). Mixed velocity buffers remain `double4` even when integration arithmetic uses `fltflt`. The displacement scratch can use the high/low representation directly, but velocities retain OpenMM compatible storage; see [`Constraints.cuh`](../plugins/openmm/cpp/platforms/cuda/src/integrator/Constraints.cuh#L123).

Automatic selection uses native double when the device performance ratio and a runtime probe support that choice, otherwise `fltflt`. `CUPO_PLUGIN_MIXED_COMPUTE=double`, `fltflt`, or `auto` controls the selection. Native double and `fltflt` results are explicitly documented in the source as differing in their bits. The plugin [rejects full double nonbonded mode](../plugins/openmm/cpp/platforms/cuda/src/nonbonded/CudaNonbondedKernels.cu#L929); its Langevin integrator also accepts only single or mixed mode.

## Documented arithmetic and grid encodings

The custom pair callback exposes FP32 distance geometry and FP32 energy and derivative results in [`pair_device.cuh`](../include/cupo/pair_device.cuh#L17). Its [compilation contract](../include/cupo/pair_device.cuh#L38) specifies a runtime policy with flush-to-zero and approximate division and square root, and requires external IR to match that policy through `--use_fast_math`. The same contract requires bitwise repeatability of requested results for identical inputs; the compiler flag alone does not establish that property. Expression inputs and prepared values are also explicitly [FP32](../include/cupo/expression.h#L51).

The standalone convolution API documents [FP32 FFT arithmetic and intermediates](../include/cupo/convolution.h#L35). Its real-grid encodings can be FP32, FP16, int16, or int32. For integer encodings, [conversion](../include/cupo/convolution.h#L53) rounds to nearest with ties to even after applying the configured binary scale, uses FP32 arithmetic, saturates integer outputs to their range, and encodes NaN as zero. These are grid conversion rules, not documented overflow rules for the fixed64 particle force buffer. They are also separate from the int32 SPME charge grid named in the nonbonded determinism test.

## What deterministic reduction requires

GPU thread scheduling can vary without affecting a calculation when independent threads update independent state. The numerical concern arises when scheduling or work partitioning changes the grouping of floating point contributions, or when there is a data race.

Floating point addition is not associative. With FP32 values:

```text
a = 100000000, b = -100000000, c = 1

(a + b) + c = 1
a + (b + c) = 0
```

Floating point atomics prevent lost updates but do not establish a fixed arrival order. FP64 arithmetic or compensated summation can reduce error without making arbitrary regrouping produce identical bits.

Integer reduction removes addition-order dependence for identical integer contributions within the supported range. However, the point of conversion matters. Let `Q` denote a specified conversion to fixed point:

```text
sum_j Q(contribution_j)
    Integer addition order does not affect the result.

sum_blocks Q(float_reduce(contributions_in_block))
    The floating point reduction inside each block must still be controlled.
```

The second form is not generally invariant under arbitrary floating point reduction order. Each thread's local accumulation also matters. Converting a floating point block subtotal to `long long` does not undo rounding differences already introduced in that subtotal.

A fixed tree with fixed per-thread traversal can provide repeatability for a fixed partition. It does not establish invariance when block sizes, neighbor order, atom order, or work partitioning change. Stronger invariance requires a reproducible accumulation method across the affected stages, such as exact accumulation or consistently quantized contributions with sufficient integer range.

## Confirmed behavior and remaining evidence gaps

| Stage | Evidence in the public tree | Finding |
| --- | --- | --- |
| Force output accumulation | Fixed point contracts in the direct, reciprocal, and tuple APIs. The plugin [overlaps direct and reciprocal force output](../plugins/openmm/cpp/platforms/cuda/src/nonbonded/CudaNonbondedKernels.cu#L1597). A plugin comment states that both outputs use the same fixed64 atomics; no public header states this. | Adding identical fixed point contributions is independent of stream arrival order. Direct workspace ownership and finish semantics are public contracts; individual internal conversion steps are not exposed. |
| Charge spreading | [`test_determinism`](../plugins/openmm/tests/test_plugin_vs_stock.py#L1266) describes an int32 fixed point SPME charge grid. [`_deterministic_forces.py`](../plugins/openmm/python/nv_openmm_plugin/_deterministic_forces.py#L4) explains that cuPotential's own spreading is unaffected by OpenMM's `DeterministicForces` property. | NVIDIA states an integer accumulation mechanism and tests repeatability under a fixed numerical algorithm. |
| Individual custom pair evaluation | [`pair_device.cuh`](../include/cupo/pair_device.cuh#L38) requires identical requested results for identical geometry, record bytes, globals, and output masks. It requires control of contraction and reassociation. | Repeatable individual contributions are an explicit caller contract. `--use_fast_math` alone is insufficient. |
| Internal force reductions | The [dedicated test docstring](../plugins/openmm/tests/test_plugin_vs_stock.py#L1268) describes int64/int32 fixed point force accumulation. | NVIDIA's implementation statement and bit-level repeatability test are evidence beyond the output type. The precise per-thread and within-block sequences remain unaudited because the runtime source is absent. |
| Nonbonded and bonded energy reductions | The [same docstring](../plugins/openmm/tests/test_plugin_vs_stock.py#L1269) describes nonbonded energy through int64 towerplate accumulation and the int32 SPME grid. It separately describes fixed-order bonded energy reduction. | Integer nonbonded energy accumulation is NVIDIA's stated mechanism, not an inference from `double*`. This does not establish that every bonded energy uses an integer accumulator. |
| Combining nonbonded energy components | [`readEnergy`](../plugins/openmm/cpp/platforms/cuda/src/nonbonded/CudaNonbondedKernels.cu#L1656) waits for producers, then adds direct, Coulomb reciprocal, and dispersion reciprocal scalars in a fixed sequence. | The visible final combination order is controlled. The reduction producing each scalar is a separate question. |
| Ordinary integration updates | [`Constraints.cuh`](../plugins/openmm/cpp/platforms/cuda/src/integrator/Constraints.cuh#L239) fixes expression grouping and explicitly selects rounded FMA operations. Ordinary stages assign each atom its own update. | Varying the execution order of independent atoms need not change their results. Precision, compiler choices, and the selected integration path still matter. |
| Constraint scheduling and fusion | [`FusedChains.cuh`](../plugins/openmm/cpp/platforms/cuda/src/integrator/FusedChains.cuh#L5) groups disjoint atom clusters. Its comments specify mixed mode equivalence to the separate kernels and allow single mode differences from FP32 contraction. | Independent cluster scheduling is distinct from changing the arithmetic implementation. |
| Center of mass momentum removal | [`Constraints.cuh`](../plugins/openmm/cpp/platforms/cuda/src/integrator/Constraints.cuh#L377) uses FP32 momentum sums, 64-thread blocks, indexed block partials, and a specified reduction structure. | The visible code fixes the structure for a given launch. It does not use integer reductions for every global operation or establish invariance across different partitions. |
| cuFFT transforms | NVIDIA documents repeatability with fixed plan input parameters, cuFFT version, and GPU model in the [cuFFT accuracy and performance guide](https://docs.nvidia.com/cuda/cufft/index.html#accuracy-and-performance). | This condition applies to cuFFT. It does not establish identical results across different convolution backends or numerical configurations. |

For an engine that already uses integer force and energy reductions, NVIDIA's determinism test docstring states comparable nonbonded accumulation in cuPotential, together with a fixed-order energy reduction for the bonded path. The public headers and the online guide state that results are deterministic, but they do not describe this mechanism. The remaining audit question concerns the detailed implementation and the scope of order invariance, rather than an absence of any statement about integer energy accumulation.

## Evidence for bit-level repeatability

NVIDIA's [NonbondedForce guide](https://docs.nvidia.com/cuda/cupotential/nonbonded-force) states: "The plugin computes strictly deterministic energies and forces for identical inputs and the same numerical configuration." It also states that OpenMM's `DeterministicForces` property has no effect on the plugin's own calculations. This is a conditional vendor guarantee for the plugin; another engine still needs to preserve the relevant input and execution conditions.

[`test_determinism`](../plugins/openmm/tests/test_plugin_vs_stock.py#L1261) parameterizes single and mixed precision, available water/TYK2/BACE systems, and cuFFT or compact-policy configurations. It creates [five fresh contexts](../plugins/openmm/tests/test_plugin_vs_stock.py#L1289) at identical positions. It compares total potential energies using [`e.hex()`](../plugins/openmm/tests/test_plugin_vs_stock.py#L1304) and complete returned force arrays using [`f.tobytes()`](../plugins/openmm/tests/test_plugin_vs_stock.py#L1311). These are bit-level comparisons of returned results, not tolerance tests or only NumPy value equality. They do not inspect every internal accumulator.

The graph and checkpoint tests are additional evidence. [`test_whole_step_graph.py`](../plugins/openmm/tests/test_whole_step_graph.py#L111) compares scheduling and graph paths; [`test_integrator_checkpoint.py`](../plugins/openmm/tests/test_integrator_checkpoint.py#L67) compares state after checkpoint continuation. Those tests use exact NumPy value equality, while the dedicated determinism test uses energy hex representations and force bytes. No tests were executed for this documentation revision.

## Conditions and limits of determinism

### Hold the FFT algorithm fixed

The [determinism test explains](../plugins/openmm/tests/test_plugin_vs_stock.py#L1274) that timing-based convolution selection can choose different FFT algorithms in separate contexts. It therefore sets `CUPO_COMPACT_AUTOTUNE=0`. The [graph tests](../plugins/openmm/tests/test_whole_step_graph.py#L17) and [checkpoint tests](../plugins/openmm/tests/test_integrator_checkpoint.py#L22) also disable that selection.

For the standalone C API, configure an explicit `CUPO_KSPACE_BACKEND_CUFFT` or supported `CUPO_KSPACE_BACKEND_COMPACT`, holding the mesh and spectrum policy fixed. If tuning an AUTO configuration, record and reproduce the resolved backend for each output mode. [`kspace.h`](../include/cupo/kspace.h#L185) distinguishes potential-only, energy-only, and combined output selections. Comparing those modes requires care if they resolve to different backends.

For the OpenMM plugin, `CUPO_COMPACT_CONVOLUTION=off` forces cuFFT with the full spectrum. `full` requires compact convolution with the full spectrum on a supported device and mesh. These settings can also change mesh planning relative to each other; choose one configuration for a repeatability check rather than expecting them to produce identical bits. The dedicated test uses `off` or `auto` with timing-based selection disabled. Device support and any fallback must match between runs.

[`test_compact_policy.py`](../plugins/openmm/tests/test_compact_policy.py#L82) explicitly uses tolerance comparisons when independently timed contexts choose different FFT backends. This reconciles a deterministic calculation for a fixed algorithm with differing results from automatic algorithm selection. Pin precision, integration arithmetic, numerical parameters, GPU model, runtime, toolkit, and FFT version when requiring repeated bits; cross-configuration equality is not established by these tests.

### Distinguish supported reorders from arbitrary relabeling

| Change | Contract or test evidence | Scope |
| --- | --- | --- |
| OpenMM's supported atom reorders | [`test_forces_survive_a_reorder`](../plugins/openmm/tests/test_reorder.py#L381) requires exact forces and energies after repeated reorders, with graphs on and off. | The test normalizes the compared coordinate snapshots and checks the expected reorder behavior. Its exact-value assertions are intended to check bit-for-bit preservation. |
| Arbitrary heterogeneous particle relabeling | [`test_particle_permutation_preserves_parameter_and_exception_associations`](../plugins/openmm/tests/test_reorder.py#L705) remaps parameters, exclusions, and output order, then [allows native rounding differences](../plugins/openmm/tests/test_reorder.py#L757). | Energy tolerances are `rtol=4e-6, atol=4e-6`; force tolerances are `rtol=4e-6, atol=2e-4`. This is not a bitwise invariance requirement. |
| Reversing a tuple's complete atom list | [`tuples.h`](../include/cupo/tuples.h#L100) explicitly guarantees identical energy and force bits. | This is a specific tuple symmetry guarantee, not a guarantee for every permutation of the system or interaction batches. |

The [reorder test documentation](../plugins/openmm/tests/test_reorder.py#L22) notes that a mixed precision get/set position round trip can change the high-part coordinates evaluated by the force kernels. Equal requested physical positions alone are insufficient if the actual FP32 input arrays differ.

### Preserve the complete trajectory state

A full trajectory also depends on initial positions and velocities, random seeds and counters, force components supplied by the host engine, constraint ordering, and energy dependent decisions such as Monte Carlo volume acceptance. A single-point force and energy test does not cover all of those paths.

The verified [0.3.4 changelog](https://docs.nvidia.com/cuda/cupotential/changelog/openmm) states that binary checkpoints preserve the plugin integrator's RNG state for exact NVT continuation. It supports NPT restart but does not guarantee exact NPT replay because barostat bookkeeping follows OpenMM semantics. Fixed backend and input conditions remain necessary; checkpoint support does not supply a general cross-GPU or cross-version guarantee.

## Validation needed for another engine

1. Start from the nonbonded integer accumulation and fixed-order bonded energy reduction that NVIDIA states in the determinism test docstring. If an implementation audit is required, inspect or request the remaining conversion granularity, per-thread traversal, and within-block reduction details; do not treat the missing runtime source as an absence of repeatability evidence.
2. Pin the FFT backend, mesh, spectrum, output mode, precision, and integration arithmetic before cross-context bit comparisons. Record resolved algorithms after any tuning and separate their numerical differences from reduction-order nondeterminism.
3. At identical actual input arrays and parameters, repeat force and energy calculations across fresh contexts and compare energy bit representations and force bytes, following `test_determinism`. Compare raw fixed64 buffers too when that is the engine's interoperability requirement. Preserve the distinction between exact supported-reorder tests, tolerance-based arbitrary relabeling, and the tuple reversal guarantee.
4. Test the integrator separately with identical force words and random inputs, then test short trajectories and checkpoint continuation with the complete force pipeline. Include configurations with global reductions and energy dependent decisions.

Representation contracts, NVIDIA's implementation statements, and the repository's test assertions were reviewed for this note. The exact vendor guarantee and checkpoint qualification were checked against the linked documentation and its Markdown sources. The runtime binaries were not inspected, and the numerical tests were not executed here. Detailed internal reduction sequences remain unaudited; conditional energy and force determinism is documented and directly covered by bit-level tests.

The public high level force APIs still expose no virial tensor output; see [0001's pressure-coupling assessment](0001_non_openmm_integration_assessment.md#virial-and-pressure-coupling).

Related notes: [integration feasibility](0001_non_openmm_integration_assessment.md), [published benchmark interpretation](0002_nvidia_blog_integration_implications.md), [speedup theory and mechanisms](0003_speedup_theory_and_mechanisms.md), and [interaction coverage and stream scheduling](0004_interaction_scope_and_stream_scheduling.md).
