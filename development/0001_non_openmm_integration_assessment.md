# cuPotential integration assessment for other MD engines

Date: 2026-11-17

Updated: 2026-11-18, with facts from the NVIDIA documentation site ([0010](0010_nvidia_documentation_review.md))

Source version: 0.3.5, runtime ABI 32

Reviewed upstream commit: `fbfc4cb6282cd52d33f7adbbcbaeaa5c19b47d74`

Scope: integration into an MD engine whose simulation loop is independent of OpenMM. No particular destination engine has been selected.

cuPotential can supply force calculations to another MD engine through its CUDA C API. The most practical first target is a single GPU simulation with coordinates and forces kept on the device, a fixed particle count, and ordinary periodic Coulomb and Lennard-Jones interactions. The engine can retain its integrator, constraints, thermostat, topology readers, and existing bonded calculations.

A production integration becomes substantially harder when the engine requires a virial tensor, distributed domain decomposition, frequent changes to particle indexing, or double precision nonbonded evaluation. The public high level API has no virial output or distributed particle ownership interface, and its nonbonded inputs are FP32. Those requirements should determine the integration layer before implementation starts.

The companion [NVIDIA blog assessment](0002_nvidia_blog_integration_implications.md) examines the published performance and accuracy evidence and reconciles its feature descriptions with this checkout. Its benchmark interpretation should accompany any performance expectations for a new engine backend.

## What the repository provides

The reusable interface is in [`include/cupo/`](../include/cupo/). The OpenMM adapter, integrator, and barostat are separate source components under [`plugins/openmm/`](../plugins/openmm/). NVIDIA's [C API getting started guide](https://docs.nvidia.com/cuda/cupotential/c-api/getting-started) explicitly supports standalone C and C++ applications without OpenMM or Python.

| Component | Available functionality | Integration consequence |
| --- | --- | --- |
| [`pair.h`](../include/cupo/pair.h) and [`dspace.h`](../include/cupo/dspace.h) | Built in Coulomb and LJ interactions, Lorentz-Berthelot or geometric LJ combining, optional LJ switching, exclusions, and custom radial pair evaluators | Can replace a supported nonbonded force calculation. Direct space uses a cuPotential cell list. |
| [`kspace.h`](../include/cupo/kspace.h) and [`splitting_plan.h`](../include/cupo/splitting_plan.h) | Coulomb P1 and attractive dispersion P6, Ewald or Prolate splitting, grid planning, spreading, convolution, and force interpolation | Can replace the matching reciprocal calculation. The application adds self and background energy corrections. |
| [`expression.h`](../include/cupo/expression.h) and [`pair_device.cuh`](../include/cupo/pair_device.cuh) | Symmetric radial energy expressions or a device evaluator supplied as LTO IR | Covers additional pair potentials and parameterizations. The callback takes distance geometry and particle records; it does not provide a general many body or orientation dependent force interface. |
| [`tuples.h`](../include/cupo/tuples.h) | Harmonic bonds and angles, periodic torsions, and unsplit Coulomb/LJ replacement pairs | Supports selected bonded terms and scaled special pairs. Other force terms remain the engine's responsibility. |
| [`math/splitting.h`](../include/cupo/math/splitting.h) and [`math/spreading.h`](../include/cupo/math/spreading.h) | Host and device scalar functions in FP32 and FP64 | Allows an engine to incorporate the math into its own pair and mesh kernels. |
| [`convolution.h`](../include/cupo/convolution.h) | Standalone GPU convolution with configurable Fourier support | Allows an engine to retain its spreading, gathering, and reductions while replacing convolution. |

The core numerical runtime is distributed as `libcupo.so`; its implementation is absent from this checkout. The repository supplies public headers and the OpenMM adapter source. [`fetch_cupo`](../fetch_cupo) downloads the runtime and scalar objects from a release. A fork can change the adapter and examples, but cannot rebuild unavailable runtime implementation code.

The [component licensing overview](../LICENSE) assigns Apache 2.0 to the plugin source and examples, and the NVIDIA proprietary software license to the runtime, public API headers, and LTO IR objects. Preserve that distinction when packaging a new engine backend.

## Available integration approaches

| Approach | Engine retains | Engine delegates | Assessment |
| --- | --- | --- | --- |
| Use the high level runtime | Simulation loop, constraints, thermostat, and optionally bonded forces | Cell list, direct pair forces, reciprocal forces, and optionally supported tuples | Best starting point for a single GPU engine. Requires adaptation of data layouts and force conventions. |
| Incorporate scalar math | Neighbor lists, particle ordering, communication, force reduction, mesh infrastructure, and virial calculation | Splitting, spreading, interpolation, and Fourier scalar functions | Better architectural fit when the engine already has strong GPU and distributed infrastructure. Requires implementation and validation of the complete matching numerical decomposition. NVIDIA designed these functions "for integration of cuPotential's splitting, spreading, and interpolation math into kernels developers have already built" ([C API reference](https://docs.nvidia.com/cuda/cupotential/c-api/reference#scalar-functions)). |
| Use standalone convolution | Particle handling, spreading, gathering, influence construction, energies, and virials | Three dimensional convolution | A narrower experiment for an engine with an existing mesh pipeline. Distributed FFT communication remains the engine's responsibility. |

For an initial backend, use the high level runtime for nonbonded energy and forces while retaining the engine's bonded terms and dynamics. Existing engines that must preserve distributed decomposition or pressure coupling may obtain a better fit from the scalar layer despite the greater numerical implementation work.

The OpenMM [`setupDevice`](../plugins/openmm/cpp/platforms/cuda/src/nonbonded/CudaNonbondedKernels.cu#L709) and [`dispatchStep`](../plugins/openmm/cpp/platforms/cuda/src/nonbonded/CudaNonbondedKernels.cu#L1550) functions are useful implementation references. Their stream and buffer management depends on OpenMM, so another engine needs its own adapter around the same C calls.

## Data and execution contract

| Item | cuPotential requirement | Work for another engine |
| --- | --- | --- |
| Nonbonded positions | Device FP32 coordinates, with a stride of at least three floats per atom | Reuse compatible interleaved storage or pack a device staging array. Separate x, y, and z arrays need packing. FP64 coordinates need conversion for this path. |
| Standard particle records | FP32 `{charge, sigma, epsilon}`, with nm and kJ/mol conventions | Convert force field units and parameter representation. The fourth coordinate component is not a substitute for the required complete parameter records. |
| Lengths and Coulomb coupling | Plan cutoffs and grid planning box lengths in nm. Each Coulomb plan takes its coupling constant as an argument | Convert Angstrom box lengths and cutoffs before plan creation and `cupo_plan_grid`. Prolate sizing selects its calibration at fixed physical lengths, so it is not invariant to length units; see [`docs/ltoir.md`](../docs/ltoir.md#L351). Pass a coupling consistent with the nm and kJ/mol conventions; the OpenMM adapter uses 138.93545764498 kJ nm/(mol e^2) in [`InteractionModel.hpp`](../plugins/openmm/cpp/platforms/cuda/src/nonbonded/InteractionModel.hpp#L18). |
| Dispersion reciprocal weights | Device FP32 `sqrt(C6)` weights, `2*sqrt(epsilon)*sigma^3`, in input atom order. A dispersion workspace sharing a cell list that contains Coulomb charges needs them as explicit `d_charges` | Compute and own the array. Update it with parameter changes and atom permutations, and keep it unchanged between spreading and interpolation; see [`kspace.h`](../include/cupo/kspace.h#L78). |
| Nonbonded box | FP32 `[ax, by, cz, bx, cx, cy]`, orthorhombic or reduced lower triangular triclinic | Pack and, if necessary, transform the engine's box and coordinates consistently. Geometry selected at creation limits later updates. |
| Force output | Three planes of signed fixed point 64 bit values with 32 fractional bits, stored through the documented integer pointer types | Accumulate directly into a compatible force buffer, or convert a cuPotential scratch buffer on the GPU and add it to the engine's forces. Decode as signed before multiplying by `2^-32`. |
| Energy output | FP64 device scalars | Direct and reciprocal calls overwrite their respective energy scalars. Tuple execution adds to its scalar. Initialize and combine them accordingly. |
| Workspace storage | Direct and reciprocal arenas are caller owned and aligned to 256 bytes. A cell list allocates internally when its arena pointer is null. | Allocate once per workspace. Reciprocal size queries exclude FFT and convolution backend owned allocations. |
| CUDA execution | Caller selects the device and supplies streams and dependencies | Order input updates, calculation, and force consumption explicitly. Success from an asynchronous API means work was enqueued. |
| API conventions | Zero-initialize each versioned descriptor, then set `struct_bytes` and `abi_version`. `cupo_last_error()` is thread-local: a failed call replaces it and a successful call clears it. C++ exceptions do not cross the C API boundary ([C API reference](https://docs.nvidia.com/cuda/cupotential/c-api/reference#runtime-and-data-conventions)). | Copy the error message in the adapter right after the failed call, on the same thread. |

See the contracts in [`cell_list.h`](../include/cupo/cell_list.h), [`dspace.h`](../include/cupo/dspace.h), and [`kspace.h`](../include/cupo/kspace.h). Fixed point accumulation does not make the underlying nonbonded evaluation double precision. Scaled contributions and accumulated forces must fit the signed fixed point range, approximately plus or minus `2^31` physical force units.

Tuple execution has a different input contract: positions are aligned `float4` or `double4` records, and periodic boxes use device `double[6]`. A backend using both tuples and nonbonded evaluation may need two box representations. Tuple precision can be FP64 even though the high level nonbonded path uses FP32.

The public direct space API consumes a `cupo_cell_list_handle`; it has no function for importing the engine's arbitrary neighbor list. Retaining other cutoff force calculations can therefore duplicate neighbor list work. This cost belongs in the performance comparison.

## A minimal force evaluation

During initialization, check `cupo_abi_version()` against `CUPO_ABI_VERSION`, inspect runtime capabilities, create splitting plans and the pair descriptor, allocate the cell list and workspaces, and initialize reciprocal influence tables. Keep the splitting plans if later self or background queries will use them; workspaces otherwise copy their values at creation. Perform linking and tuning before graph capture. With the automatic convolution backend, call `cupo_kspace_tune_convolution` after `cupo_kspace_refresh_influence`; see [`kspace.h`](../include/cupo/kspace.h#L216). Call `cupo_dspace_tune_with_kspace` or `cupo_dspace_tune_with_kspaces` before the first direct execution. The runtime rejects direct tuning that is still pending during capture; see [`dspace.h`](../include/cupo/dspace.h#L121). Without the tuning hook, run one uncaptured evaluation before capture, which keeps direct-only tuning.

Without `cupo_kspace_configure_convolution`, a reciprocal workspace uses cuFFT with the full mesh spectrum ([C API reference](https://docs.nvidia.com/cuda/cupotential/c-api/reference#reciprocal-space)). The OpenMM plugin requests the automatic backend and radial truncation for Prolate ([0002](0002_nvidia_blog_integration_implications.md#how-this-changes-the-integration-experiment)). An engine that keeps the default therefore does not run the configuration of NVIDIA's benchmarks. Timing-based convolution tuning is a performance choice, not a fixed algorithm across fresh workspaces. For bitwise comparisons, configure an explicit cuFFT or supported compact backend with the same mesh and spectrum, or reproduce the resolved backend for each output mode after tuning. The OpenMM [determinism test](../plugins/openmm/tests/test_plugin_vs_stock.py#L1274) disables timing-based selection with `CUPO_COMPACT_AUTOTUNE=0`; [automatic-policy tests](../plugins/openmm/tests/test_compact_policy.py#L82) permit numerical tolerances when independent contexts choose different FFT backends. See [0005](0005_numerical_representations_layouts_and_determinism.md#hold-the-fft-algorithm-fixed) for the determinism conditions.

The following is a sequential execution sketch, not a compilable example. All device operations use one engine owned stream `s`, and the particle order remains fixed:

```text
pack or reuse current device positions, parameters, and box
if box changed:
    cupo_cell_list_update_box(...host box...)
    upload the corresponding device box on s
    cupo_kspace_refresh_influence(...device box..., s)

clear cuPotential scratch force buffer on s
cupo_dspace_prepare(...requested energy and force flags..., s)
cupo_cell_list_execute(...positions, parameters..., s)
cupo_dspace_execute(...matching output flags..., s)
cupo_dspace_finish(...direct energy, scratch forces..., s)
cupo_kspace_execute(...reciprocal energy, scratch forces..., s)
evaluate replacement pairs through the engine or cupo_tuples_execute
convert and add scratch forces to the engine force buffer on s
order the engine's next force consumer after s
```

Run a second reciprocal workspace when using both Coulomb and dispersion, and bind its explicit `sqrt(C6)` weights as described in the data contract. The OpenMM adapter does so in [`setupDevice`](../plugins/openmm/cpp/platforms/cuda/src/nonbonded/CudaNonbondedKernels.cu#L829) and reorders the weights in [`applyAtomReorderImpl`](../plugins/openmm/cpp/platforms/cuda/src/nonbonded/CudaNonbondedKernels.cu#L1172). The header does not state that the runtime rejects a dispersion workspace without explicit weights. Tuple energy accumulators must be cleared before use. If the engine already accumulates into a compatible fixed point destination, preserve contributions from other force producers when starting cuPotential evaluation.

The total energy comprises direct space including configured exclusion corrections, reciprocal space, physical replacement pairs, self energy, the net charge background term, and the selected LJ tail correction if applicable. Add each exactly once. The mesh output does not include all these corrections. The OpenMM adapter adds self, background, and tail corrections in [`finishPendingEnergy`](../plugins/openmm/cpp/platforms/cuda/src/nonbonded/CudaNonbondedKernels.cu#L1285). Compute the net-charge correction again after each change of the box volume ([C API reference](https://docs.nvidia.com/cuda/cupotential/c-api/reference#splitting-plans)). That page also warns that `cupo_splitting_plan_get_spread_order` returns grid points per axis, not the approximation order of the spline.

After this sequential path is correct, direct and reciprocal evaluation can run on separate streams after cell list readiness. The OpenMM adapter lets both add into one fixed point force destination. A [plugin comment](../plugins/openmm/cpp/platforms/cuda/src/nonbonded/CudaNonbondedKernels.cu#L1597) describes this as fixed64 atomic accumulation, but the public headers state only that forces accumulate. Use separate destinations, or compare single-stream and concurrent forces, before you rely on a shared one. Keep energy outputs separate and join all producers before force conversion or integration. Delay host energy readback until it is needed, since an early readback can prevent overlap with other force calculations.

## Restrictions that affect feasibility

### Virial and pressure coupling

The public direct space, reciprocal space, and tuple execution APIs expose force and energy outputs without a virial or stress output. This is the largest missing interface for an engine whose pressure calculation or barostat requires the configurational virial tensor.

An energy based Monte Carlo volume move can use the available total energies. The existing OpenMM barostat follows that approach in [`MonteCarloBarostatImpl.cpp`](../plugins/openmm/cpp/openmmapi/src/MonteCarloBarostatImpl.cpp#L121). That example does not establish support for an engine's virial based pressure coupling.

[`samples/ltoir-usage/usage.cu`](../samples/ltoir-usage/usage.cu) implements analytical real, reciprocal, and exclusion virials using scalar Fourier derivatives. However, its spreading, gathering, and virial reductions run on the host, including in compact convolution mode; see [`docs/ltoir-usage.md`](../docs/ltoir-usage.md#L52). It is a mathematical reference for a new implementation. It is not a ready GPU virial backend for the high level force API.

A production solution requires a runtime extension supplying virials, or an engine implementation of the missing terms using lower level primitives. Periodic reciprocal stress includes derivatives with respect to the box; summing wrapped coordinates times returned forces is not a general replacement. Background and tail terms also need consistent volume derivatives. Validate the completed tensor against finite strain energy derivatives before using it for pressure coupling.

### Distributed simulations

The public interfaces have no MPI communicator, owned versus ghost particle distinction, halo exchange contract, or distributed FFT descriptor. A cell list is created with a fixed atom count and the calculation runs on the selected CUDA device.

There is consequently no supported direct mapping from each domain in an existing domain decomposition to an independent high level workspace. A proposed mapping would first have to establish pair ownership, periodic treatment, exclusion ownership, reciprocal assembly, and force reduction. Merely including ghost particles risks duplicate energy and force contributions.

Independent simulations on separate GPUs fit the per device resource model. Scaling one simulation across GPUs requires additional engine architecture or a distributed library extension. For an engine that already distributes pair and mesh work, the scalar layer is the more plausible integration point.

### Particle ordering and topology changes

Internal cell list sorting is transparent to the caller, and output forces return in the input atom order. Application changes to that input order require more work:

- Standard direct space workspaces copy exclusions during creation. A permutation that changes the indexed exclusion pair set requires workspace recreation.
- Custom direct space workspaces borrow their exclusion array, allowing indices to be remapped in place with stream ordering. Explicit reciprocal weights and particle parameters must follow the same permutation.
- Supported public tuple operations update parameters while retaining topology. The mapping helper in [`detail/tuples_adapter.h`](../include/cupo/detail/tuples_adapter.h#L13) is explicitly reserved for the OpenMM plugin and is outside the supported API.

These rules are documented in [`dspace.h`](../include/cupo/dspace.h#L59). A first adapter should preserve a stable logical particle order in its cuPotential buffers, using device gather and scatter kernels when the engine changes its own ordering. That adds bandwidth cost but avoids tying correctness to unsupported mapping hooks. Particle insertion, deletion, or topology changes can require rebuilding handles and dependent graphs.

### Force field and numerical compatibility

Native LJ combining supports Lorentz-Berthelot and geometric rules. Reciprocal dispersion uses factorized geometric C6 weights, `2*sqrt(epsilon)*sigma^3`, independently of the native direct combining rule. Arbitrary type pair tables or alternative dispersion conventions need a deliberate expression or correction strategy. The presence of a custom radial evaluator does not make arbitrary long range models available automatically.

Exclusions remove selected interactions; scaled 1-4 and other special pairs must be restored as physical replacement interactions through the engine or the tuple API. Check the engine's periodic treatment of special pairs, cutoff shifts, switches, and tail policies. With native dispersion splitting, switching is disabled and the direct LJ interaction uses the documented shift.

Direct and reciprocal components must use the same splitting and particle weights. A Prolate short range calculation cannot be combined with an unchanged Ewald reciprocal calculation. Self, background, and reciprocal exclusion corrections must also match the chosen splitting.

The public Ewald factory fixes alpha from the cutoff and a `5e-4` design parameter. It does not expose an arbitrary alpha setter. The OpenMM plugin applies the same limit to its source forces: it accepts only an Ewald error tolerance of `5e-4`, rejects pinned meshes, and accepts a pinned alpha only if it agrees with the cutoff-derived value ([NonbondedForce guide](https://docs.nvidia.com/cuda/cupotential/nonbonded-force#standard-source-restrictions)). Prolate uses fixed model parameters, and the Prolate grid sizing policies are calibrated to a `5e-4` combined force error for cutoffs from 0.6 to 0.9 nm. This calibration does not apply to Ewald sizing. It is not an independent Coulomb, dispersion, energy, or virial accuracy guarantee; see [`splitting_plan.h`](../include/cupo/splitting_plan.h#L29) and [`docs/ltoir.md`](../docs/ltoir.md#L358). An engine with tighter precision requirements must validate an explicit numerical configuration or use its own implementation around the scalar functions.

The custom callback returns `dU/dr`, while scalar real space math gradients return derivatives with respect to `r^2`. Their force formulas differ by the corresponding distance factors. The scalar Fourier functions use cycles per length for frequency. These conventions need explicit checks when incorporating the math into existing kernels.

Parameter updates can change particle records and weights without changing their addresses. Compiled expression preparation runs outside graph capture. No public high level output supplies an alchemical energy derivative with respect to lambda; workflows requiring that quantity need an additional implementation.

### Box updates and resource lifetime

The cell list reserves a box envelope at creation and rejects updates outside its floor or capacity. The floor is reached when "an axis has fewer than 2R+1 cells for stencil radius R" ([`cell_list.h`](../include/cupo/cell_list.h#L31)). The [NonbondedForce guide](https://docs.nvidia.com/cuda/cupotential/nonbonded-force#supported-settings-and-differences) states the plugin rule as at least three cutoff lengths for each orthorhombic box dimension, "including after NPT contraction". Check small boxes and strong NPT compression against this floor. Reciprocal mesh dimensions remain fixed for a workspace. Reserve expansion headroom and define a rebuild policy for larger changes. Refresh reciprocal influence after every accepted box change; convolution does not detect stale influence tables.

`cupo_cell_list_update_box` also reports whether an accepted box keeps the current cell counts. The header states that captured graphs remain valid for `CUPO_CELL_LIST_BOX_IN_BAND`, but makes no corresponding statement for `CUPO_CELL_LIST_BOX_BAND_CROSS`; see [`cell_list.h`](../include/cupo/cell_list.h#L32). The OpenMM adapter continues graph replay through a band crossing, as explained in [`updateBoxState`](../plugins/openmm/cpp/platforms/cuda/src/nonbonded/CudaNonbondedKernels.cu#L1409). That relies on runtime behavior outside the public contract. Until the behavior is documented, a new adapter should recapture dependent graphs after a band crossing, or test replay across one explicitly.

Serialize use of each direct and reciprocal workspace. Keep arenas, cell lists, explicit weights, and custom global buffers alive through queued work and graph replay. Destroy dependent graphs and drain work before teardown, then release workspaces before their borrowed cell list and storage.

## Build and deployment

Use a release runtime archive and matching headers. The current release specifies Linux x86-64 and CUDA 13.4; its checkout records `13.4.3` in [`.cuda-version`](../.cuda-version). NVIDIA's [installation requirements](https://docs.nvidia.com/cuda/cupotential/install) list supported architectures SM75, SM80, SM86, SM89, SM90, SM100, SM103, and SM120. CUDA 13.4 needs driver R615 or newer. Data center GPUs, and the professional RTX GPUs that CUDA forward compatibility supports, can also use a driver of the R535 to R610 branches with the `cuda-compat-13-4` package. GeForce GPUs need R615 or newer. Check the runtime archive's metadata for the architectures actually included.

The runtime provides a `Cupo::Runtime` CMake target. The adapter can link this target and the CUDA dependencies it uses without building the OpenMM plugin. Scalar device calls additionally require linking the matching `cupo_math.cu.o` with CUDA LTO, as described in [`docs/ltoir.md`](../docs/ltoir.md#L558). Scalar only consumers can omit `libcupo.so`; grid planning and runtime services require it.

Compact convolution has build, device, and mesh restrictions. Mesh axes are limited to supported sizes through 512, and support should be queried before selecting it. Compact grid sizing supports only Coulomb Prolate P1 plans; see [`kspace.h`](../include/cupo/kspace.h#L58) and [`docs/ltoir.md`](../docs/ltoir.md#L335). The OpenMM adapter keeps dispersion on dense cuFFT over the shared mesh, as noted in [`setupDevice`](../plugins/openmm/cpp/platforms/cuda/src/nonbonded/CudaNonbondedKernels.cu#L843). The high level reciprocal API can use cuFFT, and its automatic policy can fall back to cuFFT for unsupported compact cases. Backend selection and Fourier truncation are independent choices; changing retained modes changes the numerical approximation. Sharp radial truncation can introduce energy discontinuities when box changes move modes across the boundary.

Pin the runtime release and ABI in the backend build. The C API is experimental, and the runtime ABI is currently 32. A fork of the adapter should isolate this dependency behind a small engine interface so future ABI changes do not spread through the simulation code.

## Suggested implementation and validation sequence

1. Implement a force provider for one GPU with a fixed orthorhombic box and stable particle order. Retain the engine's existing dynamics and bonded terms. Start with cuFFT and a full reciprocal spectrum.
2. Validate single point forces and decomposed energies against the engine's established implementation or a converged independent reference. Include a charged system, exclusions beyond the direct cutoff, scaled special pairs, LJ mixing, switching, and energy only and force only calls.
3. Test periodic image changes, explicit atom permutations, parameter updates, and reduced triclinic boxes. Compare eager execution with graph replay, including replay across a cell list band crossing. Check repeated initialization and teardown.
4. Measure force error, energy consistency, and NVE drift before adding NVT trajectories. Establish a separate virial validation path before enabling virial based pressure coupling. For Monte Carlo volume moves, test both accepted and rejected proposals with box dependent corrections refreshed correctly.
5. Profile the complete timestep, including coordinate packing, force conversion, duplicate neighbor work, and synchronization. Then evaluate direct and reciprocal overlap, graph capture, and compact convolution at matched accuracy.

The repository already contains useful cases in [`test_far_exceptions.py`](../plugins/openmm/tests/test_far_exceptions.py), [`test_reorder.py`](../plugins/openmm/tests/test_reorder.py), [`test_triclinic.py`](../plugins/openmm/tests/test_triclinic.py), [`test_ljpme.py`](../plugins/openmm/tests/test_ljpme.py), and [`test_npt.py`](../plugins/openmm/tests/test_npt.py). Their scenarios can guide tests for another engine, but their OpenMM fixtures do not validate that new adapter.

## Verification of this assessment

All public host API headers were checked together with `gcc -std=c11 -Werror -fsyntax-only` and `g++ -std=c++17 -Werror -fsyntax-only`, using this checkout's headers and the installed CUDA 13.3 headers. Both checks passed without OpenMM headers. This verifies host header usability, not compatibility with the CUDA 13.4 release binary.

The checkout has no downloaded `libcupo.so` or scalar objects, so no standalone runtime execution or performance measurement was performed. The default local compiler is CUDA 12.9 and an additional CUDA 13.3 installation is available; neither is the matching CUDA 13.4 toolkit specified by this release. The local NVIDIA driver is 620.70.06 and the GPUs are two RTX 4070 cards and an RTX 3090. A runtime prototype needs the matching toolkit and release binaries before numerical or throughput conclusions can be drawn.

No speedup relative to another MD engine has been established.

The NVIDIA documentation site was read on 2026-11-18. It describes release 0.3.5 and agrees with the header contracts that this note cites; see [0010](0010_nvidia_documentation_review.md).
