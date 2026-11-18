# NVIDIA cuPotential documentation review

Date: 2026-11-18

Source: the [NVIDIA cuPotential documentation](https://docs.nvidia.com/cuda/cupotential), 12 pages, read on 2026-11-18 through their Markdown versions. Append `.md` to a page URL to get the Markdown. The site lists every page in [`llms.txt`](https://docs.nvidia.com/cuda/cupotential/llms.txt). The four figures were downloaded and read as images.

Source version: 0.3.5, runtime ABI 32

Reviewed upstream commit: `fbfc4cb6282cd52d33f7adbbcbaeaa5c19b47d74`

Scope: what the documentation site adds to notes 0001 to 0006, and where it differs from this checkout. Statements are labeled as documented (a page states it), calculated (arithmetic in this note), or inferred (a reasoned conclusion that still needs a measurement). No benchmark, test, or runtime call was run for this note.

## Summary

- The site documents release 0.3.5, the release of this checkout. Its installation and sample pages clone `v0.3.5`, which is still the latest upstream tag. No page contradicts a C API header that notes 0001, 0004, or 0005 cite.
- The overview page shows a different speedup chart from the checkout's [`cupotential-openmm.png`](../plugins/openmm/docs/cupotential-openmm.png). Its per-GPU maxima are 2.66x to 3.93x, against 2.64x to 3.99x in the checkout. Single values differ by up to about 0.2 in both directions. The page gives no date, OpenMM version, or precision for its chart. The patterns that [0003](0003_speedup_theory_and_mechanisms.md#4-how-the-parts-combine) uses hold on both charts (section 5).
- The error analysis page states the purpose of its NVE test: "here we focus on the contribution from any residual jump at the cutoff of the short-range splitting term". The test uses rigid water, switched Lennard-Jones, and a `1e-8` constraint tolerance to reduce the other sources. It therefore measures the electrostatics method. It does not measure bonded terms or coordinate precision.
- Inferred from the convergence figure (section 6.4): at the mesh spacing of that NVE run, the C2 Prolate Coulomb force error is about 2.5e-4. The standard PME arm targets 5e-4. The force errors differ by a factor of about 2, but the drifts differ by about 478x. In this test, the static force error does not predict the drift. The smoothness of the short-range term at the cutoff is the more probable cause. The page names "ensemble error" as the target of future parameter work.
- The NonbondedForce guide documents plugin restrictions that the notes did not list (section 3). Examples: each orthorhombic box length must be at least three cutoffs, also after NPT contraction. The source Ewald tolerance must be `5e-4`. Pinned meshes are rejected. Conversion does not keep `setIncludeDirectSpace(False)`.
- One statement in the notes was wrong. [0004](0004_interaction_scope_and_stream_scheduling.md#interaction-coverage) said that the plugin uses periodic boundaries for nonbonded exception pairs. The plugin passes the source force's flag, which defaults to nonperiodic distances since 0.3.4. Five other statements in 0002 and 0003 were out of date or imprecise (section 8).
- The C API pages add few facts beyond the headers (section 4). Two of them are traps: the spreading-order query returns grid points per axis, not the approximation order, and the net-charge correction must be computed again after each volume change.

## 1. Pages and their use in the notes

| Page | Content | Notes that use it |
| --- | --- | --- |
| [Overview](https://docs.nvidia.com/cuda/cupotential) | Components, OpenMM replacements, speedup chart, benchmark procedure | 0002, 0003, 0004 |
| [Installation](https://docs.nvidia.com/cuda/cupotential/install) | Requirements, PyPI wheel, conda channel, source build, Docker GPU selection | 0001, 0002 |
| [Samples](https://docs.nvidia.com/cuda/cupotential/samples) | `solvated_systems` and `alchemical` samples | this note only |
| [Tests](https://docs.nvidia.com/cuda/cupotential/tests) | `./run_test openmm_unit`, tests of an installed plugin | this note only |
| [Licenses](https://docs.nvidia.com/cuda/cupotential/licenses) | A link to the repository `LICENSE` file only | none (0001 cites the `LICENSE` file directly) |
| [OpenMM NonbondedForce](https://docs.nvidia.com/cuda/cupotential/nonbonded-force) | Plugin architecture, interaction definitions, conversion, custom expressions, supported settings | 0001, 0003, 0004, 0005, 0006 |
| [C API getting started](https://docs.nvidia.com/cuda/cupotential/c-api/getting-started) | Runtime archive, a complete harmonic-bond example | 0001 |
| [C API reference](https://docs.nvidia.com/cuda/cupotential/c-api/reference) | Data conventions, function groups, lifetimes | 0001, 0003, 0004, 0006 |
| [Splitting](https://docs.nvidia.com/cuda/cupotential/theory/splitting) | Ewald and C2 Prolate splitting, self energy, zero mode | 0003 |
| [Spreading](https://docs.nvidia.com/cuda/cupotential/theory/spreading) | Assignment window, C2O3, deconvolution, aliasing | 0003, 0005 |
| [Error analysis](https://docs.nvidia.com/cuda/cupotential/theory/error-analysis) | Intrinsic and extrinsic error, mesh convergence, NVE drift | 0002, 0003, 0005, 0006 |
| [Changelog](https://docs.nvidia.com/cuda/cupotential/changelog/openmm) | Plugin releases 0.3.1 to 0.3.5 | 0002, 0004, 0005 |

The site also offers an MCP server for AI clients at `https://docs.nvidia.com/cuda/cupotential/_mcp/server`. This review did not use it.

## 2. Release, platform, and procedures

Documented on the [installation page](https://docs.nvidia.com/cuda/cupotential/install#requirements):

- GPUs of SM 75, 80, 86, 89, 90, 100, 103, or 120, on Linux x86-64.
- NVIDIA driver R615 or newer, for CUDA 13.4. Data center GPUs, and the professional RTX GPUs that CUDA forward compatibility supports, can also use a driver of the R535, R570, R580, R590, R595, or R610 branch through the `cuda-compat-13-4` package. GeForce GPUs need R615 or newer.
- Python 3.12 and OpenMM 8.4 for the wheel and the conda package. The wheel requires `openmm[cuda13]>=8.4,<8.5`. PyPI has the plugin as [`nv-openmm-plugin`](https://pypi.org/project/nv-openmm-plugin/).
- The Docker environment of the repository is the recommended setup. `NV_GPU` selects the GPUs. `CUPO_DOCKER_ADD_DEVICE_NODES=1` keeps the GPUs in a container after a `systemctl daemon-reload` on the host.

The [C API getting started page](https://docs.nvidia.com/cuda/cupotential/c-api/getting-started) states that the standalone C API needs neither OpenMM nor Python. The build needs the CUDA toolkit version in the archive's `.cuda-version` file.

Documented test and benchmark procedures:

| Task | Command | Notes |
| --- | --- | --- |
| Short energy check | `./run_test openmm_unit -k "test_energy and dhfr-fp64"` | A few seconds after the build. The full `openmm_unit` suite takes a few hours on one GPU ([tests page](https://docs.nvidia.com/cuda/cupotential/tests)). "In double precision" refers to the stock OpenMM reference: the `fp64` cases compare the plugin with stock OpenMM in double ([`test_plugin_vs_stock.py`](../plugins/openmm/tests/test_plugin_vs_stock.py#L313)). The plugin itself runs only in single or mixed precision. |
| Test an installed plugin | `CUPO_TEST_INSTALLED=1 ./run_test openmm_unit -k test_unit_conversion` | Tests that need the source build are skipped. |
| One benchmark | `./run_test openmm_bench -k "apoa1-mixed-nvt"` | Under a minute after the build. Without `-k`, every system runs in NVT and NPT at mixed and single precision, about one hour on one GPU ([overview](https://docs.nvidia.com/cuda/cupotential#run-the-benchmarks-on-your-gpu)). |

The benchmarks lock the GPU clocks. This needs passwordless `sudo nvidia-smi` in the container and the `SYS_ADMIN` capability, which `./run_docker` gives. `CUPO_ALLOW_UNLOCKED_CLOCKS=1` runs with the current clocks and marks the results "clocks not locked". The results go to `plugins/openmm/build/bench-results/`. Release packages do not include reference measurements, so a completed run is reported as "measured, not compared". Tests and benchmarks fail at once if another process uses the selected GPU.

## 3. The plugin model and its restrictions

The [NonbondedForce guide](https://docs.nvidia.com/cuda/cupotential/nonbonded-force) describes one force that combines a `CutoffInteraction` with up to one Coulomb (P1) and one dispersion (P6) `LatticeSumInteraction`. Its architecture diagram assigns these parts to `libcupo`: the DSpace pair traversal, expression compilation, splitting plans, the KSpace mesh calculation, and the Standard exception replacements through the tuples API. The plugin computes the tail coefficients on the CPU. It calls the self-energy and net-charge functions of the splitting plan on the CPU.

Documented restrictions and behaviors that notes 0001 to 0006 did not list:

| Topic | Documented statement |
| --- | --- |
| Box size | "Each orthorhombic dimension must be at least three cutoff lengths, including after NPT contraction." The header rejects a box update when "an axis has fewer than 2R+1 cells for stencil radius R" ([`cell_list.h`](../include/cupo/cell_list.h#L31)). Inferred: the two rules agree for `R` = 1 and cells of at least one cutoff. |
| Accuracy target | "The source Ewald error tolerance must currently be `5e-4`." The plugin source rejects other values ([`NonbondedForce.cpp`](../plugins/openmm/cpp/openmmapi/src/NonbondedForce.cpp#L232)). |
| Explicit PME parameters | "Pinned meshes are rejected. A pinned alpha must agree with the cutoff-derived value." |
| Direct-space suppression | "Conversion does not preserve `source.setIncludeDirectSpace(False)`. A reciprocal-only source regains direct interactions, which can double-count contributions supplied by another force." |
| Accepted source methods | `PME` and `LJPME` for Standard mode. Standalone Custom mode accepts `CutoffPeriodic`. |
| Unsupported custom features | `NoCutoff`, `CutoffNonPeriodic`, interaction groups, tabulated functions, computed values, and energy parameter derivatives. |
| Custom arithmetic | "Custom parameters and arithmetic use FP32 even in a mixed-precision Context. CUDA fast math includes approximate division and square root and flushes subnormal values to zero." |
| Expression compilation | The runtime differentiates the Lepton expression with respect to `r`, compiles energy and derivative with NVRTC, and links the LTO IR into the direct-space traversal with nvJitLink. Parameter updates and atom reorderings do not recompile. |
| Pair selection | `select(mask, softcore, ordinary)` evaluates only the active branch. Multiplication by a zero mask does not make sure that a singular branch is skipped. |
| Exclusions | "Exclusions remove pair contributions by repeated evaluation; the evaluator may still run on an excluded pair." |
| Exception images | Physical exception replacements follow the source's periodic flag, but "reciprocal exclusion corrections always use periodic wrapping. This can differ from stock OpenMM when an exception spans periodic images." The plugin passes the source flag to the tuples API ([`CudaNonbondedKernels.cu`](../plugins/openmm/cpp/platforms/cuda/src/nonbonded/CudaNonbondedKernels.cu#L1027)). The 0.3.5 changelog states that reciprocal exclusion corrections for distant exception pairs in triclinic boxes "now use the same periodic image as the exception interaction". For an exception with the flag `False`, the two pages can be read differently. |
| Lattice terms | Lattice terms are not switched. A dispersion lattice sum disables automatic switching and the homogeneous tail correction. |
| Global parameters | Live globals take effect at the next evaluation without recompilation. "Energy derivatives with respect to global parameters are not provided." |

The overview page states that the plugin accelerates "nonperiodic harmonic bonds, harmonic angles, and periodic torsions" through the tuples backend. This matches the plugin policy in [0004](0004_interaction_scope_and_stream_scheduling.md#interaction-coverage), which sends periodic bonded forces to stock OpenMM.

## 4. C API details

The [C API reference](https://docs.nvidia.com/cuda/cupotential/c-api/reference) agrees with the data contract in [0001](0001_non_openmm_integration_assessment.md#data-and-execution-contract): FP32 nonbonded positions with a stride of at least three floats, FP32 `{charge, sigma, epsilon}` records, the box packed as `[ax, by, cz, bx, cx, cy]`, forces as signed fixed point with 32 fractional bits in three planes, and FP64 energies. Direct and reciprocal calls overwrite their energy scalars, and tuple execution adds to its scalar.

Facts that the page states and that the notes did not have:

- For a versioned descriptor, zero-initialize the structure, then set `struct_bytes` and `abi_version`.
- `cupo_last_error()` is thread-local. A failed call replaces it, and a successful call clears it. Copy the message before the next API call. C++ exceptions do not cross the C API boundary.
- `cupo_splitting_plan_get_spread_order` returns grid points per axis, not the approximation order of the spline. For C2O3 the value is 4, while the approximation order is 3.
- Compute the net-charge correction again when the box volume changes.
- Refresh the influence function before the first energy or force evaluation and after every box update.
- Without `cupo_kspace_configure_convolution`, a reciprocal workspace uses cuFFT with the full mesh spectrum. The OpenMM plugin requests the automatic backend with radial truncation, so an engine that keeps the default does not run the benchmarked configuration.
- A cell list allocates internally when its arena pointer is null. Direct-space workspaces always need a caller-owned arena.
- The staged reciprocal calls `cupo_kspace_spread`, `cupo_kspace_convolve`, and `cupo_kspace_interpolate` evaluate the mesh calculation one stage at a time. They take no position argument, because spreading reads the bound cell list ([`kspace.h`](../include/cupo/kspace.h#L249)).
- An expression can declare lattice-direct terms. The runtime then subtracts the selected long-range contribution, and the application supplies the reciprocal calculation, the self and background corrections, and the exception interactions.
- The scalar functions are designed "for integration of cuPotential's splitting, spreading, and interpolation math into kernels developers have already built".
- `cupo/convolution.h` is designed for compact Fourier kernels. The page cites Simmonett and Brooks [2021] for the use of a fine real-space grid with fewer reciprocal modes.
- The getting started example evaluates one harmonic bond with `CUPO_TUPLES_FLOAT64`, four doubles per atom, and prints `Energy: 5.000000 kJ/mol` and forces of ±100 kJ/(mol nm). It is a short first test of a runtime archive.

## 5. The speedup chart on the overview page

The overview page shows a chart of six systems on seven GPUs in NPT, with the printed maxima in an "NPT Max" column. The checkout chart is dated 12 and 13 November 2026 and labeled OpenMM 8.4.0, NPT, mixed precision. The site chart has no such labels.

The values below are read from the marker positions. The axis was calibrated from the gridlines at 1.0x and 4.0x. Markers that overlap another marker carry an asterisk and an uncertainty of about ±0.05. The other values are within about ±0.03. The checkout values are from [0003](0003_speedup_theory_and_mechanisms.md#1-the-reported-speedups).

| GPU | DHFR | TYK2 | BACE | ApoA1 | Water | Cellulose | Max, site | Max, checkout |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RTX PRO 6000 Blackwell Server | 3.30 | 3.89* | 3.93* | 3.45* | 3.22 | 3.39* | 3.93x | 3.95x |
| L40S | 3.62 | 3.92* | 3.92* | 3.38* | 3.33* | 3.05 | 3.92x | 3.99x |
| A10 | 2.54* | 2.50* | 2.6–2.7* | 2.6–2.7* | 2.6–2.7* | 3.35 | 3.35x | 3.35x |
| T4 | 1.90 | 2.06* | 2.17 | 2.09* | 2.04* | 2.66 | 2.66x | 2.64x |
| B200 | 3.19* | 3.16* | 2.98 | 2.60 | 2.33* | 2.40* | 3.19x | 3.24x |
| H100 PCIe | 3.42 | 3.14* | 3.14* | 2.73* | 2.63 | 2.73* | 3.42x | 3.34x |
| A100 PCIe | 3.02* | 3.04* | 2.92 | 2.65* | 2.51 | 2.72* | 3.04x | 3.08x |

Calculated from the two tables: the largest differences are water on the T4 (2.04 against 2.22), DHFR on the RTX PRO 6000 (3.30 against 3.44), and TYK2 on the B200 (3.16 against 3.04). Inferred: the two charts come from separate benchmark sessions, and differences of about 0.2 are within the session-to-session spread. Do not attribute a difference smaller than that to a mechanism.

The patterns that 0003 section 4 uses hold on both charts:

- The two largest maxima are on the GPUs with slow FP64, the RTX PRO 6000 and the L40S.
- On the B200, H100, and A100, DHFR has the largest or nearly the largest speedup, and water or cellulose the smallest.
- On the T4 and the A10, cellulose has the largest speedup.
- Water is among the two lowest values on six of the seven GPUs on the site chart, and on five of seven on the checkout chart.

## 6. Theory pages and the error analysis

### 6.1 Splitting and spreading

The [splitting page](https://docs.nvidia.com/cuda/cupotential/theory/splitting) credits the prolate approach to Liang et al. [2026] and defines the C2 Prolate kernel as in [0003](0003_speedup_theory_and_mechanisms.md#22-prolate-splitting). It also states that the Coulomb splitting plan gives both the self and the net-charge corrections, and that "direct, reciprocal, self, background, and any exclusion corrections must use the same split."

The [spreading page](https://docs.nvidia.com/cuda/cupotential/theory/spreading) states that the C2O3 weights are C² in the particle positions, which gives a C² mesh energy and C¹ analytical forces for a fixed box and mesh. The deconvolution uses discrete stencil power. "The Ewald B-spline path uses squared stencil DFT amplitudes with a treatment for near-zero values; the C2-order-3 (C2O3) path uses stencil power averaged over fractional particle positions within a grid cell."

### 6.2 Intrinsic and extrinsic error

The [error analysis page](https://docs.nvidia.com/cuda/cupotential/theory/error-analysis) follows the definition of Predescu et al. [2020]. Intrinsic error stays when the direct and reciprocal sums of the chosen decomposition are evaluated exactly. Extrinsic error comes from the mesh. The page states that C2 Prolate has no intrinsic error, because `u_lr = 1/r` beyond the cutoff and `u_sr` is its exact complement inside. Its Fourier kernel has a nonzero tail, so a finite mesh still has extrinsic error.

### 6.3 Mesh convergence study

| Setting | Documented value |
| --- | --- |
| System | `water_100k`, 100,002 atoms of TIP3P-FB water, the first 25 saved NPT frames |
| Numerical path | "Host C++ real space in FP32, FP32 mesh/FFT storage, and FP64 reference stencil evaluation" |
| Window | C2O3 with discrete deconvolution for every curve, also for Ewald. The Ewald curves are therefore not stock OpenMM PME. |
| Meshes | `G` = 32, 48, 64, 80, 96, 128, 160, 192, 224 points per axis |
| Coulomb splittings | Ewald with `alpha` = 4.590 nm^-1 (0.6 nm) and 2.989 nm^-1 (0.9 nm), u-series 2 with `b` ≈ 1.63, C2 Prolate with `c` = 10.75. Not retuned during the sweep. |
| Dispersion splittings | Ewald with `alpha` = 4.484 and 1.968 nm^-1, C2 Prolate with `c` ≈ 12.764 |
| Reference | OpenMM 8.4.0, CUDA double precision, 192³ mesh, 2.0 nm cutoff, `alpha` = 4.0 nm^-1. A 256³ check on the first frame changed the Coulomb force by 5.40e-6 (relative). |
| Metrics | Relative energy error and relative L2 force error per channel. Geometric mean over the frames, with bars of one standard deviation in log space. |

The page lists kernel evaluation, roundoff, and reference error as possible causes of a plateau "once the other contributions are small", and states that they "require separate checks". It does not separate them.

Read from the Coulomb figure for the 0.9 nm cutoff (calculated from pixel positions, about ±10%): the C2 Prolate force error is 2.2e-3 at `G` = 32, 2.8e-4 at 64, 1.7e-4 at 80, 1.1e-4 at 96, and 1.7e-5 at 224. The fixed Ewald curve levels off at 5.0e-4, and u-series 2 at 1.4e-4.

### 6.4 NVE drift study

The test setup is in [0003](0003_speedup_theory_and_mechanisms.md#22-prolate-splitting). The page adds these points:

- Purpose: "Many factors can affect energy drift; here we focus on the contribution from any residual jump at the cutoff of the short-range splitting term. Tight constraint tolerances and Lennard-Jones switching help reduce other sources of drift."
- Integration: "synchronized velocity-Verlet with RATTLE constraints, a 2 fs timestep, and constraint tolerance `1e-8`". No thermostat, barostat, or center-of-mass motion removal.
- Normalization: the energy is divided by `N_DOF k_B T` with `N_DOF` = 60,000 and `T` = 300 K. The fit uses 49,901 samples from 20 ps to 10 ns.
- Control: "All three trajectories start from the same positions and velocities". They share the integrator, the mixed precision, and the Lennard-Jones treatment (a 0.9 nm cutoff, a switch from 0.8 nm, no tail correction). Only the electrostatics differ. The converged reference (2.0 nm, 128³) drifted `-4.536e-7 ns^-1`.
- Interpretation: the difference "cannot be assigned to the splitting alone: the methods also use different meshes and reciprocal-space evaluation rules."
- Future work: "re-optimize the numerical parameters to minimize **ensemble error**, rather than focusing primarily on force errors at fixed configurations."

Inferred link between the two studies. The NVE run uses a 45³ mesh in a 6.6987 nm box, a spacing of 0.1489 nm (calculated). The `water_100k` box in [`data/water_100k`](../data/water_100k/) is 10.036 nm, so the same spacing is `G` ≈ 67 there (calculated, with the initial box). Interpolation in the figure gives a C2 Prolate Coulomb force error of about 2.5e-4 at that spacing. The standard PME arm uses a `5e-4` tolerance. The fixed Ewald curve, with `alpha` = 2.989 nm^-1 near the 2.920 nm^-1 of that arm, levels off at 5.0e-4. Inferred: that plateau is the error of the truncated `erfc` term, because the mesh error keeps falling. The static force errors of the two NVE arms therefore differ by a factor of about 2, while their drifts differ by about 478x. Two conclusions follow, both inferred:

1. In this test, the drift is not set by the size of the force error at fixed positions. For a fixed box and mesh, the mesh forces are the exact gradient of the mesh energy ([spreading page](https://docs.nvidia.com/cuda/cupotential/theory/spreading#the-complete-mesh-calculation)). Such an error changes the energy that the dynamics conserves, but it does not make that energy drift. The jump of the truncated `erfc` term at the cutoff does. This agrees with the purpose that NVIDIA states.
2. The test does not show how much of the remaining cuPotential drift comes from FP32 coordinates, FP32 pair arithmetic, or the mesh. The reference arm, with stock OpenMM forces from FP32 positions, drifted only `-4.5e-7 ns^-1`. In this test, the whole drift of the cuPotential arm, `1.4e-6 ns^-1`, therefore limits what a change of the coordinate representation could remove. Signed contributions can cancel, so the limit is approximate. A test of coordinate precision needs bonded terms, larger boxes, coordinates far from the origin, or longer runs. [0006](0006_mint32_feasibility_consensus.md#8-open-questions) applies this to MINT32.

## 7. Changelog items that affect the notes

| Release | Item | Effect |
| --- | --- | --- |
| 0.3.1 | Fixed the pressure unit conversion of `MonteCarloBarostat`. Before, `1.0*unit.bar` reached the barostat as `1e-25`. | Inferred: NPT runs with the plugin barostat before 0.3.1 sampled a pressure near zero. |
| 0.3.1 | `LangevinMiddleIntegrator` supports virtual sites, for example TIP4P. | |
| 0.3.3 | Switched to C2 Prolate splitting with refitted C2O3 spreading and calibrated grid sizes. Retired the C0 prolate implementations. | Prolate results before 0.3.3 use a different method. |
| 0.3.4 | Triclinic boxes, LJPME with Ewald or Prolate per term, live globals and parameter offsets, XML serialization, checkpoints with the RNG state, and the TYK2 alchemical sample. Exceptions default to nonperiodic distances. | [0002](0002_nvidia_blog_integration_implications.md#reconcile-the-tutorial-with-the-reviewed-checkout): the blog tutorial calls LJPME, triclinic boxes, and offsets unsupported, so it describes a release before 0.3.4 (inferred). |
| 0.3.5 | New SETTLE, SHAKE, and Langevin kernels. Native double or float-float selected per GPU, with the `CUPO_PLUGIN_MIXED_COMPUTE` override. Other Python threads can run during `step()`. Fixed reciprocal exclusion corrections for distant exception pairs in triclinic boxes. CUDA 13.4. | [0003](0003_speedup_theory_and_mechanisms.md#31-one-fused-integrator-kernel) and [0005](0005_numerical_representations_layouts_and_determinism.md#precision-and-storage-in-the-openmm-plugin) describe the integrator code of this release. |

## 8. Changes made to notes 0001 to 0006

Each changed note has an "Updated" line under its date.

- [0001](0001_non_openmm_integration_assessment.md): driver and forward compatibility details, the API conventions of section 4, the default convolution configuration, internal cell-list allocation, the purpose of the scalar functions, the plugin's tolerance rule, the three-cutoff box floor, and the net-charge update after a volume change.
- [0002](0002_nvidia_blog_integration_implications.md): the documented reference of the convergence study, the second speedup chart, and the changelog evidence that the blog tutorial describes 0.3.3 or earlier.
- [0003](0003_speedup_theory_and_mechanisms.md): the second chart, the stated purpose and the reference arm of the NVE test, the inference of section 6.4, the conditions of the convergence study (with its Ewald `alpha` values, which differ from the plan rule), Simmonett and Brooks [2021], and the documented benchmark procedure. Corrected: the harness defaults (NVT is the command-line default, and the pytest stage runs both ensembles), and the availability of the per-system benchmark data (release packages do not include it).
- [0004](0004_interaction_scope_and_stream_scheduling.md): corrected the statement that the plugin uses periodic boundaries for nonbonded exception pairs. The plugin passes the source flag, which defaults to nonperiodic distances since 0.3.4. Added the documented bonded scope of the plugin, the unsupported custom features, `select`, lattice-direct terms, and the Python thread change of 0.3.5.
- [0005](0005_numerical_representations_layouts_and_determinism.md): the documented FP32 custom arithmetic and fast math, and the deconvolution rule of each window.
- [0006](0006_mint32_feasibility_consensus.md): the C API statement on coordinates, the staged reciprocal calls, the purpose of the scalar functions (Route C), the cell-list floor (section 6.3), the purpose and the reference arm of the NVE test with the limit that they set (open question 1), and NVIDIA's NVE protocol as the control arm (section 9).

## 9. Not checked

- The exception-image rule for a pair with the periodic flag `False` (section 3). The guide and the 0.3.5 changelog can be read differently.
- The conditions of the site chart: date, OpenMM version, precision, and the plugin settings. Ask NVIDIA, or rerun with `./run_test openmm_bench`.
- The figure readings in sections 5 and 6 come from pixel positions, not from data files. The convergence data is not published.
- The `water_100k` frames are NPT frames. Their box lengths can differ from the 10.036 nm of the initial structure, so `G` ≈ 67 is approximate.
- No command on the site was run. The getting started example needs the release runtime and the CUDA 13.4 toolkit, and this machine has CUDA 13.3 and 12.9 ([0001](0001_non_openmm_integration_assessment.md#verification-of-this-assessment)).

## References

- Barnett, A. H., Magland, J., & af Klinteberg, L. (2019). A parallel nonuniform fast Fourier transform library based on an "exponential of semicircle" kernel. *SIAM Journal on Scientific Computing*, 41(5), C479–C504. https://doi.org/10.1137/18M120885X
- Deserno, M., & Holm, C. (1998). How to mesh up Ewald sums. I. A theoretical and numerical comparison of various particle mesh routines. *The Journal of Chemical Physics*, 109, 7678–7693. https://doi.org/10.1063/1.477414
- Liang, J., Lu, L., Barnett, A., Greengard, L., & Jiang, S. (2026). Accelerating molecular dynamics simulations using fast Ewald summation with prolates. *Nature Communications*, 17, 6681. https://doi.org/10.1038/s41467-026-73232-8
- Predescu, C., et al. (2020). The u-series: A separable decomposition for electrostatics computation with improved accuracy. *The Journal of Chemical Physics*, 152, 084113. https://doi.org/10.1063/1.5129393
- Simmonett, A. C., & Brooks, B. R. (2021). A compression strategy for particle mesh Ewald theory. *The Journal of Chemical Physics*, 154, 054112. https://doi.org/10.1063/5.0040966
