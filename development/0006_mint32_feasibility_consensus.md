# Feasibility of MINT32 coordinates in cuPotential: consensus note

Date: 2026-11-17

Source version: 0.3.5, runtime ABI 32

Reviewed upstream commit: `fbfc4cb6282cd52d33f7adbbcbaeaa5c19b47d74`

Paper: T.-S. Lee, "MINT32: A Minimum-Image INT32 Coordinate Representation for Fast and Accurate Molecular Dynamics on GPUs", ChemRxiv, posted 2026-01-03, [doi:10.26434/chemrxiv-2026-dshbp](https://doi.org/10.26434/chemrxiv-2026-dshbp). Published version: *J. Chem. Inf. Model.* 66(8), 4645–4657 (2026), [doi:10.1021/acs.jcim.5c03063](https://doi.org/10.1021/acs.jcim.5c03063), open copy [PMC13078822](https://pmc.ncbi.nlm.nih.gov/articles/PMC13078822/).

Paper version read: ChemRxiv refuses automated access (HTTP 403). This note checks paper statements against the posted abstract in the Crossref record and against the local revised manuscript source dated 2026-04-04 (`~/Data/CurrentProjects/MINT32/papers/mint32_amber_paper/tex/` and `~/Data/CurrentProjects/MINT32/table/table_benchmarks.tex`). The posted abstract reports a 7–10x drift reduction. The revised source adds Factor IX and reports 5–10x. Numbers below come from the revised source. Statements about the prototype code come from a local copy of the public [AMBER evaluation patch](https://github.com/taisung/MINT32_Data/blob/439c7758b374b5ec08d3392df3563577a73fba0d/mint32_amber_patch_2025-11-05.diff) at revision `439c7758`. This note did not read the published article text.

This note combines two independent assessments of the same question. Statements are labeled documented (a source states it), calculated (arithmetic run for this note), or inferred. No cuPotential GPU calculation, patch build, or MD trajectory was run for this note. The runtime implementation is not in this checkout, so every statement about runtime internals is an inference from the headers and documentation.

## Summary

- MINT32 maps one box length per axis onto the full 32-bit integer range. Integer subtraction then gives an exact, wrapped displacement. In an AMBER `pmemd.cuda` prototype it reduced NVE drift by 3.2x to 10x relative to SPFP, at 0% to 5% arithmetic cost (documented; ratios calculated).
- cuPotential's nonbonded path accepts only FP32 Cartesian positions. The closed runtime forms the pair displacements and mesh offsets (documented). No public entry point takes integer coordinates or displacements.
- Three routes exist. Route A keeps MINT32 in the engine and exports FP32. It works today and gives no MINT32 benefit in the nonbonded and PME paths. Route B adds integer input to the runtime. It gives the full benefit and needs NVIDIA. Route C uses engine-owned pair and mesh kernels with cuPotential's scalar math and convolution. It needs no runtime change but gives up the runtime's force pipeline.
- The size of the benefit in cuPotential is unknown. First measure how sensitive the existing plugin is to the FP32 input spacing (section 9).
- A native design should derive the mesh node and offset from a 64-bit integer product. The AMBER patch stores one FP32 grid coordinate instead (sections 2 and 6.1).

## 1. What the paper documents

| Element | MINT32 design |
| --- | --- |
| Representation | `X = round(x * S)`, `S = 2^32 / L` per axis, orthorhombic box. General cell: `U = round(u * 2^32)` on fractional coordinates `u = H^-1 r`. |
| Resolution | Uniform `L / 2^32`: 2.3e-8 Å for a 100 Å box. That is about 160x finer than the worst FP32 spacing in a centered box, and about 330x in an uncentered box. |
| Minimum image | `ΔX = X_i - X_j` in 32-bit arithmetic. Overflow wraps the result. No branch or rounding step. |
| Force input | `Δx = float(ΔX) * S^-1` in FP32. FP64 promotion before scaling gave no significant drift gain and cost too much on consumer GPUs. |
| NPT | Proposed: change `S`, keep `X`. Not implemented and not tested. |

The benchmarks are NVE at 1 fs with an 8 Å cutoff, fourth-order B-spline PME, and a 1e-7 Å SHAKE tolerance. SPFP and MINT32 ran on an RTX 3090, DPFP on a Titan V. The "Double Bond" variant evaluates bonded terms in FP64 and the "Float Bond" variant in FP32, both from integer differences. Drift is in `k_B T`/ns per degree of freedom at 300 K. The last column is calculated.

| System | Atoms | SPFP | MINT32, Double Bond | MINT32, Float Bond | DPFP | SPFP / MINT32 |
| --- | --- | --- | --- | --- | --- | --- |
| TIP3P water, 100 ns | 12,288 | +2.22e-4 | +2.22e-5 (one MINT32 row, variant not stated) | — | +2.09e-5 | 10.0 |
| DHFR, 50 ns | 22,930 | -4.64e-5 | -1.46e-5 | -9.25e-6 | -8.74e-6 | 3.2 and 5.0 |
| Factor IX, 50 ns | 90,906 | +3.48e-5 | +6.43e-6 | +4.24e-6 | +3.78e-6 | 5.4 and 8.2 |

MINT32 drift is 1.06x to 1.70x the DPFP drift (calculated). The prototype ran at 84% to 89% of production AMBER speed. Against an SPFP build with the same 32x32 tile layout, the MINT32 overhead was 0% to 5% (documented).

Documented limits: orthorhombic prototype, NVE only, three systems, and a 32-bit grid recommended for boxes below about 200 Å. Calculated: the spacing is 4.7e-8 Å at 200 Å and reaches 1e-7 Å (0.01 fm) at 429.5 Å.

## 2. Where the measured gain comes from

The manuscript's representation table gives the precision at each stage (documented):

| Stage | AMBER SPFP | MINT32 prototype |
| --- | --- | --- |
| Storage | FP64 | INT32 |
| Nonbonded displacement | FP32 | INT32 to FP32 |
| Bonded displacement | FP32 | INT32 to FP64 |
| PME grid coordinate | FP32 | INT32 to FP32 |
| Constraints | FP64 | INT32 to FP64 |
| Integration | FP64 | FP64 to INT32 |

Inferred: SPFP [Le Grand et al. 2013] already stores, constrains, and integrates in FP64, which is far finer than the MINT32 grid (7e-15 Å at 50 Å). In this comparison the gain therefore comes from the inputs of the force kernels, not from the stored state. The Float Bond result fits this reading: it drifted 34% to 37% less than Double Bond (calculated).

The inference is specific to the AMBER comparison. An engine whose baseline state is FP32 can also gain in integration and constraints. Measure that part separately.

**Displacements** (calculated, 400,000 random pairs near the edge of a centered 100 Å box). Subtracting FP32 copies of absolute coordinates gives errors up to 3.8e-6 Å at every separation. Integer-first subtraction with FP32 conversion gives errors up to 0.8e-7 Å near 1 Å and 6.1e-7 Å near 8 Å. The remaining error is the INT32-to-FP32 "staircase" that the manuscript describes, about `r * 2^-24` per rounding. The gain is about 50x at contact distance and about 6x at the cutoff.

**PME coordinates** (not settled). The manuscript's alignment analysis finds a worst-case grid-position error 132x to 231x smaller with MINT32 (documented). It attributes the water result to this. The patch, however, computes `u * K` in FP64 and stores it as one FP32 whole-grid coordinate (verified in the `kReImageCoord.h` hunk). The charge-spreading kernel reads that FP32 value (`kCQB.h` hunk). The only other integer read in `kPMEInterpolation.cu` is a debug dump that `MINT32_PME_SAMPLE` enables. Calculated: an FP32 value just below `K` = 48 to 144 has a rounding error up to 1.9e-6 to 7.6e-6 grid units. That is more than 250 times the manuscript's MINT32 bound and about twice its FP32 bound. The SPFP lines that the hunk removes store the same kind of value. Inferred: in the public patch the PME inputs improve little, and the direct-space displacements carry most of the gain. The sources do not decide this: the bound describes the representation, the patch describes one implementation, and no ablation separates the two paths.

Consequence for cuPotential: the coordinate format that the force kernels receive is what matters, and every conversion step must keep the precision.

## 3. Where coordinates enter cuPotential

Documented in the headers, unless marked:

| Entry point | Accepted coordinates | Consequence |
| --- | --- | --- |
| Cell list | Device FP32 positions, stride 3 or more ([`cell_list.h`](../include/cupo/cell_list.h#L108)). "Positions are wrapped internally; any periodic image is accepted" ([line 102](../include/cupo/cell_list.h#L102)). | Every nonbonded displacement starts from FP32 absolute coordinates. |
| Direct space | Borrows the cell list ([`dspace.h`](../include/cupo/dspace.h#L83)). | The runtime forms the displacements. |
| Custom pair callback | FP32 `r`, `r2`, `1/r`, `1/r2` ([`pair_device.cuh`](../include/cupo/pair_device.cuh#L17)). Returns energy and `dU/dr` ([line 24](../include/cupo/pair_device.cuh#L24)). | Called after the geometry exists. |
| Reciprocal space | Borrows the same list and its weights ([`kspace.h`](../include/cupo/kspace.h#L93)). Spreads after the list build ([line 38](../include/cupo/kspace.h#L38)). | Mesh offsets also derive from FP32 positions (inferred). |
| Tuples (bonded) | `float4` or `double4` positions ([`tuples.h`](../include/cupo/tuples.h#L148)), FP64 box ([line 150](../include/cupo/tuples.h#L150)). | An FP64 path exists. No integer type. |
| Box | FP32, nm, lower-triangular ([`box.h`](../include/cupo/box.h#L13)). Reduced triclinic allowed ([`cell_list.h`](../include/cupo/cell_list.h#L40)). | |

No exported function injects displacements, replaces the sorted coordinate storage, or returns a pair list (checked against every `CUPO_API` declaration in `include/cupo/`). Casting an `int4` array to `float*` does not help: the runtime would read integer bit patterns as floats.

The custom callback is not an extension point for MINT32 (inferred). Particle records are opaque bytes ([`cell_list.h`](../include/cupo/cell_list.h#L44)), so the callback could recompute `r` from integer coordinates in a record. The runtime would still apply its own FP32 direction, cutoff test, and mesh mapping.

OpenMM plugin (documented). The nonbonded adapter passes the FP32 `posq` array with stride 4 ([`CudaNonbondedKernels.cu`](../plugins/openmm/cpp/platforms/cuda/src/nonbonded/CudaNonbondedKernels.cu#L1567)) and rejects double precision ([line 929](../plugins/openmm/cpp/platforms/cuda/src/nonbonded/CudaNonbondedKernels.cu#L929)). The bonded adapter creates FP32 tuples ([`GroupedBonded.cu`](../plugins/openmm/cpp/platforms/cuda/src/bonded/GroupedBonded.cu#L179)) and also reads `posq` ([line 252](../plugins/openmm/cpp/platforms/cuda/src/bonded/GroupedBonded.cu#L252)). Only the integrator uses the mixed-precision correction ([`CudaLangevinMiddleKernel.cu`](../plugins/openmm/cpp/platforms/cuda/src/integrator/CudaLangevinMiddleKernel.cu#L393)). Inferred: for its force inputs, the plugin is in the same class as the paper's SPFP baseline. See [0005](0005_numerical_representations_layouts_and_determinism.md#precision-and-storage-in-the-openmm-plugin).

## 4. What an integrator can change

| Component | What you can change |
| --- | --- |
| `libcupo.so`: cell lists, direct and reciprocal force machinery, tuple execution, compact convolution | Configure and call its APIs. Its internal kernels cannot be edited and rebuilt from this checkout ([`fetch_cupo`](../fetch_cupo#L4), [`LICENSE`](../LICENSE#L12)). |
| `cupo_math.cu.o`: splitting and spreading functions | Link the supplied LTO-IR into your kernels ([`docs/ltoir.md`](../docs/ltoir.md#L93)). The underlying CUDA implementation source is not supplied ([`LICENSE`](../LICENSE#L13)). |
| OpenMM plugin: adapters, scheduling, integration, constraints, barostat | The source is available ([`plugins/openmm/`](../plugins/openmm/), [`LICENSE`](../LICENSE#L3)). You can modify and rebuild it. |
| Custom pair evaluators and your own kernels | You supply their implementation within the exposed interface contracts ([`pair.h`](../include/cupo/pair.h#L62), [`pair_device.cuh`](../include/cupo/pair_device.cuh#L30)). |

All four rows are documented. Three qualifications apply:

- Neither binary is in a GitHub clone. `fetch_cupo` downloads the release tarball when the tree has "neither cupo's libraries nor cupo's sources" ([`fetch_cupo`](../fetch_cupo#L9)). The tarball holds one `cupo_math.cu.o` per math variant beside the runtime ([`docs/ltoir.md`](../docs/ltoir.md#L9), [line 21](../docs/ltoir.md#L21), [line 69](../docs/ltoir.md#L69)).
- The runtime license forbids reverse engineering ([`LICENSE-cupo.txt`](../LICENSE-cupo.txt#L38)) and modification ([line 39](../LICENSE-cupo.txt#L39)) of the binaries. Editing an API header cannot add a coordinate mode, because the runtime must implement it.
- A rebuilt plugin still links the downloaded runtime ([`run_build`](../plugins/openmm/run_build#L45)). It can change what it passes to the runtime, not how the runtime uses it.

This table decides the routes. Route A only calls the runtime. Route B needs a change inside `libcupo.so`, so it needs NVIDIA. Route C puts the coordinate-dependent work into your own kernels and reuses `cupo_math.cu.o` and the standalone convolution API.

## 5. Integration routes

### Route A: integer state in the engine, FP32 at the API boundary

The engine keeps MINT32 (or FP64) state and writes an FP32 Cartesian mirror each step. This works today.

- Nonbonded and PME: no MINT32 benefit. The runtime receives rounded absolute positions.
- Bonded terms: use `CUPO_TUPLES_FLOAT64` ([`tuples.h`](../include/cupo/tuples.h#L30)) with a `double4` mirror built from the integers. The displacements keep their precision, at the cost of FP64 evaluation and a 32-byte record. The paper's best variant, FP32 bonded arithmetic on integer differences, is not available.
- Integration and constraints stay in the engine and keep their benefit.

Use Route A first in an engine that already has MINT32. It is also the baseline for later comparisons.

### Route B: integer coordinates as a runtime input

NVIDIA, or a party with the runtime source, adds an integer coordinate mode (section 6). This route keeps the runtime's force pipeline. The engine does not have to adopt MINT32 state. An engine with integer state passes its array directly. An engine with FP64 state converts once per step. The plugin in mixed precision converts `posq + posqCorrection`, about 48 significant bits, once per step (inferred). In single precision it has only FP32 positions, and the conversion gains nothing.

### Route C: engine-owned kernels with cuPotential math and convolution

Its parts are documented:

- The scalar splitting functions in `cupo_math.cu.o` take a squared distance ([`math/splitting.h`](../include/cupo/math/splitting.h#L28)). An engine pair kernel can form integer displacements first. The gradients are with respect to `r^2` ([line 19](../include/cupo/math/splitting.h#L19)).
- The spreading functions take a grid offset in [0, 1), in FP32 or FP64 ([`math/spreading.h`](../include/cupo/math/spreading.h#L21)). The engine controls how it derives that offset. The Apache-licensed sample does this on the host ([`usage.cu`](../samples/ltoir-usage/usage.cu#L416)).
- The standalone convolution in `libcupo.so` works on grids, not particles ([`convolution.h`](../include/cupo/convolution.h#L156)). Its FFT arithmetic stays FP32 ([line 35](../include/cupo/convolution.h#L35)). `CUPO_GRID_INT32` is a grid encoding and is unrelated to MINT32 ([line 50](../include/cupo/convolution.h#L50)).

The engine must supply the spatial traversal, force assembly, exclusions, self and background terms, the influence grid with its normalization and deconvolution, the energy reductions, and the force interpolation ([`docs/ltoir.md`](../docs/ltoir.md#L413)). Scalar-only use does not need `libcupo.so` ([line 565](../docs/ltoir.md#L565)). The speed of the runtime's direct-space kernel does not carry over, so measure the prototype independently. Use Route C for a prototype that does not wait for NVIDIA.

### Plugin-only changes

A plugin change can give integer state to the integrator and constraints. Mixed precision already integrates from compensated positions, so the expected value is small without Route B (inferred).

## 6. Changes for native support (Route B)

This is a proposed design. It does not describe existing functionality.

| Component | Change | Notes |
| --- | --- | --- |
| Coordinate contract | Add an integer mode with a defined type, stride, origin, and scale. Keep the FP32 entry point. | `cupo_cell_list_execute` takes `const float*` ([`cell_list.h`](../include/cupo/cell_list.h#L117)). The change needs a new ABI version and a capability bit. The current bits do not cover it ([`version.h`](../include/cupo/version.h#L32)). |
| Cell list | Bin from the integer: `(U * n_cells) >> 32`. Carry the integer words through the spatial sort. | Every integer is a valid wrapped coordinate, so the "unwrappable positions" error ([`cell_list.h`](../include/cupo/cell_list.h#L103)) and the wrap rounding disappear. An `int4` record has the 16 bytes of a `float4`, but sorted copies and mirrors add traffic. Specify the meaning of `w` if it carries charge bits. |
| Direct pairs | Subtract integers, then convert and scale in FP32. Use the result for the cutoff test, the radial geometry, and the force direction. | Three conversions and three multiplies per pair, or six multiplies for a reduced triclinic box. The paper's 0% to 5% holds for AMBER's kernel on an RTX 3090. Integer-to-float conversion can run at a lower rate than FP32 arithmetic on current GPUs. This note did not verify that. |
| Mesh mapping | Node and offset from one 64-bit product, for spreading and gathering. | See 6.1. |
| Box and NPT | The device box defines `H`. A box change leaves the integers unchanged. | Exact only for atomic scaling. The plugin barostat displaces molecular centroids ([`CudaBarostatKernel.cu`](../plugins/openmm/cpp/platforms/cuda/src/barostat/CudaBarostatKernel.cu#L43)), so the engine converts again after a move. |
| Tuples | Optional integer input. | Low priority: FP64 tuples exist. |
| Force output | None. | Fixed64 with 32 fractional bits stays ([`dspace.h`](../include/cupo/dspace.h#L198)). The force quantum `2^-32` is unrelated to the coordinate spacing `L / 2^32`. |

### 6.1 Mesh mapping

```text
product = uint64(u) * G
node    = product >> 32
offset  = (product & 0xffffffff) / 2^32
```

The node is exact for any mesh size. The planner picks measured cuFFT sizes, 7-smooth sizes, or compiled compact sizes ([`kspace.h`](../include/cupo/kspace.h#L51)), so a mesh axis is often not a power of two. Only the offset becomes FP32. Calculated: a low word within 128 of `2^32` rounds to an offset of exactly 1.0. The window requires [0, 1) ([`math/spreading.h`](../include/cupo/math/spreading.h#L21)), so handle that case explicitly.

Verified example: for `G = 64` and phases 1760936591 and 1760936599, early FP32 conversion gives 26.2399997711 for both. The split gives node 26 with offsets 0.2399999946 and 0.2400001138.

### 6.2 Modular arithmetic and the box

Signed integer overflow is undefined behavior in C++, and the patch subtracts `int` values directly. Specify unsigned subtraction and a defined reinterpretation of the result. Also define the rounding of coordinate updates, half-box ties, range checks before float-to-integer conversion, and image counters for unwrapped trajectories.

The nonbonded box is FP32 and the tuple box is FP64. With fractional integers, `Δr = H * Δu`, so the box enters every displacement. One box definition must serve the integrator, cell grid, direct forces, reciprocal operators, and tuples.

### 6.3 Skewed cells

Wrapping each fractional component does not select the shortest Cartesian image in general. Verified example: for lattice vectors (10, 0, 0), (4.9, 10, 0), (0, 0, 10) nm, the wrapped fractional displacement (0.49, 0.30, 0) has length 7.0411 nm. Subtracting the first vector gives 4.7092 nm.

Calculated for this note: component wrapping is still correct for every pair inside the cutoff when the cutoff is at most half the smallest perpendicular cell width. It also never moves a farther pair inside the cutoff. Random pairs in three reduced cells confirm this. The manuscript's truncated octahedron meets the condition (width 46.1 Å, cutoff 8 Å).

Quantities evaluated beyond the cutoff need a full lattice-image procedure. Examples are the exclusion corrections ([`dspace.h`](../include/cupo/dspace.h#L73)) and tuples with `use_pbc` ([`tuples.h`](../include/cupo/tuples.h#L63)). In a cell-list kernel, each neighbor cell has an integer lattice translation that the runtime can add exactly (inferred). Start with orthorhombic, fixed-box NVE.

## 7. Numerical check of the conversion boundary

This host example needs only NumPy, and its output is reproducible. In a 10 nm box, the integer states `Xi` = 1717986918 and 1717986926 both round to `float32(xi)` = 4.0 nm. The absolute-first path therefore prints 0.09999990463256836 nm for both. The integer-first path prints 0.09999999403953552 and 0.09999997913837433 nm. It distinguishes the states, but it does not keep every grid step, because the conversion of a large integer difference still rounds.

```python
import numpy as np

h = 10.0 / (1 << 32)
Xi = round(4.0 / h)
Xj = round(4.1 / h)
for shift in (0, 8):
    a = Xi + shift
    d = ((Xj - a + (1 << 31)) % (1 << 32)) - (1 << 31)
    absolute_first = np.float32(Xj * h) - np.float32(a * h)
    integer_first = np.float32(d) * np.float32(h)
    print(shift, float(absolute_first), float(integer_first))
```

## 8. Open questions

1. **The error budget differs.** The paper's baselines cut the direct PME term at 8 Å. cuPotential's C2 Prolate short-range term and its first two derivatives vanish at the cutoff ([`math/splitting.h`](../include/cupo/math/splitting.h#L22)). [0003](0003_speedup_theory_and_mechanisms.md#22-prolate-splitting) reports an NVIDIA NVE test in which that configuration drifted about 478 times less than OpenMM's default PME, with different meshes. If coordinate noise dominates the remaining drift, integer input helps more than in AMBER. If the mesh or the FP32 pair arithmetic dominates, it helps little.
2. **The units differ.** The paper uses `k_B T`/ns per degree of freedom. The NVIDIA figures in 0003 use `ns^-1`. This checkout does not define that normalization. Convert both before any comparison.
3. **The production benefit is indirect.** Production runs use thermostats, and the manuscript calls drift a diagnostic. The value for a 4 fs NPT run lies in constraint quality and reproducibility.
4. **Repeatability.** An FP32 calculation is already repeatable for identical inputs under the conditions in [0005](0005_numerical_representations_layouts_and_determinism.md#conditions-and-limits-of-determinism). MINT32 does not make force reductions, constraint solves, or FFTs bit-identical. Integer input adds two narrower properties (inferred). The result no longer depends on which periodic image the caller supplies. It also no longer depends on how a mixed-precision position is split into high and low parts ([`test_reorder.py`](../plugins/openmm/tests/test_reorder.py#L22)).

## 9. Recommended sequence

1. **Measure the sensitivity with the existing plugin**. No runtime change is needed.
   - Run NVE at 1 fs in mixed precision with the plugin `NonbondedForce`, whose default is Prolate ([`test_plugin_vs_stock.py`](../plugins/openmm/tests/test_plugin_vs_stock.py#L1484)), and a stock `VerletIntegrator`, which the tests combine with it ([`test_compact_policy.py`](../plugins/openmm/tests/test_compact_policy.py#L52)). Use a tight constraint tolerance. Use no thermostat, barostat, or momentum removal.
   - Start with rigid water, which has no bonded terms, to isolate the nonbonded and PME inputs. Then run DHFR, where the FP32 tuples read the same shifted `posq`.
   - Translate the initial positions by `T = k * L` for `k` in {-0.5, 0, 1, 3, 7, 15}. The cell list accepts any periodic image, but the limit for "unwrappable" positions is not documented, so check the large shifts first. OpenMM does not reorder atoms when the plugin force is the only nonbonded force ([`CudaNonbondedKernels.cu`](../plugins/openmm/cpp/platforms/cuda/src/nonbonded/CudaNonbondedKernels.cu#L1099); [0003](0003_speedup_theory_and_mechanisms.md#35-cuda-graphs-and-host-synchronization)), so `posq` keeps the offset. Confirm this by reading the positions back.
   - Plot drift against the FP32 spacing at the largest coordinate. Calculated for `L` = 6.2 nm: it doubles with each step, from 2.4e-7 nm to 7.6e-6 nm. A curve that is flat at small shifts shows that FP32 inputs do not limit this path.
   - Do not extrapolate to `L / 2^32`. Calculated for the same box: an integer-first FP32 displacement still carries up to 1.0e-7 nm near a 0.9 nm cutoff and 1.3e-8 nm near 0.1 nm when the scale `L / 2^32` is rounded to FP32. With an exact scale, the value near the cutoff is 0.75e-7 nm. The centered FP32 case (`k` = -0.5) carries up to 2.4e-7 nm.
2. **If the drift is sensitive,** build a Route C prototype for an orthorhombic box. Validate it in this order: representation and geometry against a high-precision reference; mesh nodes, offsets, and weights across boundaries; force comparisons with only the coordinate representation varied; NVE with timestep and tolerance sweeps; then speed and repeatability with a fixed FFT backend.
3. **Send NVIDIA the measurements and the Route B proposal:** integer fractional input, the exact mesh split, defined modular arithmetic, and orthorhombic boxes first. Ask for the pair-kernel cost on SM86, SM89, and SM120.
4. **For an engine that already stores MINT32,** start with Route A. Compare its NVE drift with the engine's own integer nonbonded kernels on the same system. The difference is what Route B or C must recover.
5. **If the drift is not sensitive,** keep MINT32 in the engine for integration and constraints. Do not request a runtime change.

Related notes: [0001](0001_non_openmm_integration_assessment.md), [0003](0003_speedup_theory_and_mechanisms.md), [0004](0004_interaction_scope_and_stream_scheduling.md), and [0005](0005_numerical_representations_layouts_and_determinism.md).

## References

- Le Grand, S., Götz, A. W., & Walker, R. C. (2013). SPFP: Speed without compromise—A mixed precision model for GPU accelerated molecular dynamics simulations. *Computer Physics Communications*, 184, 374–380. https://doi.org/10.1016/j.cpc.2012.09.022
- Lee, T.-S. (2026). MINT32: A Minimum-Image INT32 Coordinate Representation for Fast and Accurate Molecular Dynamics on GPUs. *ChemRxiv*. https://doi.org/10.26434/chemrxiv-2026-dshbp
- Lee, T.-S. (2026). MINT32: A Minimum-Image INT32 Coordinate Representation for Fast and Accurate Molecular Dynamics on GPUs. *Journal of Chemical Information and Modeling*, 66(8), 4645–4657. https://doi.org/10.1021/acs.jcim.5c03063
