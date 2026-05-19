# Final Run 10 Jobs No Branches

## Goal

This package implements mode-specific SLURM launches for two unbranched axons with pseudo-branch recording points.

Each simulation launch fixes:

- topology: `one_node_branching`
- fiber diameter: `5.7` or `2.5`
- morphology scenario: `no_branch_reference`
- mode: one of `aligned`, `misaligned_0.5`, `misaligned_0.25`, `no_EC`, `no_EC_isolated`
- stimulation protocol: `sync` or `delay`

Inside each launch, the code runs:

- edge distance: `0.1`, `0.5`, `1.0`
- frequencies: `50..1000 Hz` with step `50`

For each distance, one HDF5 file is created, and all frequencies are stored inside that file as separate groups.

## Files in this folder

- `final_results_worker.py`
- `check_h5_integrity.py`
- `smoke_test_all.py`
- `submit_on_fd2p5_aligned.sbatch`
- `submit_on_fd2p5_misaligned_0p5.sbatch`
- `submit_on_fd2p5_misaligned_0p25.sbatch`
- `submit_on_fd2p5_no_EC.sbatch`
- `submit_on_fd2p5_no_EC_isolated.sbatch`
- `submit_on_fd5p7_aligned.sbatch`
- `submit_on_fd5p7_misaligned_0p5.sbatch`
- `submit_on_fd5p7_misaligned_0p25.sbatch`
- `submit_on_fd5p7_no_EC.sbatch`
- `submit_on_fd5p7_no_EC_isolated.sbatch`
- `submit_analyze_outputs.sbatch`

## Output folder structure

Outputs are written to:

```text
final_result/
├── one_node_branching/
│   ├── fiber_d_5.7_um/
│   │   └── no_branch_reference/
│   │       ├── distance_0.1/
│   │       ├── distance_0.5/
│   │       └── distance_1.0/
│   └── fiber_d_2.5_um/
│       └── no_branch_reference/
```

## File naming

### one_node_branching

- `on_nb` = no_branch_reference

### HDF5 format

```text
<prefix>_fd<diameter>_ed<distance>_<mode>_<sync|delay_Xms>_amp<abs_amp>.h5
```

Examples:

- `on_nb_fd5.7_ed0.1_aligned_sync_amp5.h5`
- `on_nb_fd5.7_ed0.1_aligned_delay_0p5ms_amp5.h5`
- `on_nb_fd5.7_ed0.1_misaligned_0.5_sync_amp5.h5`
- `on_nb_fd5.7_ed0.1_misaligned_0.25_sync_amp5.h5`
- `on_nb_fd5.7_ed0.1_no_EC_sync_amp5.h5`
- `on_nb_fd5.7_ed0.1_no_EC_isolated_sync_amp5.h5`

## Morphology

### Scenario `no_branch_reference`

Axon A:
- no branching

Axon B:
- no branching

Implementation:
- `parent_axon_nodes_A = 27`
- `branches_num_A = 0`
- `parent_axon_nodes_B = 27`
- `branches_num_B = 0`
- pseudo branch reference node = `8`

## Recording points

### no_branch_reference

Axon A:
- `before_like`
- `branch_like`
- `main_like`
- `terminal_main`

Axon B:
- `before_branch`
- `branch_point`
- `after_branch_main`
- `terminal_main`

## Stimulus

Production run:
- biphasic
- `amp = -5 nA` for `5.7 um`
- `amp = -1 nA` for `2.5 um`
- `phase_us = 40`
- `gap_us = 5`
- `t_start = 10 ms`
- `t_end = 1010 ms`
- `h_stop = 1010 ms`

Smoke test:
- same model structure
- short run: about `1 ms`
- uses a reduced frequency set just to verify file creation and integrity

## HDF5 structure

Each HDF5 file contains root attrs:

- `topology`
- `scenario`
- `fiber_diameter_um`
- `edge_dist_um`
- `mode`
- `amp_nA`
- `stim_description`
- `dt_ms`
- `h_stop_ms`
- `created_by`
- `test_mode`
- `frequencies_hz`

Root groups:

- `/AxonA_params`
- `/AxonB_params`
- `/Summary`

Frequency groups:

- `/Frequency_050Hz`
- `/Frequency_100Hz`
- ...
- `/Frequency_1000Hz`

Each frequency group stores:

- `AxonA`
- `AxonB`

and inside them the standard model output written by `run_simulation_two_axons()`.

## How SLURM runs

Simple mode:
- `1 sbatch = 1 diameter/mode configuration`
- no array jobs
- no chunk scheme
- no internal multiprocessing

Each sbatch runs sequentially through:

- `3 distances`
- `1 mode`
- `20 frequencies`

So one sbatch does:

- `3 * 1 * 20 = 60` model runs and writes `3` HDF5 files

All 10 simulation sbatch files together do:

- `10 * 60 = 600` model runs and write `30` HDF5 files for `no_branch_reference`

Each sbatch also has a stimulation protocol flag:

```bash
STIM_PROTOCOL="sync"   # sync or delay
STIM_B_DELAY_MS="0.5"  # used only when STIM_PROTOCOL="delay"
```

For delayed co-stimulation, change only:

```bash
STIM_PROTOCOL="delay"
```

The delayed files will include `_delay_0p5ms_` in the filename instead of `_sync_`.

## Commands

### Run one sbatch

Example:

```bash
sbatch final_run_10jobs_no_branches/submit_on_fd5p7_aligned.sbatch
```

### Run all 10 simulation jobs

```bash
sbatch final_run_10jobs_no_branches/submit_on_fd2p5_aligned.sbatch
sbatch final_run_10jobs_no_branches/submit_on_fd2p5_misaligned_0p5.sbatch
sbatch final_run_10jobs_no_branches/submit_on_fd2p5_misaligned_0p25.sbatch
sbatch final_run_10jobs_no_branches/submit_on_fd2p5_no_EC.sbatch
sbatch final_run_10jobs_no_branches/submit_on_fd2p5_no_EC_isolated.sbatch
sbatch final_run_10jobs_no_branches/submit_on_fd5p7_aligned.sbatch
sbatch final_run_10jobs_no_branches/submit_on_fd5p7_misaligned_0p5.sbatch
sbatch final_run_10jobs_no_branches/submit_on_fd5p7_misaligned_0p25.sbatch
sbatch final_run_10jobs_no_branches/submit_on_fd5p7_no_EC.sbatch
sbatch final_run_10jobs_no_branches/submit_on_fd5p7_no_EC_isolated.sbatch
```

### Analyze outputs

```bash
sbatch final_run_10jobs_no_branches/submit_analyze_outputs.sbatch
```

### Run smoke test

```bash
python final_run_10jobs_no_branches/smoke_test_all.py
```

### Validate HDF5 integrity manually

```bash
python final_run_10jobs_no_branches/check_h5_integrity.py
```

## Acceptance

The final package is acceptable only if `smoke_test_all.py` prints:

```text
ACCEPTANCE TEST PASSED
```
