# Final Run 8 Jobs

## Goal

This package implements 8 independent production-ready SLURM launches.

Each launch fixes:

- topology: `connector_branching` or `one_node_branching`
- fiber diameter: `5.7` or `2.5`
- morphology scenario: `one_branch` or `multiple_branches`

Inside each launch, the code runs:

- edge distance: `0.1`, `0.5`, `1.0`
- mode: `aligned`, `misaligned_0.5`, `misaligned_0.25`, `no_EC`, `no_EC_isolated`
- frequencies: `50..1000 Hz` with step `50`

For each `(distance, mode)` pair, one HDF5 file is created, and all frequencies are stored inside that file as separate groups.

## Files in this folder

- `final_results_worker.py`
- `check_h5_integrity.py`
- `smoke_test_all.py`
- `submit_all_final_run.sh`
- `submit_cb_fd5p7_one_branch.sbatch`
- `submit_cb_fd5p7_multiple_branches.sbatch`
- `submit_on_fd5p7_one_branch.sbatch`
- `submit_on_fd5p7_multiple_branches.sbatch`
- `submit_cb_fd2p5_one_branch.sbatch`
- `submit_cb_fd2p5_multiple_branches.sbatch`
- `submit_on_fd2p5_one_branch.sbatch`
- `submit_on_fd2p5_multiple_branches.sbatch`

## Output folder structure

Outputs are written to:

```text
final_result/
├── one_node_branching/
│   ├── fiber_d_5.7_um/
│   │   ├── one_branch/
│   │   │   ├── distance_0.1/
│   │   │   ├── distance_0.5/
│   │   │   └── distance_1.0/
│   │   └── multiple_branches/
│   │       ├── distance_0.1/
│   │       ├── distance_0.5/
│   │       └── distance_1.0/
│   └── fiber_d_2.5_um/
│       ├── one_branch/
│       └── multiple_branches/
└── connector_branching/
    ├── fiber_d_5.7_um/
    │   ├── one_branch/
    │   └── multiple_branches/
    └── fiber_d_2.5_um/
        ├── one_branch/
        └── multiple_branches/
```

## File naming

### one_node_branching

- `on_ob` = one_branch
- `on_mb` = multiple_branches

### connector_branching

- `cb_ob` = one_branch
- `cb_mb` = multiple_branches

### HDF5 format

```text
<prefix>_fd<diameter>_ed<distance>_<mode>_amp<abs_amp>.h5
```

Examples:

- `on_ob_fd5.7_ed0.1_aligned_amp5.h5`
- `on_ob_fd5.7_ed0.1_misaligned_0.5_amp5.h5`
- `on_ob_fd5.7_ed0.1_misaligned_0.25_amp5.h5`
- `on_ob_fd5.7_ed0.1_no_EC_amp5.h5`
- `on_ob_fd5.7_ed0.1_no_EC_isolated_amp5.h5`

- `cb_mb_fd2.5_ed1.0_aligned_amp1.h5`
- `cb_mb_fd2.5_ed1.0_misaligned_0.5_amp1.h5`
- `cb_mb_fd2.5_ed1.0_misaligned_0.25_amp1.h5`
- `cb_mb_fd2.5_ed1.0_no_EC_amp1.h5`
- `cb_mb_fd2.5_ed1.0_no_EC_isolated_amp1.h5`

## Morphologies

Post-branch parameterization:

- main branch: `main_after_branch_diam_scale = 0.6`, `main_after_branch_param_mode = scaled_radial`
- daughter branch: `daughter_branch_diam_scale = 0.6`, `daughter_branch_param_mode = ascent_full`
- `scaled_radial` keeps parent-like longitudinal MRG geometry and scales radial parameters
- `ascent_full` uses the precomputed full MRG parameter set for the reduced diameter; ASCENT is not run

### Scenario `one_branch`

Axon A:
- no branching

Axon B:
- `(8 nodes) - (branch) - (18 nodes)`

Implementation:
- `parent_axon_nodes_A = 27`
- `branches_num_A = 0`
- `parent_axon_nodes_B = 27`
- `branches_num_B = 1`
- `branch_sequence_nodes_B = [8]`

### Scenario `multiple_branches`

Axon A:
- `(8 nodes) - (branch) - (4 nodes) - (branch) - (4 nodes) - (branch) - (4 nodes) - (branch) - (8 nodes)`

Axon B:
- same structure

Implementation:
- `parent_axon_nodes_A = 29`
- `branches_num_A = 4`
- `branch_sequence_nodes_A = [8, 4, 4, 4]`
- `parent_axon_nodes_B = 29`
- `branches_num_B = 4`
- `branch_sequence_nodes_B = [8, 4, 4, 4]`

## Recording points

### one_branch

Axon A:
- `before_like`
- `main_like`
- `terminal_main`

Axon B:
- `before_branch`
- `after_branch_main`
- `terminal_main`

### multiple_branches

For both axons, branch-specific labels are recorded using the model's existing branch groups.
At minimum this includes:
- `before_branch`
- `after_branch_main`
- `terminal_main`

If available, branch-point and daughter labels are also saved.

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
- `1 sbatch = 1 full configuration`
- no array jobs
- no chunk scheme
- no internal multiprocessing

Each sbatch runs sequentially through:

- `3 distances`
- `4 modes`
- `20 frequencies`

So one sbatch does:

- `3 * 4 * 20 = 240` model runs

All 8 sbatch files together do:

- `8 * 240 = 1920` model runs

## Commands

### Run one sbatch

Example:

```bash
sbatch final_run_8jobs/submit_cb_fd5p7_one_branch.sbatch
```

### Run all 8

```bash
bash final_run_8jobs/submit_all_final_run.sh
```

### Run smoke test

```bash
python final_run_8jobs/smoke_test_all.py
```

### Validate HDF5 integrity manually

```bash
python final_run_8jobs/check_h5_integrity.py
```

## Acceptance

The final package is acceptable only if `smoke_test_all.py` prints:

```text
ACCEPTANCE TEST PASSED
```
