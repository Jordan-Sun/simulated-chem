# Simulated-Chem
Simulated-Chem is a GEOS-Chem load balancer and simulator.
The main goal of Simulated-Chem is to genereate a load balanced assignment as mapping files for our **modified** chemistry module of GEOS-Chem.
Therefore, one has to compile a **modified** version of GCHP by following **"GCHP Setup"** before using the load balanced assignment.


# GCHP Setup
## Compiling the modified GCHP
We modified `v14.5.2` of GCHP such that it can take a load balanced assignment as mapping files that one would generate by this to balance the workload, see **"Usage"** for details.

1. Clone the source code using the following script adapted from [GCHP download instructions](https://gchp.readthedocs.io/en/14.5.2/user-guide/downloading.html).
```Bash
git clone https://github.com/geoschem/GCHP.git GCHP
cd GCHP
git checkout tags/14.5.2
git submodule set-url src/GCHP_GridComp/GEOSChem_GridComp/geos-chem https://github.com/Jordan-Sun/geos-chem.git
git submodule update --init --remote
cd src/GCHP_GridComp/GEOSChem_GridComp/geos-chem
git switch experiment/dynamic_balance
```
2. Compile the code following the [GCHP compilation instructions](https://gchp.readthedocs.io/en/14.5.2/user-guide/compiling.html).

## Creating Run directory

Before one can run GCHP, one needs a Run directory. See [create a Run directory instructions](https://gchp.readthedocs.io/en/14.5.2/user-guide/rundir-init.html).

# Preparation
Before using Simulated-Chem, you will need two files: a workload file, and an original assignment file.
Some example inputs are provided under the `test` directory in this repository.
You may then skip the preparation part if you just want to work with those examples.

To generate these two input files, you need to first run the modified GCHP with diagnostic turned on once to obtain the KPP diagnostics. See below for details on how to generate these two input files from KPP diagnostics.

## Running the modified GCHP to generate KPP diagnostics 
Before running, you should enable `KppDiag` in the `HISTORY.rc` file in your run directory to generate the diagnostic file needed for Simulated-Chem. Additionally, set the `duration` and `frequency` to match that of the Chemistry module (20 minutes by default).

Check [list of configuration files](https://gchp.readthedocs.io/en/14.5.2/user-guide/configuration-files.html) for more details on configuration files and how to modify them.

To run GCHP, follow the [run instructions](https://gchp.readthedocs.io/en/14.5.2/user-guide/running.html). You may need to download input data for the model following [download data instructions](https://gchp.readthedocs.io/en/14.5.2/user-guide/getting-input-data.html).

## Workload file
The workload file stores a workload matrix in a csv format.
Each column represents the work within an interval, and each row represent the work of an atmospheric column that is simulated.

A good estimate for the workload is the number of KPP steps, which can be obtained by reading the KPP Diagnostic outputs through the `read_nc4_dir` function in `Workload`.
After that, you should write it out as a workload file using the `write_csv` function in `Workload`.

Note that the workload is independent of your system configuration. Therefore, you can reuse the workload file for any reassignments of the same resolution. The workloads for the first 7 days of `c24`, `c48`, `c90`, and `c180` resolution are included in the repo as examples.

## Original assignment file
The original assignment file stores how GEOS-Chem originally assigns columns to processes.

It can be obtained by reading any one of the Kpp Diagnostic output file through the `read_nc4` function in `Assignment`.
After that, you should write it out as an assignment file using the `write_csv` function in `Assignment`.

Unlike the workload file, the assignment file is dependent on the system configuration. The original assignments for 6 and 24 processors at `c24` resolution, 36, 144, and 576 processors at `c48` and `c90` resolution, as well as 144 and 576 processors at `c180` resolution are included in the repo as examples.

# Usage
## Generating a load balanced assignment
Once you have the workload and the original assignment file, you can use any of the following load balancing algorithms.

- **Global Balancing**: Applies a greedy algorithm that tries to balance across all processes.
- **Local Balancing**: Applies the same greedy algorithm but limited to balancing only within processes on the same node, assuming the default process to node mapping.
- **Hierarchical Greedy**: Takes a new process to node mapping. Then applies the same greedy algorithm also locally.
- You may encounter deprecated algorithms when you are exploring the code. For those, you are on your own.

To generate the assignment file, first set the configurations (`res, hosts, ptile`) in `batch_run_greedy.sh` to match your configuration, keep the `swap_alg_name` as `greedy`.
- For **Global Balancing**, set `hosts` to 1, `grouping` will be ignored.
- For **Local Balancing**, set `hosts` equal to the number of hosts, and set `grouping` to empty string "".
- For **Hierarchical Greedy**, set `hosts` equal to the number of hosts, and set `grouping` to path to the hostfile listing the new process to node mapping.

Then set the `workload_base` and `original_assignment_base` as the path to the directory holding your workload file and original assignment file, respectively.
After that, run the script with:
```Bash
./batch_run_greedy.sh
```

This should generate an assignment file.
However, before we can use it for GCHP, we need to convert it into mapping files.

## Converting an assignment file to mapping files readable by modified GCHP
To do so, you need to set the configurations (`res, procs`) in `assignment.py` to match your configuration. You also have to set the `workload_base`, `original_assignment_base`, and `assignment_base/strategy` as the path to the directory holding your workload file, original assignment file, and load balanced assignment file, respectively. After that, run the script with:
```Bash
python assignment.py
```

## Running the modified GCHP with mapping files

This subsection assumes you can already run GCHP without the mapping files. Refer to *"Running the modified GCHP to generate KPP diagnostics"* if GCHP won't run even without mapping files.
When the mapping files are not provided, our modified GCHP will run without any load balancing and behave exactly like the original GCHP.

Once you obtained the mapping files, create a `ReassignmentDir.rc` file directly under the run directory with the path to the mapping files directory. Then run GCHP just like one normally would following the same [run instructions](https://gchp.readthedocs.io/en/14.5.2/user-guide/running.html) before.

For benchmarking both original and load balanced code, turn off any diagnostic settings in the `HISTORY.rc` file. 