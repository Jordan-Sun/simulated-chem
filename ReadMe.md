# Simulated-Chem
Simulated-Chem is a GEOS-Chem load balancer and simulator.
It can genereate a load balanced assignment for the chemistry module for GEOS-Chem based on the workload, either column time or estimates such as number of KPP steps.
It can also simulate the time an assignment would take based on the workload, as well as summarizing the amount of column movements an assignment would require.

# GCHP Setup
## Compiling the modified GCHP
We modified v14.5.2 of GCHP to take an assignment file that one would generate by this to balance the workload. One may clone the source code using the following script adapted from [GCHP download instructions](https://gchp.readthedocs.io/en/14.5.2/user-guide/downloading.html).
```Bash
git clone https://github.com/geoschem/GCHP.git GCHP
cd GCHP
git checkout tags/14.5.2
git submodule set-url src/GCHP_GridComp/GEOSChem_GridComp/geos-chem https://github.com/Jordan-Sun/geos-chem.git
git submodule update --init --remote
cd src/GCHP_GridComp/GEOSChem_GridComp/geos-chem
git switch experiment/dynamic_balance
```
The rest are the same as the standard procedure, just compile the code following the [GCHP compilation instructions](https://gchp.readthedocs.io/en/14.5.2/user-guide/compiling.html).

## Running the modified GCHP without reassignment files
When the reassignment files are not provided, our modified GCHP will not do any load balancing and behave exactly like the original GCHP. To do so, first [create a Run directory](https://gchp.readthedocs.io/en/14.5.2/user-guide/rundir-init.html), then follow the [run instructions](https://gchp.readthedocs.io/en/14.5.2/user-guide/running.html). You may need to download input data for the model following [download data instructions](https://gchp.readthedocs.io/en/14.5.2/user-guide/getting-input-data.html).

Before running, you might want to modify a couple settings in the `HISTORY.rc` file in your run directory to generate the necessary diagnostic files. Check [list of configuration files](https://gchp.readthedocs.io/en/14.5.2/user-guide/configuration-files.html) for more details.

# Preparation
Before using the load balancer, you will need two files: a workload file, and an original assignment file. Some example inputs are provided under the `test` directory in this repository.

## Workload file
The workload file stores workload matrix in a csv format.
Each column represents the work within an interval, and each row represent the work of an atomespheric column that is simulated.

The number of KPP steps serves as a good estimate for the workload, you can obtain it by enabling `KppDiag` and setting the duration and frequency to match that of the Chemistry module (20 minutes by default) when running GCHP.
The Kpp Diagnostic outputs can then be converted into workload through the `read_nc4_dir` function in `Workload`.

You can write it out as a workload file using the `write_csv` function in `Workload`, and reuse workload file for reassignments for the same resolution. The workload for the first 7 days of `c24` and `c48` resolution are included in the repo as an example.

## Original assignment file
The original assignment file stores how GEOS-Chem originally assigns columns. You can obtain it by reading any one of the Kpp Diagnostic output file through the `read_nc4` function in `Assignment`.

You can write it out as an assignment file using the `write_csv` function in `Assignment`, and reuse assignment file for reassignments for the same resolution and number of processors. The original assignment for 6 and 24 processors at `c24` resolution, as well as 36, 144, and 576 processors at `c48` resolution, are included in the repo as an example.

# Usage
## Generating a load balanced assignment
Once you have the neceesary files, you can use any of the algorithms to generate a load balanced assignment.
Here, we will use the greedy heuristic as an example to generate the assignment.

First, set the configurations (`res, hosts, ptile`) in `greedy.py` to match your configuration, keep the `swap_alg_name` to be `greedy`.
Then set the `workload_base` and `original_assignment_base` as the path to the directory holding your workload file and original assignment file, respectively.
After that, run the script with:
```Bash
python greedy.py
```

## Converting an assignment file to mapping files readable by GCHP
After the previous step, you should have obtained an assignment file. Before we can use it as an reassignment file for GCHP, we need to convert it into mapping files.
To do so, you need to set the configurations (`res, procs`) in `assignment.py` to match your configuration. You also have to set the `workload_base`, `original_assignment_base`, and `assignment_base/strategy` as the path to the directory holding your workload file, original assignment file, and load balanced assignment file, respectively. After that, run the  script with:
```Bash
python assignment.py
```

The script will also automatically simulate the amount of time that GEOS-Chem would take given the workload assuming no communication overhead and number of movements the reassignment would require.
