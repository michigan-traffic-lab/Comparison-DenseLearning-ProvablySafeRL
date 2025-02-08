# Installation
> The overall installation process should take 5~10 minutes on a recommended computer.

## 1. Create a virtual environment and activate it
To ensure high flexibility, it is recommended to use a virtual environment when running the code. To set up the virtual environment, please follow the commands provided below:
```bash
conda create -n venv python=3.8.10
conda activate venv
```

## 2. Install all required packages
To install the packages required for this repository, execute the command provided below:
```bash
conda install pip==24.0
pip install -r requirements.txt
cd ../benchmarks/CommonRoad/gym_commonroad_sumo
pip install -e .
```
* Note: Please set the global environment variable `SUMO_HOME` to the path of the SUMO installation directory which can be found by `whereis sumo`. For example, `export SUMO_HOME=/home/user_name/anaconda3/envs/venv/bin/sumo`.

## 3. Install MATLAB and matlabengine
A MATLAB not higher than R2023a, not lower than R2020b, is needed to run the SPOT codes. The corresponding matlabengine version `{your version}` can be found at https://pypi.org/project/matlabengine.
```bash
pip install matlabengine=={your version}
```
For example, R2020b = 9.9.4, R2021a = 9.10.1, R2021b = 9.11.19, R2022a = 9.12.10, R2022b = 9.13.5, R2023a = 9.14.4. 

## 4. Install optimization solver
Additionally, an optimization solver is necessary for the provably safe reinforcement learning model. The implementation currently can be used with Gurobi (https://www.gurobi.com/) and SCIP (https://www.scipopt.org/index.php#download). If you have a Gurobi license, we recommend using Gurobi as it is more efficiently implemented. To install the solver, please follow the instructions on the respective website.

## 5. Install CoRA (Optional)
To regenerate the reachable set by MATLAB, the CORA toolbox needs to be installed, which is available at https://cora.in.tum.de. Otherwise, it is not required.

-> Next Page: [Code Introduction](code.md)