# Code Introduction
This project is developed based on the open-source code for the provably safe reinforcement learning model. You can find the [code](https://codeocean.com/capsule/9949621/tree/v1) and the [paper](https://ieeexplore.ieee.org/abstract/document/10068193) of the model.

## Code Structure
The code structure is shown in the following diagram.
```
Comparison-DenseLearning-ProvablySafeRL/
|_benchmarks/
  |_CommonRoad/ # CommonRoad environment
    |_agent/ # source code and model parameters for the base model
    |_analysis_tools/ # analysis tools
    |_auxiliary/ # auxiliary functions
    |_gym_commonroad_sumo/
      |_gym_commonroad_sumo/ # CommonRoad-SUMO interface
        |_action/ # action space and vehicle models
        |_NDE_RL_NADE/ # NADE environment
        |_observation/ # observation space
        |_reward/ # reward function for default CommonRoad benchmarks
        |_simulation/ # CommonRoad-SUMO simulation environment
        |_utils/ # utility functions
        |___init__.py
        |_commonroad_sumo_env.py # GYM environment for CommonRoad benchmarks
        |_draw_reachable_set.py # visualization function
        |_file_reader.py # read CommonRoad scenario files
        |_file_writer.py # write CommonRoad scenario files
        |_gym_commonroad_nade.py # GYM environment for NADE
      |_setup.py
    |_reachability/ # reachable set and calculation methods
    |_safedriver/ 
      |_models/ # models for SafeDriver
      |_yaml_config/ # configuration files for SafeDriver
      |_SafeDriver.py # SafeDriver implementation
    |_source_data/ # data for NADE
    |_spot_predictor/ # SPOT predictor
    |_src/ # source code for the provably safe reinforcement learning model
    |_sumo_config/ # SUMO configuration files
  |_deployment_nade.py # main script for evaluation with NADE
  |_deployment.py # main script for evaluation with CommonRoad benchmarks
  |_NADE_result_analysis.py # script for analyzing the evaluation results
|__docs/ # documentation
|__LICENSE # license of this repository
|__README.md # guidance for running the code
|__requirements.txt
```

## Changes
We have made the following changes compared to the original code to ensure a fair comparison between the provably safe reinforcement learning model and the Dense Learning algorithm.

- **Evaluation environment**

    We implement an NADE environment for Commonroad agent testing, instead of using the original pre-generated scenarios. Support code for NADE is in the `gym_commonroad_sumo/gym_commonroad_sumo/NDE_RL_NADE` folder. For more details of the implementation, please see the [code](https://github.com/michigan-traffic-lab/Dense-Deep-Reinforcement-Learning/) and the [paper](https://www.nature.com/articles/s41467-021-21007-8).

- **Implementation of Set-Based Prediction of Traffic Participants**

    Set-Based Prediction of Traffic Participants ([SPOT](https://ieeexplore.ieee.org/abstract/document/7995951/)) is used to predict the occupancy of background vehicles (BVs). Then, the safety shield of the provably safe reinforcement learning model will solve a mixed-integer optimization problem to avoid intersections with these occupancy sets of BVs. The open-source code can be found [here](https://koschi.gitlab.io/spot). Note that the public version only supports M1 and M2 reachable sets, while M1, M2, and M3 reachable sets are used in the paper. Therefore, we have to ignore the M3 reachable set in the prediction of BVs.

- **Reachable set**

    To ensure the fairness of comparison, we change the vehicles' length and width from [4.6, 1.6] meters to [5, 1.8] meters, and modify the output acceleration range from [-1,1; -0.2,0.2] meters per second squared to [-4,2; -0.2,0.2] meters per second squared and regenerate the reachable sets.

- **Infeasible situations**

    During the evaluation of the provably safe reinforcement learning model, the safety shield occasionally fails to find feasible solutions in safety-critical scenarios, with an infeasibility rate as high as $1.75 \times 10^{−2}$. In the original implementation, the testing episode would terminate upon encountering such situations, which is inadequate for thoroughly evaluating the safety performance of the motion planner. To make a fair comparison, we have implemented that the base model takes control if the safety shield is unable to find a feasible solution.

<- Last Page: [Installation](installation.md)

-> Next Page: [Evaluation](evaluation.md)

