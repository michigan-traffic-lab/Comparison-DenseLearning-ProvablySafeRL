# Evaluation
Go to the folder`benchmarks/CommonRoad` at first by running:
```bash
cd benchmarks/CommonRoad
```

## Evaluation with the Commonroad benchmarks

Testing in the basic environment can be conducted by running 
```bash
python deployment.py --output_path <your-output-path> --mode <tested-model-name>
```
To test the base model (i.e., unsafe planner), set `--mode unsafe`. To test the provably safe reinforcement learning model (i.e., safe planner), set `--mode safe`. Note that the developed SafeDriver is currently not supported in this settings.

## Evaluation with NADE

We further implement the NADE SUMO-Commonroad interactive environment in `gym_commonroad_sumo/gym_commonroad_sumo/gym_commonroad_nade.py`. To use this environment, run
```bash
python deployment_nade.py --output_path <your-output-path> --mode <tested-model-name>
```
To test the base model (i.e., unsafe planner), set `--mode unsafe`. To test the provably safe reinforcement learning model (i.e., safe planner), set `--mode safe`. To test the SafeDriver, set `--mode safedriver`. All results are recorded in the `<your-output-path>` folder. Normally, we only save data for crash scenarios. If you want to save data for all scenarios, add `--test` to the command.

The file structure of the testing results is shown below:

```
your-output-path/
|__crash/ # data for crash scenarios
|__saved_data/ # detailed information for planners within each scenario
|__statistical_results/ # statistical results for this experiment
|__tested_and_safe/ # data for safe scenarios
|__weight0.npy # weight of each testing episode
```

## Testing results
To calculate the final crash rate, run
```bash
python NADE_result_analysis.py --root_folder <path-to-your-result-folder>
```
Statistical results are saved in `results/statistical_results`.

Since it usually takes thousands of CPU*hours to get a reliable crash rate, we provided the evaluation results [here](https://zenodo.org/records/14837947). The data folder has the following structure:

```
Data_Comparison-DenseLearning-ProvablySafeRL/
|_evaluation_basemodel/ # evaluation results for the base model
  |_crash/ # data for crash scenarios
  |_leave_network/ # data for offroad scenarios
  |_saved_data/ # detailed information for planners within each scenario
  |_statistical_results/ # statistical results for this experiment
  |_tested_and_safe/ # data for safe scenarios
  |_weight0.npy # weight of each testing episode
  |_...
|_evaluation_basemodel_safetyshield/ # evaluation results for the base model with the safety shield, which has the same structure as evaluation_basemodel
|_evaluation_basemodel_safedriver/ # evaluation results for the base model with SafeDriver, which has the same structure as evaluation_basemodel
```

As shown in the folloiwing table, the base model with SafeDriver demonstrated the lowest crash rate in NDE at $2.22 \times 10^{−4}$. This represents a 96.1% reduction compared to the base model and a 49.1% decrease compared to the base model with a safety shield.

| AV planning module                  | Crash Rate in NDE       |
|------------------------|------------------------|
| Base model             | $5.69 \times 10^{−3}$  |
| Base model with the safety shield (i.e., provably safe reinforcement learning model) | $4.36 \times 10^{−4}$  |
| Base model with SafeDriver   | $2.22 \times 10^{−4}$  |


<- Last Page: [Code Introduction](code.md)

-> Next Page: [Visualization](visualization.md)