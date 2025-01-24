# Comparison of Dense Learning and Provably Safe Reinforcement Learning for Autonomous Driving

## Table of Contents:
1. [Introduction of the Project](#Introduction-of-the-Project)
   - [Overview](#overview)
   - [Code Structure](#code-structure)
2. [System Requirements](#system-requirements)
    - [Hardware](#hardware)
    - [Software](#software)
3. [Getting Started](#getting-started)
   - [Installation](docs/installation.md)
   - [Code Introduction](docs/code.md)
   - [Evaluation](docs/evaluation.md)
   - [Visualization](docs/visualization.md)
4. [License](#license)

## Introduction of the Project

### Overview

A research community is dedicated to developing provably safe reinforcement learning methods to enhance the safety of autonomous vehicles (AVs). In this project, we aim to thoroughly evaluate the safety performance of a selected provably safe reinforcement learning model, as detailed in [this paper](https://ieeexplore.ieee.org/abstract/document/10068193). The evaluation will be conducted within the naturalistic and adversarial driving environment (NADE). Additionally, we will compare this model with our SafeDriver system to verify how effectively the Dense Learning algorithm can enhance AV safety.

To facilitate this assessment, we integrated the CommonRoad software with NADE using the CommonRoad-SUMO interface. Within the SUMO simulation, all background vehicles (BVs) will be controlled by NADE. The simulation data from these BVs will be converted into CommonRoad scenario objects and then sent to the provably safe reinforcement learning model, which will compute the necessary control inputs. Based on these inputs and the dynamic model of the system, the CommonRoad software will simulate the ego vehicle. The CommonRoad-SUMO interface ensures that all traffic participants' states are synchronized between the SUMO and CommonRoad simulations.

The provably safe reinforcement learning model is composed of a base reinforcement learning model and a safety shield. We evaluated the safety performance of the base model alone, the complete provably safe reinforcement learning model, and the base model enhanced with SafeDriver. During testing, the safety shield of the provably safe reinforcement learning model sometimes failed to find feasible solutions in critical scenarios, with an infeasibility rate reaching $1.75 \times 10^{-2}$. Originally, testing would halt if the safety shield encountered such issues, which was insufficient for a comprehensive safety evaluation of the motion planner. To ensure a fair comparison, we modified the approach such that the base model takes over if the safety shield cannot find a feasible solution. Among the three AV planning modules tested, the base model incorporating SafeDriver achieved the lowest crash rate in NADE at $2.22 \times 10^{-4}$, representing a 96.1% reduction in crashes compared to the base model and a 49.1% decrease compared to the base model coupled with a safety shield. 

### Code Structure
The code structure is shown in the following diagram.

<img src='docs/figure/code_structure.png' width='700'>

## System Requirements

### Hardware
This code can run on a computer with the following hardware configuration:
- RAM: 32+ GB
- CPU: 8+ cores, 3.0+ GHz/core

It is highly recommended to run this code on a High-Performance Computing (HPC) cluster to reduce the time for data collection and training.

### Software
This code is developed and tested under
- Ubuntu 18.04 operating system
- Python 3.8.10

## Getting Started
- [Installation](docs/installation.md)
- [Code Introduction](docs/code.md)
- [Evaluation](docs/evaluation.md)
- [Visualization](docs/visualization.md)

## License
This code is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE).