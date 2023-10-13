
[<img width="256" src="https://www.seekpng.com/png/full/205-2051271_university-of-amsterdam-logo-university-of-amsterdam-logo.png" alt="University of Amsterdam Logo" />](https://www.uva.nl/en)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[<img src="https://mns-research.nl/images/logo_hua931301caa9e2c039e68fbb874deb22a_17897_0x70_resize_lanczos_2.png" alt="MNS Research Logo" />](https://mns-research.nl)

# ROFARS-MNS
## Resource Optimization for Facial Recognition Systems

Facial recognition systems have become indispensable in various sectors like security, marketing, and healthcare. However, deploying them in real-time scenarios presents noteworthy computational challenges. This thesis delves deep into the capabilities and boundaries of machine learning algorithms, including UCB-1, SW-UCB, D-UCB, and LSTM, to enhance resource allocation and predict traffic flow for these systems.

### Key Findings:
- **UCB-1** is computationally efficient, ideal for quick decisions but struggles in non-stationary environments.
- **SW-UCB** and **D-UCB** perform better in dynamic environments due to their recency bias, but their efficiency is contingent upon hyperparameter tuning.
- **LSTM** excels in stationary or slowly changing situations but faces challenges during rapid transitions.
- Data handling techniques, particularly imputation methods within the LSTM framework, can drastically affect outcomes.

Conclusively, when selecting an algorithm for practical applications, it's vital to weigh the specific needs and constraints of the scenario. Future research can focus on improving these algorithms and exploring better data handling strategies.

### System Diagrams
![Data progression from multiple cameras to the facial recognition system.](images/camera.png)

The above diagram showcases the data flow from multiple cameras through the predictive unit, leading to the facial recognition system.

![Interactions between agents and the environment.](images/agentscheme.png)

### Code Structure & Environment
- **rofarsEnv.py**: Simulation environment.
- **agents.py**: Contains implementations for LSTM and UCB agents (Authored by Jasper Bruin).
- **example.py**: Demonstrates the integration of all components.
- **UCBtest.py**: Script for testing and training various UCB agents.
- **RNNtest.py**: Script that employs historical traffic data from agents in `agents.py` for testing and training.

### Dataset
- `data/train_test.txt` (878,858 entries)

### Dependencies
The code is compatible with Python 3.7+. To set up the required environment:
`pip install numpy==1.24.2 pandas==1.5.3 tqdm==4.64.1`


### Usage

1. Extend `agents.py` with your algorithm implementations.
2. Modify `example.py` to suit your experiments.
3. Run `python example.py`


### Potential Approaches & Solutions

- **Multi-armed Bandit Problem**:
    
    - [Epsilon-greedy strategy](https://gdmarmerola.github.io/ts-for-bernoulli-bandit)
    - [Thompson sampling](https://gdmarmerola.github.io/ts-for-bernoulli-bandit)
    - [Upper Confidence Bound (UCB)](https://gdmarmerola.github.io/ts-for-bernoulli-bandit)
    - [Non-stationary bandit](https://gdmarmerola.github.io/non-stationary-bandits)
- **Traffic Flow Prediction**: [Research Article](https://www.sciencedirect.com/science/article/pii/S2210537922000725) <img width="400" src="https://ars.els-cdn.com/content/image/1-s2.0-S2210537922000725-gr1_lrg.jpg" alt="Traffic Flow Prediction">
    
- **(Deep) Reinforcement Learning**:
    
    - [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/)
    - [Stable Baselines](https://stable-baselines.readthedocs.io/en/master/)
    - [Deep RL Course by Hugging Face](https://huggingface.co/deep-rl-course/unit1/rl-framework)

### Contributors

- Jasper Bruin: Authored LSTM and UCB agents.

### Contact

- **Cyril Hsu**: [s.h.hsu@uva.nl](mailto:s.h.hsu@uva.nl)
- **Dr. Chrysa Papagianni**: [c.papagianni@uva.nl](mailto:c.papagianni@uva.nl)
- **Jasper Bruin**: [jasperbruin@me.com](mailto:jasperbruin@me.com)

### License

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

This software adheres to the GNU GENERAL PUBLIC LICENSE (Version 3, 29 June 2007). It can be redistributed and/or modified under the conditions stated in the license. Read the full terms [here](https://www.gnu.org/licenses/).
