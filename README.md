[<img width="256" src="https://www.seekpng.com/png/full/205-2051271_university-of-amsterdam-logo-university-of-amsterdam-logo.png" />](https://www.uva.nl/en)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[![](https://mns-research.nl/images/logo_hua931301caa9e2c039e68fbb874deb22a_17897_0x70_resize_lanczos_2.png)](https://mns-research.nl)

# ROFARS-MNS
Resource Optimization for Facial Recognition Systems

This project hosts the LSTM and UCB agent implementations for resource optimization in facial recognition systems, contributed by Jasper Bruin as part of his thesis work on Machine Learning-based resource optimization.

## Python-based Environment
* rofarsEnv.py - simulation environment
* agents.py - implementations for agents (LSTM and UCB agents by Jasper Bruin)
* example.py - example of how everything is put together

## Dataset
* data/train_test.txt (878,858 lines)

## Dependencies
* the code supports Python 3.7+
pip install numpy==1.24.2 pandas==1.5.3 tqdm==4.64.1

## Usage
* add your algorithm implementations to agents.py
* adapt example.py for use of experiments
python example.py

## Possible Solutions
* [Multi-armed bandit problem](https://en.wikipedia.org/wiki/Multi-armed_bandit)
  - [Epsilon-greedy strategy](https://gdmarmerola.github.io/ts-for-bernoulli-bandit)
  - [Thompson sampling](https://gdmarmerola.github.io/ts-for-bernoulli-bandit)
  - [Upper Confidence Bound (UCB)](https://gdmarmerola.github.io/ts-for-bernoulli-bandit)
  - [Non-stationary bandit](https://gdmarmerola.github.io/non-stationary-bandits)
  
* [Traffic flow prediction](https://www.sciencedirect.com/science/article/pii/S2210537922000725)

* [(Deep) reinforcement learning](https://en.wikipedia.org/wiki/Deep_reinforcement_learning)
  - https://spinningup.openai.com/en/latest/
  - https://stable-baselines.readthedocs.io/en/master/
  - https://huggingface.co/deep-rl-course/unit1/rl-framework

## Contributors
* Jasper Bruin - Author of the LSTM and UCB agents 

## Contact
Cyril Hsu - s.h.hsu@uva.nl
Dr. Chrysa Papagianni - c.papagianni@uva.nl

## License 

GNU GENERAL PUBLIC LICENSE  
Version 3, 29 June 2007

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
