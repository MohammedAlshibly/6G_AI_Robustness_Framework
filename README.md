# 6G_AI_Robustness_Framework
Overview
This repository contains the Python implementation developed for the research project:
“Improving Robustness Real-World Operation of 6G Wireless Networks Using Artificial Intelligence”
The framework provides a modular and research-oriented implementation for evaluating the robustness of next-generation 6G wireless networks under dynamic and adverse operating conditions. It integrates stochastic geometry, wireless channel modeling, dynamic blockage simulation, graph neural networks (GNNs), reinforcement learning (PPO skeleton), and performance evaluation modules.
The repository is designed to support reproducible research and facilitate future extensions in AI-enabled wireless communication systems.
________________________________________
Key Features
Wireless Environment Modeling
•	Homogeneous Poisson Point Process (PPP) deployment.
•	Dynamic user and infrastructure placement.
•	Stochastic geometry-based network representation.
Channel Modeling
•	CI-Alpha-Mu inspired path-loss model.
•	Atmospheric attenuation effects.
•	Shadow fading approximation.
•	SINR calculation utilities.
Dynamic Blockage Modeling
•	Moving rectangle blockage process.
•	Vehicle and pedestrian obstruction simulation.
•	Link blockage detection mechanisms.
Graph-Based Network Representation
•	Graph snapshot generation from network datasets.
•	Node feature extraction.
•	Dynamic adjacency construction.
Artificial Intelligence Components
•	Message Passing Neural Network (MPNN) encoder.
•	Graph Neural Network (GNN) architecture.
•	PPO-based reinforcement learning agent skeleton.
•	Extensible architecture for future AI algorithms.
Performance Evaluation
•	Throughput analysis under dynamic blockages.
•	Latency evaluation during traffic surges.
•	Robustness assessment under jamming attacks.
•	Publication-quality figure generation.
________________________________________
Repository Structure
6G_AI_Robustness_Framework/
│
├── main.py
├── README.md
├── requirements.txt
├── LICENSE
├── CITATION.cff
│
├── data/
│   ├── nodes.csv
│   ├── edges_time_series.csv
│   ├── kpis_time_series.csv
│   ├── actions_time_series.csv
│   ├── adversary_events.csv
│   └── scenarios.csv
│
├── outputs/
│   ├── figure1_cdf_blockage.png
│   ├── figure2_latency_eMBB.png
│   ├── figure3_latency_URLLC.png
│   └── figure4_jamming_throughput.png
│
└── docs/
________________________________________
Dataset Requirements
The framework expects the following CSV files:
File	Description
nodes.csv	Network nodes and coordinates
edges_time_series.csv	Dynamic connectivity information
kpis_time_series.csv	Performance indicators
actions_time_series.csv	AI actions and decisions
adversary_events.csv	Jamming and attack events
scenarios.csv	Scenario descriptions
________________________________________
Installation
Clone the repository:
git clone https://github.com/USERNAME/REPOSITORY.git
cd REPOSITORY
Install dependencies:
pip install -r requirements.txt
________________________________________
Running the Framework
Execute:
python main.py --dataset ./data --out ./outputs
or simply:
python main.py
and provide the dataset directory when prompted.
________________________________________
Generated Outputs
The framework automatically generates the following publication-quality figures:
Figure 1
CDF of Achievable Throughput under Dynamic Blockages
Figure 2
Latency Response during Traffic Surge (eMBB)
Figure 3
Latency Response during Traffic Surge (URLLC)
Figure 4
Throughput During Jamming Events
________________________________________
Artificial Intelligence Architecture
The implemented AI framework combines:
•	Graph Neural Networks (GNN)
•	Message Passing Neural Networks (MPNN)
•	Proximal Policy Optimization (PPO)
The PPO component is provided as a research skeleton and template for future training and experimentation. Full policy optimization procedures can be integrated depending on the target environment and dataset.
________________________________________
Research Applications
The framework can be utilized for:
•	6G Network Management
•	AI-Driven Resource Allocation
•	Dynamic Beam Management
•	Network Robustness Evaluation
•	Adversarial Resilience Assessment
•	Intelligent Wireless Communications Research
________________________________________
Citation
If you use this software in your research, please cite both the associated research article and the software repository.
A DOI will be assigned through Zenodo upon repository release.
________________________________________
License
This project is released under the MIT License.
________________________________________
Author
Dr. Mohammed A. M. Al-Shibly
Director, Computer Center
University of Fallujah, Iraq
ORCID: 0009-0002-6717-7364
________________________________________
Disclaimer
This repository is intended for academic and research purposes. The provided AI modules are designed as research-grade implementations and may require further optimization and validation before deployment in operational wireless communication systems.
