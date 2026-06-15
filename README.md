# PowerGrid: World-Model-Based Deep RL for Microgrid Energy Management

This repository contains the official implementation accompanying the paper:

**"World-Model-Based Deep Reinforcement Learning for Real-Time Microgrid Energy Management"**
Kajal Pokharel, Hepeng Li, and Haibo He

The proposed framework integrates a Dreamer-style world model with continuous control policies for real-time microgrid energy management under AC power-flow and operational constraints. The agent jointly learns a latent dynamics model of the microgrid and a control policy, using imagination-based multi-step rollouts to improve sample efficiency and constraint satisfaction.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/hepengli/powergrid.git
   cd powergrid
   ```

2. (Optional) Create and activate the provided conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate powergrid
   ```

3. Install the package in editable mode:
   ```bash
   pip install -e .
   ```

4. Install Ray RLlib:
   ```bash
   pip install "ray[rllib]"
   ```

## Repository Structure

```
powergrid/
├── data/          # Real-world CAISO datasets used for training and testing
├── microgrid/     # Microgrid environments (e.g., ieee13, cigre_mv, ieee34)
├── powergrid/     # Core source code: world model, RSSM, actor-critic, environments
├── environment.yml
├── setup.py
└── README.md
```

## Usage

To train the proposed DreamerV3-based agent on a specific microgrid:

```bash
cd microgrid/<microgrid_name>
python dreamer_train.py
```

For example, to train on the IEEE 13-bus microgrid:

```bash
cd microgrid/ieee13
python dreamer_train.py
```

Replace `ieee13` with `cigre_mv` or `ieee34` to train on the CIGRE-MV or IEEE 34-bus microgrids, respectively.

## Citation

If you find this code useful in your research, please consider citing our paper:

```bibtex
@article{pokharel2025worldmodel,
  title={World-Model-Based Deep Reinforcement Learning for Real-Time Microgrid Energy Management},
  author={Pokharel, Kajal and Li, Hepeng and He, Haibo},
  journal={IEEE Transactions on Smart Grid},
  year={2025}
}
```

## License

This project is licensed under the terms described in [LICENSE.txt](LICENSE.txt).

## Contact

For questions regarding this repository, please contact Hepeng Li (hepeng.li@maine.edu).
