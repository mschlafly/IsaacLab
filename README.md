
## Usage

On windows, after setting up an environment, activate the environment from the command prompt with `call env_isaaclab\Scripts\activate.bat`

To run my environment with the skrl package for RL using PPO as a default: `python scripts\reinforcement_learning\skrl\train.py --task "Peg-in-hole-Direct-v0"`. Use `--num_env` to set the number of simultaneous environments and `--headless` to run without visualization. 


## License

The Isaac Lab framework is released under [BSD-3 License](LICENSE). The `isaaclab_mimic` extension and its corresponding standalone scripts are released under [Apache 2.0](LICENSE-mimic). The license files of its dependencies and assets are present in the [`docs/licenses`](docs/licenses) directory.

mschlafly's code is under the MIT license.

## Acknowledgement

This repo uses Isaac Lab initiated from the [Orbit](https://isaac-orbit.github.io/) framework show in the below citation:

```
@article{mittal2023orbit,
   author={Mittal, Mayank and Yu, Calvin and Yu, Qinxi and Liu, Jingzhou and Rudin, Nikita and Hoeller, David and Yuan, Jia Lin and Singh, Ritvik and Guo, Yunrong and Mazhar, Hammad and Mandlekar, Ajay and Babich, Buck and State, Gavriel and Hutter, Marco and Garg, Animesh},
   journal={IEEE Robotics and Automation Letters},
   title={Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments},
   year={2023},
   volume={8},
   number={6},
   pages={3740-3747},
   doi={10.1109/LRA.2023.3270034}
}
```
