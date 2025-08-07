# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
UR Peg-in-hole environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# When I change the id, the environment could not be found

gym.register(
    id="Peg-in-hole-Direct-v0",
    entry_point=f"{__name__}.peg-in-hole_env:UREnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.peg-in-hole_env:UREnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
