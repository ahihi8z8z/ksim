from gymnasium.envs.registration import register

register(
    id="KsimEnv-V0",
    entry_point="ksim_env.envs:KsimEnv",
    kwargs={}
)
