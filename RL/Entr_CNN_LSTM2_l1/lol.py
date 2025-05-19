from abm_env import ABMDeepHedgeEnv
import numpy as np

def sample_path(seed, n_live_steps=300):
    env, prices = ABMDeepHedgeEnv(seed=seed), []
    obs, _ = env.reset()
    start_call = env.sim.info.prices[env.mid_C][-1]   # цена в конце warm-up
    for _ in range(n_live_steps):
        action = env.action_space.sample()            # любой action (или PASS)
        obs, r, done, trunc, _ = env.step(action)
        prices.append(env.sim.info.prices[env.mid_C][-1])
    return start_call, prices

# --- одинаковый seed --------------------------------------------------
s1, p1 = sample_path(seed=123)
s2, p2 = sample_path(seed=123)
print(s1, p1[-1], '/n')
print( s2, p2[-1])
print("warm-up equal:", s1 == s2)      # → True
print("live different:", p1 != p2)     # → True (почти всегда)

# --- другой seed ------------------------------------------------------
s3, p3 = sample_path(seed=999)
print("warm-up still equal:", s1 == s3) # → True
print("live path differ:", p1 != p3)    # → True
