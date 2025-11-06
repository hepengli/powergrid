import numpy as np
from utils_plot import read_tb_events_auto, plot_metric

case =  "cigre_mv_mg"
base_dir = "/Users/hepeng.li/ray_results"

dreamer = read_tb_events_auto(case, "dreamer", base_dir, max_seeds=3)
ppo     = read_tb_events_auto(case, "ppo",     base_dir, max_seeds=3)
sac     = read_tb_events_auto(case, "sac",     base_dir, max_seeds=3)

# Choose whether to rescale the x-axis to 50k env steps
TOTAL_ENV_STEPS = 50_000
NUM_LOG_STEPS = dreamer['return'].shape[1]  # 10000
x_steps = np.linspace(1, TOTAL_ENV_STEPS, NUM_LOG_STEPS)

# Optional: smoothing window (set to 1 to disable)
SMOOTH = 51  # moving-average window size; must be odd. Try 1, 21, 51, etc.

algs = {"DreamerV3": dreamer, "PPO": ppo, "SAC": sac}

plot_metric("return", algs, x_steps, smooth=SMOOTH, fname="./figures/{}_return.png".format(case))
plot_metric("safety", algs, x_steps, smooth=SMOOTH, fname="./figures/{}_safety.png".format(case))