import numpy as np
from pathlib import Path
from utils_plot import read_tb_events_auto, read_csv_training_auto, align_x, plot_metric

case =  "cigre_mv_mg"
base_dir = str(Path.cwd().parent._str)

# Choose whether to rescale the x-axis to 50k env steps
TOTAL_ENV_STEPS = 50_000
NUM_LOG_STEPS = 10000

dreamer = read_tb_events_auto(case, "dreamer", base_dir, max_seeds=3, max_steps=TOTAL_ENV_STEPS, num_points=NUM_LOG_STEPS)
ppo     = read_tb_events_auto(case, "ppo",     base_dir, max_seeds=3, max_steps=TOTAL_ENV_STEPS, num_points=NUM_LOG_STEPS)
sac     = read_tb_events_auto(case, "sac",     base_dir, max_seeds=3, max_steps=TOTAL_ENV_STEPS, num_points=NUM_LOG_STEPS)
x_steps = np.linspace(1, TOTAL_ENV_STEPS, NUM_LOG_STEPS)

# Optional: smoothing window (set to 1 to disable)
SMOOTH = 51  # moving-average window size; must be odd. Try 1, 21, 51, etc.

algs = {"DreamerV3": dreamer, "PPO": ppo, "SAC": sac}

plot_metric(case, "return", algs, x_steps, smooth=SMOOTH, fname="./figures/{}_return.png".format(case))
plot_metric(case, "safety", algs, x_steps, smooth=SMOOTH, fname="./figures/{}_safety.png".format(case))