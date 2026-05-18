import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--total_timesteps", type=int, default=100)
# Python 3.13 中 argparse 默认不再把 _ 和 - 互换
print("trying --total-timesteps:")
try:
    args = parser.parse_args(["--total-timesteps", "128"])
    print("OK:", args.total_timesteps)
except SystemExit as e:
    print("FAILED:", e)

print("\ntrying --total_timesteps:")
try:
    args = parser.parse_args(["--total_timesteps", "128"])
    print("OK:", args.total_timesteps)
except SystemExit as e:
    print("FAILED:", e)
