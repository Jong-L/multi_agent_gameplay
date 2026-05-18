"""
Compatibility entrypoint for the GRU-fast PPO implementation.

The fast GRU sequence evaluation is now integrated directly in custom_ppo.py.
"""

from custom_ppo import *

if __name__ == "__main__":
    main()
