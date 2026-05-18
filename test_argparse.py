import sys
sys.path.insert(0, 'Python/training')
from ppo_resume_trainer import _build_arg_parser
parser = _build_arg_parser()
args = parser.parse_args(['--total-timesteps', '128', '--num-steps', '4', '--save-every-n-updates', '9999'])
print('total_timesteps:', args.total_timesteps)
print('num_steps:', args.num_steps)
print('save_every_n_updates:', args.save_every_n_updates)
