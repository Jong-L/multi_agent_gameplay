extends Resource
class_name RewardConfig

## 奖励配置文件资源

# 枚举定义
enum WallPotentialMode { NONE, LINEAR, INVERSE, COLLISION ,EXP}
enum WallPotentialCalculateMode { MIN, WEIGHTED_AVERAGE, AVERAGE }
enum BallPotentialMode { NEAREST, ALL }
enum StarveFunc{LNEAR ,QUADRATIC,SQRT}

# 奖励常量
@export var collect_ball_A: float = 10.0
@export var collect_ball_B: float = 15.0
@export var bear_damage: float = -10.0
@export var cause_damage_to_enemy: float = 10.0
@export var cause_damage_to_player: float = 15.0
@export var kill_enemy: float = 30.0
@export var kill_player: float = 45.0
@export var run: float = 0.0
@export var attack: float = -0.05
@export var died: float = -20.0

# 饥饿机制
@export var starve_time: float = 1500.0
@export var max_starve_duration: float = 10.0
@export var starve_reward_decrease: float = 0.01
@export var starve_more_func: StarveFunc=StarveFunc.QUADRATIC

# 塑形奖励
@export var ball_potential_scale: float = 1.0
@export var ball_potential_mode: BallPotentialMode = BallPotentialMode.NEAREST
@export var center_reward_scale: float = 0.0

# 撞墙惩罚
@export var wall_collision_penalty: float = 0.5
@export var wall_potential_mode: WallPotentialMode = WallPotentialMode.NONE
@export var wall_potential_calculate_mode: WallPotentialCalculateMode = WallPotentialCalculateMode.MIN
