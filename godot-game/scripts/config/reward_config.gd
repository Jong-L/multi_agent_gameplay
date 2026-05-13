extends Resource
class_name RewardConfig

## 奖励配置文件资源

# 枚举定义
enum WallPotentialMode { NONE, LINEAR, INVERSE, COLLISION ,EXP}
enum WallPotentialCalculateMode { MIN, WEIGHTED_AVERAGE, AVERAGE }
enum BallPotentialMode { NEAREST, ALL }
enum BallPotentialFunc { LINEAR, EXPONENTIAL, INVERSE, DISTANCE_REWARD }
enum StarveFunc{LNEAR ,QUADRATIC,SQRT}

# 奖励常量
@export var collect_ball_A: float 
@export var collect_ball_B: float 
@export var bear_damage: float 
@export var cause_damage_to_enemy: float 
@export var cause_damage_to_player: float
@export var kill_enemy: float 
@export var kill_player: float 
@export var run: float 
@export var idle:float 
@export var attack: float 
@export var died: float 

# 饥饿机制
@export var starve_time: float 
@export var max_starve_duration: float 
@export var starve_reward_decrease: float 
@export var starve_more_func: StarveFunc=StarveFunc.QUADRATIC

# 塑形奖励
@export var ball_potential_scale: float = 1.0
@export var ball_potential_mode: BallPotentialMode = BallPotentialMode.NEAREST
@export var ball_potential_func: BallPotentialFunc = BallPotentialFunc.LINEAR
@export var distance_reward_scale: float = 0.1
@export var center_reward: float

# 撞墙惩罚
@export var wall_collision_penalty: float = 0.5
@export var wall_potential_mode: WallPotentialMode = WallPotentialMode.NONE
@export var wall_potential_calculate_mode: WallPotentialCalculateMode = WallPotentialCalculateMode.MIN
