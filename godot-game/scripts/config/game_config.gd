extends Resource
class_name GameConfig

## 游戏全局配置 Resource

## 训练玩家选择枚举
## ALL: 所有玩家训练; PLAYER_0~3: 仅对应玩家训练（其余强制 IDLE）
enum TrainingPlayer { ALL = -1, PLAYER_0 = 0, PLAYER_1 = 1, PLAYER_2 = 2, PLAYER_3 = 3 }

@export_category("Reward")
@export var reward_logger_enabled: bool = true
@export var use_per_player_reward: bool = false   # true: 每个玩家用独立 reward_config_pX.tres; false: 统一用 reward_config.tres

@export_category("LiDAR")
@export var ray_count: int = 32            #射线检测数量

@export_category("Observation")
@export var use_observation_valid_mask: bool = false

@export_category("Training")
@export var training_player_id: TrainingPlayer = TrainingPlayer.ALL       # ALL: 所有玩家训练; PLAYER_0~3: 仅该玩家训练（其余强制 IDLE）
@export var reset_on_wall: bool = false        # 撞墙是否触发环境重置
@export var wall_reset_threshold: int = 2     # 连续撞墙多少8个物理帧才触发 reset（需 reset_on_wall=true）

@export_category("debug")
@export var enable_info_window:bool
