extends Resource
class_name GameConfig

## 游戏全局配置 Resource

@export_category("Reward")
@export var reward_logger_enabled: bool = true
@export var use_per_player_reward: bool = false   # true: 每个玩家用独立 reward_config_pX.tres; false: 统一用 reward_config.tres

@export_category("LiDAR")
@export var ray_count: int = 32            #射线检测数量（map_state 维度）
