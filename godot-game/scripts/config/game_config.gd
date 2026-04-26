extends Resource
class_name GameConfig

## 游戏全局配置 Resource

@export_category("Reward")
@export var reward_logger_enabled: bool = true

@export_category("LiDAR")
@export var ray_count: int = 32            #射线检测数量（map_state 维度）
@export var debug_draw_rays: bool = true       #射线可视化调试开关
