extends Node

## 数学/随机工具函数单例
## 提供项目中通用的随机位置生成、矩形计算等方法


## 在矩形内生成随机位置（留边距避免贴边）
## @param rect 目标矩形区域
## @param margin 距矩形边缘的最小距离
## @return 矩形内的随机 Vector2 坐标
func random_pos_in_rect(rect: Rect2, margin: float = 0.0) -> Vector2:
	var x := randf_range(rect.position.x + margin, rect.end.x - margin)
	var y := randf_range(rect.position.y + margin, rect.end.y - margin)
	return Vector2(x, y)


## 在给定中心点周围生成随机偏移
## @param radius 最大偏移半径
## @param min_ratio 最小距离比例（0~1），避免生成在正中心
## @return 以原点为中心的随机偏移向量
func random_offset(radius: float, min_ratio: float = 0.3) -> Vector2:
	var angle := randf() * TAU
	var r := randf_range(radius * min_ratio, radius)
	return Vector2(cos(angle), sin(angle)) * r


## 根据竞技场边界和象限方向，生成从对应角向内延伸的子矩形
## 例如：左上角 → 子矩形从竞技场左上角向右下延伸 extent 尺寸
## @param bounds 竞技场完整边界 Rect2
## @param extent 子矩形尺寸（从角向内延伸的大小）
## @param direction_x 方向：-1=左半区（左上/左下角）, 1=右半区（右上/右下角）
## @param direction_y 方向：-1=上半区（左上/右上角）, 1=下半区（左下/右下角）
## @return 从对应角向内延伸的 Rect2
func quadrant_rect(bounds: Rect2, extent: Vector2, direction_x: int, direction_y: int) -> Rect2:
	var origin := Vector2(
		bounds.position.x if direction_x < 0 else bounds.end.x - extent.x,
		bounds.position.y if direction_y < 0 else bounds.end.y - extent.y
	)
	return Rect2(origin, extent)


## 计算饥饿惩罚的衰减倍率
## @param starve_duration 已饥饿的持续时间（秒）
## @param func_type 增长函数类型："linear"(线性), "quadratic"(二次), "sqrt"(平方根)
## @return 衰减倍率（>= 1.0）
func starve_rate_multiplier(starve_duration: float, func_type: String = "linear") -> float:
	match func_type:
		"linear":
			# 线性增长
			return 0.1 + starve_duration
		"quadratic":
			# 二次增长：惩罚加速变重
			return starve_duration * starve_duration * 0.3
		"sqrt":
			# 平方根增长：初期快，后期慢
			return 1.0 + sqrt(starve_duration)
		_:
			# 未知类型默认线性
			return 0.1 + starve_duration

# 从 TileMapLayer 计算世界坐标矩形
func _tilemap_to_world_rect(layer: TileMapLayer) -> Rect2:
	var used := layer.get_used_rect()
	var cell_size := layer.tile_set.tile_size
	var s := layer.scale
	var p := layer.position#图块偏移位置
	return Rect2(
		p.x + used.position.x * cell_size.x * s.x,
		p.y + used.position.y * cell_size.y * s.y,
		used.size.x * cell_size.x * s.x,
		used.size.y * cell_size.y * s.y
	)

# 从 TileMapLayer 提取所有已使用 tile 的世界坐标
func _tilemap_to_world_positions(layer: TileMapLayer) -> Array[Vector2]:
	var result: Array[Vector2] = []
	var used_cells := layer.get_used_cells()
	var cell_size := layer.tile_set.tile_size
	var s := layer.scale
	var p := layer.position
	for cell in used_cells:
		var world_pos := Vector2(
			p.x + cell.x * cell_size.x * s.x + cell_size.x * s.x * 0.5,
			p.y + cell.y * cell_size.y * s.y + cell_size.y * s.y * 0.5
		)
		result.append(world_pos)
	return result
