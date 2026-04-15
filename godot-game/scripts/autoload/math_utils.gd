extends Node

## 数学/随机工具函数单例
## 提供项目中通用的随机位置生成、矩形计算等方法


## 在矩形内生成随机位置（留边距避免贴边）
## @param rect 目标矩形区域
## @param margin 距矩形边缘的最小距离
## @return 矩形内的随机 Vector2 坐标
static func random_pos_in_rect(rect: Rect2, margin: float = 0.0) -> Vector2:
	var x := randf_range(rect.position.x + margin, rect.end.x - margin)
	var y := randf_range(rect.position.y + margin, rect.end.y - margin)
	return Vector2(x, y)


## 在给定中心点周围生成随机偏移
## @param radius 最大偏移半径
## @param min_ratio 最小距离比例（0~1），避免生成在正中心
## @return 以原点为中心的随机偏移向量
static func random_offset(radius: float, min_ratio: float = 0.3) -> Vector2:
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
static func quadrant_rect(bounds: Rect2, extent: Vector2, direction_x: int, direction_y: int) -> Rect2:
	var origin := Vector2(
		bounds.position.x if direction_x < 0 else bounds.end.x - extent.x,
		bounds.position.y if direction_y < 0 else bounds.end.y - extent.y
	)
	return Rect2(origin, extent)
