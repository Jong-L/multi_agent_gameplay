extends Node2D
class_name Pathfinding

## 路径寻找组件（分离行为避障）
## 使用 Separation（分离）行为让敌人避免相互重叠
##
## 算法原理：
##   1. 检测周围邻居（其他 Enemy）
##   2. 计算分离向量：Σ(自身位置 - 邻居位置).normalized() / 距离
##   3. 分离向量 × 分离力 + 目标方向 = 最终移动方向
##
## 参考：Craig Reynolds 的 Steering Behaviors

@export var neighbour_check_radius: float = 10    ## 邻居检测半径（像素）
@export var separation_force: float = 300         ## 分离力强度（值越大避障越激进）

## 缓存的物理查询对象
var _cached_shape: CircleShape2D
var _cached_query: PhysicsShapeQueryParameters2D

func _ready() -> void:
	_cached_shape = CircleShape2D.new()
	_cached_query = PhysicsShapeQueryParameters2D.new()
	_cached_query.shape = _cached_shape
	_cached_query.collide_with_areas = true
	_cached_query.collide_with_bodies = false

## 计算带避障的移动方向
## @param target_position: 目标世界坐标
## @return: 移动方向向量（未归一化，保留大小信息）
func find_path(target_position: Vector2) -> Vector2:
	_cached_shape.radius = neighbour_check_radius
	_cached_query.transform.origin = global_position
	
	var space_state = get_world_2d().direct_space_state
	var results = space_state.intersect_shape(_cached_query)
	
	var neighbours: Array[Enemy] = []
	if results.size() > 0:
		for result in results:
			var collider = result.collider
			var parent = collider.get_parent()
			## 只考虑其他 Enemy（排除自己）
			if parent is Enemy and parent != self.get_parent():
				neighbours.push_back(parent)
	
	var separation_direction = _calculate_separation(neighbours)
	## 分离方向 × 分离力 + 目标方向
	return (separation_direction * separation_force) + (target_position - global_position)

## 计算分离向量
## 距离越近的邻居贡献越大（1/distance 加权）
## @param neighbours: 邻居数组
func _calculate_separation(neighbours: Array[Enemy]) -> Vector2:
	var separation_vector = Vector2.ZERO
	
	for neighbour in neighbours:
		var to_me = global_position - neighbour.global_position
		var distance = to_me.length()
		
		if distance > 0:
			separation_vector += to_me.normalized() / distance
	
	return separation_vector

## 设置检测半径
## @param radius: 新半径
func set_check_radius(radius: float) -> void:
	neighbour_check_radius = max(1.0, radius)

## 设置分离力
## @param force: 新分离力
func set_separation_force(force: float) -> void:
	separation_force = max(0.0, force)

## 获取当前检测半径
func get_check_radius() -> float:
	return neighbour_check_radius

## 获取当前分离力
func get_separation_force() -> float:
	return separation_force
