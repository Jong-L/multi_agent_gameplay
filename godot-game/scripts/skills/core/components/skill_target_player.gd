class_name SkillTargetPlayer
extends SkillComponent

## 目标筛选组件（玩家专用）
## 在施法者周围检测存活玩家，用于敌人 AI 的攻击目标获取
##
## 检测条件：
##   1. 属于 "player" 组
##   2. 存活状态（not is_dead）
##   3. 距离在 check_range 内
##   4. 在施法者前方扇形视野内（attack_fov）
##
## 与 SkillGetTarget 的区别：
##   - SkillGetTarget：物理检测（PhysicsShapeQuery），检测所有 Entity
##   - SkillTargetPlayer：遍历分组，专用于敌人检测玩家

@export var check_range: float = 25          ## 检测半径（像素）
@export var attack_fov: float = 100.0        ## 攻击视野角度（度）

func _activate(context: SkillContext) -> void:
	var caster = context.caster
	var targets: Array[Entity] = []
	
	for node in get_tree().get_nodes_in_group("player"):
		if not node is Player:
			continue
		if node.is_dead:
			continue
		
		var distance = caster.position.distance_to(node.position)
		if distance > check_range:
			continue
		
		## 扇形视野检查
		var face_dir = Vector2.RIGHT
		if caster.animated_sprite != null:
			face_dir = Vector2.LEFT if caster.animated_sprite.flip_h else Vector2.RIGHT
		
		var to_target = (node.position - caster.position).normalized()
		var fov = deg_to_rad(attack_fov)
		if to_target.dot(face_dir) > cos(fov / 2):
			targets.push_back(node)
	
	context.targets = targets  ## 将检测到的玩家目标填充到技能上下文中，供后续伤害组件使用

## 获取检测范围内最近的玩家
## @param context: 技能上下文
## @return: 最近的 Player 或 null
func get_nearest_player(context: SkillContext) -> Player:
	var caster = context.caster
	var nearest: Player = null
	var min_dist = INF
	
	for node in get_tree().get_nodes_in_group("player"):
		if not node is Player or node.is_dead:
			continue
		
		var distance = caster.position.distance_to(node.position)
		if distance > check_range:
			continue
		
		## 扇形视野检查
		var face_dir = Vector2.RIGHT
		if caster.animated_sprite != null:
			face_dir = Vector2.LEFT if caster.animated_sprite.flip_h else Vector2.RIGHT
		
		var to_target = (node.position - caster.position).normalized()
		var fov = deg_to_rad(attack_fov)
		if to_target.dot(face_dir) > cos(fov / 2):
			if distance < min_dist:
				min_dist = distance
				nearest = node
	
	return nearest  ## 在符合条件的玩家中找到距离最近的一个，用于单体攻击技能的优先目标选择

## 获取检测范围内所有玩家数量
## @param context: 技能上下文
## @return: 玩家数量
func get_player_count(context: SkillContext) -> int:
	var caster = context.caster
	var count = 0
	
	for node in get_tree().get_nodes_in_group("player"):
		if not node is Player or node.is_dead:
			continue
		
		var distance = caster.position.distance_to(node.position)
		if distance > check_range:
			continue
		
		## 扇形视野检查
		var face_dir = Vector2.RIGHT
		if caster.animated_sprite != null:
			face_dir = Vector2.LEFT if caster.animated_sprite.flip_h else Vector2.RIGHT
		
		var to_target = (node.position - caster.position).normalized()
		var fov = deg_to_rad(attack_fov)
		if to_target.dot(face_dir) > cos(fov / 2):
			count += 1
	
	return count  ## 统计视野范围内可攻击的玩家数量，用于范围技能的伤害分配或 AI 决策

## 检查是否有玩家在检测范围内
## @param context: 技能上下文
## @return: true 表示有玩家可攻击
func has_target_in_range(context: SkillContext) -> bool:
	return not get_player_count(context) == 0  ## 快速检查视野范围内是否存在可攻击的玩家，用于 AI 攻击决策或技能可用性判断

## 获取最远的玩家目标
## @param context: 技能上下文
## @return: 最远的 Player 或 null
func get_farthest_player(context: SkillContext) -> Player:
	var caster = context.caster
	var farthest: Player = null
	var max_dist = 0.0
	
	for node in get_tree().get_nodes_in_group("player"):
		if not node is Player or node.is_dead:
			continue
		
		var distance = caster.position.distance_to(node.position)
		if distance > check_range:
			continue
		
		## 扇形视野检查
		var face_dir = Vector2.RIGHT
		if caster.animated_sprite != null:
			face_dir = Vector2.LEFT if caster.animated_sprite.flip_h else Vector2.RIGHT
		
		var to_target = (node.position - caster.position).normalized()
		var fov = deg_to_rad(attack_fov)
		if to_target.dot(face_dir) > cos(fov / 2):
			if distance > max_dist:
				max_dist = distance
				farthest = node
	
	return farthest  ## 在视野范围内找到距离最远的玩家，用于某些特殊技能的优先目标选择（如狙击、追击等）

## 获取指定角度偏移的玩家目标
## @param context: 技能上下文
## @param angle_offset: 角度偏移（度）
## @return: 符合条件的 Player 或 null
func get_player_at_angle(context: SkillContext, angle_offset: float) -> Player:
	var caster = context.caster
	var base_dir = Vector2.RIGHT
	if caster.animated_sprite != null:
		base_dir = Vector2.LEFT if caster.animated_sprite.flip_h else Vector2.RIGHT
	
	var target_dir = base_dir.rotated(deg_to_rad(angle_offset))
	var best_player: Player = null
	var best_dot = -1.0
	
	for node in get_tree().get_nodes_in_group("player"):
		if not node is Player or node.is_dead:
			continue
		
		var distance = caster.position.distance_to(node.position)
		if distance > check_range:
			continue
		
		var to_target = (node.position - caster.position).normalized()
		var dot = to_target.dot(target_dir)
		if dot > best_dot:
			best_dot = dot
			best_player = node
	
	return best_player if best_dot > cos(deg_to_rad(attack_fov / 2)) else null  ## 在指定角度方向上寻找最接近的玩家，用于定向攻击或扇形区域的特定角度目标选择

## 获取检测范围内生命值最低的玩家
## @param context: 技能上下文
## @return: 生命值最低的 Player 或 null
func get_lowest_health_player(context: SkillContext) -> Player:
	var caster = context.caster
	var lowest: Player = null
	var min_health = INF
	
	for node in get_tree().get_nodes_in_group("player"):
		if not node is Player or node.is_dead:
			continue
		
		var distance = caster.position.distance_to(node.position)
		if distance > check_range:
			continue
		
		## 扇形视野检查
		var face_dir = Vector2.RIGHT
		if caster.animated_sprite != null:
			face_dir = Vector2.LEFT if caster.animated_sprite.flip_h else Vector2.RIGHT
		
		var to_target = (node.position - caster.position).normalized()
		var fov = deg_to_rad(attack_fov)
		if to_target.dot(face_dir) > cos(fov / 2):
			if node.current_health < min_health:
				min_health = node.current_health
				lowest = node
	
	return lowest  ## 在视野范围内找到生命值最低的玩家，用于 AI 的优先击杀策略或治疗技能的优先目标选择

## 获取检测范围内生命值百分比最低的玩家
## @param context: 技能上下文
## @return: 生命值百分比最低的 Player 或 null
func get_lowest_health_percent_player(context: SkillContext) -> Player:
	var caster = context.caster
	var lowest: Player = null
	var min_percent = 1.0
	
	for node in get_tree().get_nodes_in_group("player"):
		if not node is Player or node.is_dead:
			continue
		
		var distance = caster.position.distance_to(node.position)
		if distance > check_range:
			continue
		
		## 扇形视野检查
		var face_dir = Vector2.RIGHT
		if caster.animated_sprite != null:
			face_dir = Vector2.LEFT if caster.animated_sprite.flip_h else Vector2.RIGHT
		
		var to_target = (node.position - caster.position).normalized()
		var fov = deg_to_rad(attack_fov)
		if to_target.dot(face_dir) > cos(fov / 2):
			var percent = node.current_health / node.max_health
			if percent < min_percent:
				min_percent = percent
				lowest = node
	
	return lowest  ## 在视野范围内找到生命值百分比最低的玩家，相比绝对生命值更能反映危急程度，用于智能目标选择

## 按距离排序获取玩家列表
## @param context: 技能上下文
## @param ascending: true=由近到远，false=由远到近
## @return: 排序后的 Player 数组
func get_players_sorted_by_distance(context: SkillContext, ascending: bool = true) -> Array[Player]:
	var caster = context.caster
	var players_with_dist: Array[Dictionary] = []
	
	for node in get_tree().get_nodes_in_group("player"):
		if not node is Player or node.is_dead:
			continue
		
		var distance = caster.position.distance_to(node.position)
		if distance > check_range:
			continue
		
		## 扇形视野检查
		var face_dir = Vector2.RIGHT
		if caster.animated_sprite != null:
			face_dir = Vector2.LEFT if caster.animated_sprite.flip_h else Vector2.RIGHT
		
		var to_target = (node.position - caster.position).normalized()
		var fov = deg_to_rad(attack_fov)
		if to_target.dot(face_dir) > cos(fov / 2):
			players_with_dist.append({"player": node, "distance": distance})
	
	## 排序
	players_with_dist.sort_custom(func(a, b): 
		if ascending:
			return a.distance < b.distance
		else:
			return a.distance > b.distance
	)
	
	var result: Array[Player] = []
	for item in players_with_dist:
		result.append(item.player)
	
	return result  ## 将视野范围内的玩家按距离排序返回，用于多段攻击、连锁技能或 AI 的优先级排序

## 获取检测范围内特定 ID 的玩家
## @param context: 技能上下文
## @param player_id: 目标玩家 ID
## @return: 指定 ID 的 Player 或 null
func get_player_by_id(context: SkillContext, player_id: int) -> Player:
	var caster = context.caster
	
	for node in get_tree().get_nodes_in_group("player"):
		if not node is Player or node.is_dead:
			continue
		
		if node.player_id != player_id:
			continue
		
		var distance = caster.position.distance_to(node.position)
		if distance > check_range:
			continue
		
		## 扇形视野检查
		var face_dir = Vector2.RIGHT
		if caster.animated_sprite != null:
			face_dir = Vector2.LEFT if caster.animated_sprite.flip_h else Vector2.RIGHT
		
		var to_target = (node.position - caster.position).normalized()
		var fov = deg_to_rad(attack_fov)
		if to_target.dot(face_dir) > cos(fov / 2):
			return node
	
	return null  ## 在视野范围内查找指定 ID 的玩家，用于锁定特定目标或队友技能的目标选择

## 计算到目标的预期命中时间
## @param context: 技能上下文
## @param projectile_speed: 投射物速度（像素/秒）
## @return: 预期时间（秒），-1 表示无法命中
func get_time_to_hit(context: SkillContext, projectile_speed: float) -> float:
	var nearest = get_nearest_player(context)
	if nearest == null or projectile_speed <= 0:
		return -1.0
	
	var caster = context.caster
	var distance = caster.position.distance_to(nearest.position)
	return distance / projectile_speed  ## 根据投射物速度计算到达最近目标所需的时间，用于预判射击或延迟效果的时机计算

## 检查目标是否在移动
## @param context: 技能上下文
## @return: true 表示有目标在移动
func has_moving_target(context: SkillContext) -> bool:
	for node in get_tree().get_nodes_in_group("player"):
		if not node is Player or node.is_dead:
			continue
		
		var player = node as Player
		if not player.is_moving:
			continue
		
		var caster = context.caster
		var distance = caster.position.distance_to(player.position)
		if distance > check_range:
			continue
		
		## 扇形视野检查
		var face_dir = Vector2.RIGHT
		if caster.animated_sprite != null:
			face_dir = Vector2.LEFT if caster.animated_sprite.flip_h else Vector2.RIGHT
		
		var to_target = (player.position - caster.position).normalized()
		var fov = deg_to_rad(attack_fov)
		if to_target.dot(face_dir) > cos(fov / 2):
			return true
	
	return false  ## 检查视野范围内是否有正在移动的玩家，用于预判技能或追踪技能的触发条件判断

## 获取目标的移动方向预测
## @param context: 技能上下文
## @param target: 目标玩家
## @return: 预测位置
func predict_target_position(context: SkillContext, target: Player, time_ahead: float) -> Vector2:
	if target == null or not target.is_moving:
		return target.global_position if target else Vector2.ZERO
	
	## 基于当前速度向量预测未来位置
	var velocity = Vector2.ZERO
	match target.pending_action:
		Player.Action.MOVE_UP:
			velocity = Vector2.UP * target.run_speed
		Player.Action.MOVE_DOWN:
			velocity = Vector2.DOWN * target.run_speed
		Player.Action.MOVE_LEFT:
			velocity = Vector2.LEFT * target.run_speed
		Player.Action.MOVE_RIGHT:
			velocity = Vector2.RIGHT * target.run_speed
		_:
			return target.global_position
	
	return target.global_position + velocity * time_ahead  ## 根据目标的当前移动状态预测其未来位置，用于投射物的提前量计算或 AI 的拦截路径规划

## 获取最佳预判攻击位置
## @param context: 技能上下文
## @param projectile_speed: 投射物速度
## @return: 最佳攻击位置（世界坐标）
func get_leading_shot_position(context: SkillContext, projectile_speed: float) -> Vector2:
	var nearest = get_nearest_player(context)
	if nearest == null or projectile_speed <= 0:
		return Vector2.ZERO
	
	var time_to_hit = get_time_to_hit(context, projectile_speed)
	if time_to_hit < 0:
		return nearest.global_position
	
	return predict_target_position(context, nearest, time_to_hit)  ## 计算考虑目标移动提前量的最佳攻击位置，使投射物能够命中移动中的目标
