class_name SkillContext
extends RefCounted

## 技能执行上下文
## 在 Skill.activate() 时创建，贯穿整个技能执行流程
## 用于在 SkillComponent 之间传递数据
##
## 数据流：
##   1. SkillGetTarget 检测目标 → 填充 targets
##   2. SkillDealDamage 读取 targets → 造成伤害
##   3. SkillAnimationRunner 读取 caster → 播放动画

var caster: Entity                      #施法者实体
var skill: Skill                        #当前执行的技能
var targets: Array[Variant] = []        #目标列表

func _init(caster: Entity, skill: Skill) -> void:
	self.caster = caster
	self.skill = skill

## 获取目标位置
## 支持类型：Entity（返回 global_position）或 Vector2
## @param idx: 目标索引
## @return: 世界坐标位置，无效时返回 ZERO
func get_target_position(idx: int) -> Vector2:
	if idx < 0 or idx >= targets.size():
		return Vector2.ZERO
	
	var target = targets[idx]
	if target is Entity:
		return target.global_position
	elif target is Vector2:
		return target
	return Vector2.ZERO

## 添加目标
## @param target: Entity 或 Vector2
func add_target(target: Variant) -> void:
	targets.append(target)

## 清空目标列表
func clear_targets() -> void:
	targets.clear()

## 获取目标数量
func get_target_count() -> int:
	return targets.size()  ## 返回当前目标列表中的目标数量，用于检查是否有有效目标

## 检查是否有有效目标
func has_targets() -> bool:
	return not targets.is_empty()  ## 如果目标列表不为空则返回 true，用于在执行伤害等操作前确认目标存在

## 获取第一个目标
## @return: 第一个目标或 null
func get_first_target() -> Variant:
	if targets.is_empty():
		return null
	return targets[0]  ## 返回目标列表中的第一个元素，如果不存在则返回 null，用于快速获取主要目标

## 获取所有目标
## @return: 目标数组副本
func get_all_targets() -> Array[Variant]:
	return targets.duplicate()  ## 返回目标列表的副本，防止外部修改原始数据，用于需要遍历所有目标的场景

## 获取施法者位置
## @return: 施法者世界坐标
func get_caster_position() -> Vector2:
	if caster != null:
		return caster.global_position
	return Vector2.ZERO  ## 如果施法者存在则返回其全局位置，否则返回零向量，用于计算距离和方向

## 获取施法者朝向
## @return: 施法者朝向向量（基于 flip_h）
func get_caster_facing() -> Vector2:
	if caster == null or caster.animated_sprite == null:
		return Vector2.RIGHT
	return Vector2.LEFT if caster.animated_sprite.flip_h else Vector2.RIGHT  ## 根据施法者的精灵翻转状态返回朝向向量，flip_h 为 true 时朝左，否则朝右

## 计算到目标的距离
## @param idx: 目标索引
## @return: 距离（像素）
func distance_to_target(idx: int) -> float:
	var target_pos = get_target_position(idx)
	var caster_pos = get_caster_position()
	return caster_pos.distance_to(target_pos)  ## 计算施法者到指定目标位置的距离，用于判断是否在攻击范围内

## 计算到目标的方向
## @param idx: 目标索引
## @return: 归一化方向向量
func direction_to_target(idx: int) -> Vector2:
	var target_pos = get_target_position(idx)
	var caster_pos = get_caster_position()
	return (target_pos - caster_pos).normalized()  ## 计算从施法者指向目标位置的归一化方向向量，用于击退、位移等效果的方向计算

## 检查目标是否在扇形视野内
## @param idx: 目标索引
## @param fov_degrees: 视野角度（度）
## @return: true 表示在视野内
func is_target_in_fov(idx: int, fov_degrees: float) -> bool:
	var to_target = direction_to_target(idx)
	var facing = get_caster_facing()
	var fov = deg_to_rad(fov_degrees)
	return to_target.dot(facing) > cos(fov / 2)  ## 使用点积判断目标是否在施法者前方扇形区域内，用于近战攻击的目标筛选

## 创建特效节点
## @param scene: 特效场景
## @param position: 生成位置
## @param parent: 父节点（默认场景根节点）
## @return: 创建的节点
func spawn_effect(scene: PackedScene, position: Vector2, parent: Node = null) -> Node:
	if scene == null:
		return null
	
	var instance = scene.instantiate()
	if parent == null:
		parent = caster.get_tree().get_root()
	parent.add_child(instance)
	instance.global_position = position
	return instance  ## 实例化特效场景并添加到指定父节点，设置位置后返回实例，用于生成攻击特效、命中效果等

## 获取技能冷却进度
## @return: 0.0~1.0，0 表示冷却完毕
func get_cooldown_progress() -> float:
	if skill == null or skill.cooldown <= 0:
		return 0.0
	return skill.current_cooldown / skill.cooldown  ## 返回当前冷却时间与总冷却时间的比值，用于 UI 显示冷却进度

## 检查技能是否处于冷却中
func is_on_cooldown() -> bool:
	if skill == null:
		return false
	return skill.current_cooldown > 0  ## 检查技能当前是否有剩余冷却时间，用于防止重复触发或显示冷却状态

## 获取施法者属性
## @param property: 属性名
## @param default: 默认值
## @return: 属性值
func get_caster_property(property: String, default: Variant = null) -> Variant:
	if caster == null:
		return default
	return caster.get(property) if caster.get(property) != null else default  ## 安全地获取施法者的属性值，如果施法者不存在或属性不存在则返回默认值

## 设置施法者属性
## @param property: 属性名
## @param value: 属性值
func set_caster_property(property: String, value: Variant) -> void:
	if caster != null:
		caster.set(property, value)  ## 安全地设置施法者的属性值，在施法者存在时才执行设置操作

## 应用伤害到目标
## @param idx: 目标索引
## @param damage: 伤害值
## @return: true 表示成功应用
func apply_damage_to_target(idx: int, damage: float) -> bool:
	if idx < 0 or idx >= targets.size():
		return false
	
	var target = targets[idx]
	if target is Entity:
		target.bear_damage(damage, caster)
		return true
	return false  ## 对指定索引的目标应用伤害，如果目标是 Entity 类型则调用 bear_damage 方法（传入施法者），返回是否成功应用

## 应用伤害到所有目标
## @param damage: 伤害值
## @return: 受到伤害的目标数量
func apply_damage_to_all(damage: float) -> int:
	var count = 0
	for target in targets:
		if target is Entity:
			target.bear_damage(damage, caster)
			count += 1
	return count  ## 遍历所有目标并对 Entity 类型的目标应用伤害（传入施法者），返回实际受到伤害的目标数量，用于范围攻击技能

## 添加外部推力到施法者（击退效果）
## @param direction: 方向向量
## @param speed: 速度大小
## @param decay: 衰减速率
func apply_pushback_to_caster(direction: Vector2, speed: float, decay: float) -> void:
	if caster != null:
		caster.external_velocity = direction.normalized() * speed
		caster.external_velocity_decay = decay  ## 给施法者添加外部推力和衰减率，用于技能释放时的自身位移效果如后坐力

## 播放施法者动画
## @param anim_name: 动画名
## @param high_priority: 是否高优先级
func play_caster_animation(anim_name: String, high_priority: bool = false) -> void:
	if caster != null:
		caster.play_animation(AnimationWrapper.new(anim_name, high_priority))  ## 让施法者播放指定动画，可设置高优先级以打断其他动画

## 创建定时器回调
## @param delay: 延迟时间（秒）
## @param callback: 回调函数
func delay_call(delay: float, callback: Callable) -> void:
	if caster != null:
		await caster.get_tree().create_timer(delay).timeout
		callback.call()  ## 在指定延迟后执行回调函数，使用 await 实现异步延迟，用于技能的多段效果或延迟伤害

## 获取场景树
## @return: SceneTree 或 null
func get_tree() -> SceneTree:
	if caster != null:
		return caster.get_tree()
	return null  ## 返回施法者所在的场景树，用于访问全局节点或创建定时器

## 获取所有玩家
## @return: 玩家数组
func get_all_players() -> Array[Player]:
	var tree = get_tree()
	if tree == null:
		return []
	
	var players: Array[Player] = []
	for node in tree.get_nodes_in_group("player"):
		if node is Player:
			players.append(node)
	return players  ## 从场景树中获取所有属于 "player" 组的 Player 节点，用于范围技能或敌我识别

## 获取所有敌人
## @return: 敌人数组
func get_all_enemies() -> Array[Enemy]:
	var tree = get_tree()
	if tree == null:
		return []
	
	var enemies: Array[Enemy] = []
	for node in tree.get_nodes_in_group("enemy"):
		if node is Enemy:
			enemies.append(node)
	return enemies  ## 从场景树中获取所有属于 "enemy" 组的 Enemy 节点，用于敌我识别或范围效果

## 在范围内查找所有实体
## @param center: 中心位置
## @param radius: 半径
## @param include_dead: 是否包含死亡实体
## @return: 实体数组
func find_entities_in_range(center: Vector2, radius: float, include_dead: bool = false) -> Array[Entity]:
	var tree = get_tree()
	if tree == null:
		return []
	
	var entities: Array[Entity] = []
	var all_entities: Array[Entity] = []
	
	# 收集所有实体
	for node in tree.get_nodes_in_group("player"):
		if node is Entity:
			all_entities.append(node)
	for node in tree.get_nodes_in_group("enemy"):
		if node is Entity:
			all_entities.append(node)
	
	# 筛选在范围内的
	for entity in all_entities:
		if not include_dead and entity.is_dead:
			continue
		if entity.global_position.distance_to(center) <= radius:
			entities.append(entity)
	
	return entities  ## 在指定范围内查找所有实体（玩家和敌人），可选择是否包含死亡实体，用于范围技能的目标检测

## 获取最近的玩家
## @return: 最近的 Player 或 null
func get_nearest_player() -> Player:
	var players = get_all_players()
	if players.is_empty():
		return null
	
	var nearest: Player = null
	var min_dist = INF
	var caster_pos = get_caster_position()
	
	for player in players:
		if player.is_dead:
			continue
		var dist = player.global_position.distance_to(caster_pos)
		if dist < min_dist:
			min_dist = dist
			nearest = player
	
	return nearest  ## 在所有存活玩家中找到距离施法者最近的一个，用于自动瞄准或索敌技能

## 获取最近的敌人
## @return: 最近的 Enemy 或 null
func get_nearest_enemy() -> Enemy:
	var enemies = get_all_enemies()
	if enemies.is_empty():
		return null
	
	var nearest: Enemy = null
	var min_dist = INF
	var caster_pos = get_caster_position()
	
	for enemy in enemies:
		if enemy.is_dead:
			continue
		var dist = enemy.global_position.distance_to(caster_pos)
		if dist < min_dist:
			min_dist = dist
			nearest = enemy
	
	return nearest  ## 在所有存活敌人中找到距离施法者最近的一个，用于友方单位的索敌或范围攻击

## 在扇形范围内查找目标
## @param radius: 半径
## @param fov_degrees: 视野角度
## @param target_group: 目标分组（"player" 或 "enemy"）
## @return: 目标数组
func find_targets_in_sector(radius: float, fov_degrees: float, target_group: String) -> Array[Entity]:
	var tree = get_tree()
	if tree == null:
		return []
	
	var results: Array[Entity] = []
	var facing = get_caster_facing()
	var fov = deg_to_rad(fov_degrees)
	var caster_pos = get_caster_position()
	
	for node in tree.get_nodes_in_group(target_group):
		if not node is Entity or node.is_dead:
			continue
		
		var to_target = (node.global_position - caster_pos).normalized()
		var distance = node.global_position.distance_to(caster_pos)
		
		if distance <= radius and to_target.dot(facing) > cos(fov / 2):
			results.append(node)
	
	return results  ## 在施法者前方扇形区域内查找指定分组的目标，用于近战技能或锥形范围攻击的目标检测

## 计算伤害加成
## @param base_damage: 基础伤害
## @return: 最终伤害
func calculate_damage(base_damage: float) -> float:
	# 可扩展：根据施法者属性、buff、装备等计算
	return base_damage  ## 计算最终伤害值，当前直接返回基础伤害，可扩展为根据施法者属性、增益效果、装备等进行加成计算

## 显示伤害数字
## @param position: 显示位置
## @param damage: 伤害值
## @param color: 颜色
func show_damage_number(position: Vector2, damage: float, color: Color = Color.AZURE) -> void:
	FloatText.show_damage_text(str(int(damage)), position, color)  ## 在指定位置显示伤害数字浮动文本，使用 FloatText 单例实现

## 创建冲击波效果
## @param position: 中心位置
## @param radius: 冲击波半径
## @param duration: 持续时间
func create_shockwave(position: Vector2, radius: float, duration: float = 0.3) -> void:
	# 可扩展：实例化冲击波特效场景
	pass  ## 创建冲击波效果的占位方法，可扩展为实例化冲击波特效场景并播放动画

## 播放音效
## @param sound_path: 音效资源路径
func play_sound(sound_path: String) -> void:
	# 可扩展：使用 AudioStreamPlayer 播放
	pass  ## 播放音效的占位方法，可扩展为使用 AudioStreamPlayer 或 AudioManager 播放指定音效

## 发送事件
## @param event_name: 事件名
## @param data: 事件数据
func emit_event(event_name: String, data: Dictionary = {}) -> void:
	# 可扩展：通过 EventBus 发送自定义事件
	pass  ## 发送自定义事件的占位方法，可扩展为通过 EventBus 或其他事件系统发送技能相关事件

## 记录日志
## @param message: 日志消息
func log(message: String) -> void:
	print("[SkillContext] %s: %s" % [skill.name if skill else "Unknown", message])  ## 打印带技能名称前缀的日志消息，用于调试和追踪技能执行流程

## 获取调试信息
## @return: 调试信息字典
func get_debug_info() -> Dictionary:
	return {
		"caster": caster.name if caster else "null",
		"skill": skill.name if skill else "null",
		"target_count": targets.size(),
		"cooldown_progress": get_cooldown_progress(),
		"is_on_cooldown": is_on_cooldown()
	}  ## 返回包含施法者、技能、目标数量、冷却进度等信息的字典，用于调试面板显示或日志记录
