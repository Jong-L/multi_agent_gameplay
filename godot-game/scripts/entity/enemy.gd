class_name Enemy
extends Entity

enum State { PATROL, CHASE, ATTACK_WINDUP, ATTACKING, ATTACK_RECOVERY, RETURN }

var target:Player
var state:State = State.PATROL
var state_timer:float = 0.0

# 巡逻相关
var patrol_target:Vector2 = Vector2.ZERO
var patrol_idle_timer:float = 0.0  # 到达巡逻点后的等待时间

# 追击相关
var chase_origin:Vector2 = Vector2.ZERO  # 开始追击时的位置
var out_of_bounds_timer:float = 0.0      # 离开游走范围计时

# 重生相关
var respawn_timer:float = 0.0
var is_respawning:bool = false

# 游走范围（从 Road TileMapLayer 动态获取）
var patrol_rect:Rect2 = Rect2()

# 黏性系数：新候选距离 < 当前距离 × sticky_factor 才切换
const STICKY_FACTOR:float = 0.7

@onready var skill_controller:SkillController=$SkillController
@onready var hit_particles:CPUParticles2D=$CpuHitParticles
@onready var pathfinding:Pathfinding=$Pathfinding

@export var speed:float=30
@export var patrol_speed:float=20
@export var stop_distance:float=8
@export var attack_windup_time:float=0.2
@export var attack_recovery_time:float=0.6
@export var attack_position_offset:float=20
@export var attack_range:float=35
@export var attack_fov:float=100.0
@export var sight_range:float=120.0
@export var out_of_bounds_time:float=2.0
@export var patrol_idle_time:float=1.5
@export var respawn_time:float=8.0

func _ready() -> void:
	super._ready()
	_init_patrol_rect()
	patrol_target = _pick_patrol_target()

func _init_patrol_rect() -> void:
	"""从场景中的 Road TileMapLayer 动态获取巡逻范围"""
	var road_layer = get_tree().get_first_node_in_group("road")
	if road_layer and road_layer is TileMapLayer:
		var used = road_layer.get_used_rect()
		var tile_size = road_layer.tile_set.tile_size
		var scale_v = road_layer.scale
		var pos = road_layer.position
		# 将 tile 坐标转换为世界坐标
		patrol_rect = Rect2(
			pos.x + used.position.x * tile_size.x * scale_v.x,
			pos.y + used.position.y * tile_size.y * scale_v.y,
			used.size.x * tile_size.x * scale_v.x,
			used.size.y * tile_size.y * scale_v.y
		)
	else:
		# 后备：硬编码范围（Road tiles: -7,-7 to 6,6, scale=1.5, tile_size=16）
		patrol_rect = Rect2(-168, -168, 336, 336)

func _process(delta: float) -> void:
	# 重生倒计时（独立于其他状态）
	if is_respawning:
		respawn_timer -= delta
		if respawn_timer <= 0:
			_respawn()
		return
	
	if is_dead:return
	
	# 目标死亡：立即重新评估
	if target != null and target.is_dead:
		target = null
		if state in [State.ATTACK_WINDUP, State.ATTACKING]:
			# 攻击中目标死亡，中断攻击回到后摇
			state = State.ATTACK_RECOVERY
			state_timer = attack_recovery_time * 0.5
			play_animation(AnimationWrapper.new("idle",false))
			return
		elif state == State.CHASE:
			# 追击中目标死亡，尝试找新目标
			var new_target = _find_nearest_player()
			if new_target != null:
				target = new_target
			else:
				state = State.PATROL
				patrol_target = _pick_patrol_target()
				patrol_idle_timer = 0.0
				return
	
	if target == null and state in [State.CHASE, State.ATTACK_WINDUP, State.ATTACKING]:
		# 没有目标但还在攻击/追击状态，尝试找新目标或回到巡逻
		var new_target = _find_nearest_player()
		if new_target != null:
			target = new_target
			if state in [State.ATTACK_WINDUP, State.ATTACKING]:
				state = State.CHASE
		else:
			state = State.PATROL
			patrol_target = _pick_patrol_target()
			patrol_idle_timer = 0.0
			return
	
	# 衰减外部推力
	if external_velocity!=Vector2.ZERO:
		external_velocity=external_velocity.move_toward(Vector2.ZERO,external_velocity.length()*external_velocity_decay*delta)
		if external_velocity.length()<1.0:
			external_velocity=Vector2.ZERO
	
	match state:
		State.PATROL:
			_process_patrol(delta)
		State.CHASE:
			_process_chase(delta)
		State.ATTACK_WINDUP:
			_process_windup(delta)
		State.ATTACKING:
			_process_attacking(delta)
		State.ATTACK_RECOVERY:
			_process_recovery(delta)
		State.RETURN:
			_process_return(delta)

# ==================== 多目标感知 ====================

func _get_alive_players() -> Array[Player]:
	"""获取所有活着的玩家"""
	var result:Array[Player] = []
	for node in get_tree().get_nodes_in_group("player"):
		if node is Player and not node.is_dead:
			result.append(node)
	return result

func _find_nearest_player() -> Player:
	"""找到最近的活玩家，没有则返回null"""
	var players = _get_alive_players()
	if players.is_empty():
		return null
	var nearest:Player = null
	var nearest_dist:float = INF
	for p in players:
		var dist = position.distance_to(p.position)
		if dist < nearest_dist:
			nearest_dist = dist
			nearest = p
	return nearest

func _update_target() -> void:
	"""CHASE阶段的黏性目标切换：新候选距离<当前×STICKY_FACTOR才换"""
	var candidate = _find_nearest_player()
	if candidate == null:
		# 没有活玩家了
		target = null
		return
	if target == null or target.is_dead:
		# 当前没有目标，直接锁定最近的
		target = candidate
		return
	if candidate == target:
		# 最近的还是当前目标，保持
		return
	# 不同玩家：只有显著更近才切换
	var current_dist = position.distance_to(target.position)
	var candidate_dist = position.distance_to(candidate.position)
	if candidate_dist < current_dist * STICKY_FACTOR:
		target = candidate

func _can_see_any_player() -> bool:
	"""检查是否有任何活玩家在圆形视野内"""
	for p in _get_alive_players():
		if position.distance_to(p.position) <= sight_range:
			return true
	return false

func _find_nearest_visible_player() -> Player:
	"""找到视野内最近的活玩家"""
	var nearest:Player = null
	var nearest_dist:float = INF
	for p in _get_alive_players():
		var dist = position.distance_to(p.position)
		if dist <= sight_range and dist < nearest_dist:
			nearest_dist = dist
			nearest = p
	return nearest

# ==================== 重生 ====================

func _start_respawn() -> void:
	"""敌人死亡后开始重生倒计时"""
	is_respawning = true
	respawn_timer = respawn_time
	# 隐藏敌人（视觉+物理）
	visible = false
	# 禁用碰撞
	$CollisionShape2D.set_deferred("disabled", true)
	$Area2D/CollisionShape2D.set_deferred("disabled", true)
	# 清除速度
	velocity = Vector2.ZERO
	external_velocity = Vector2.ZERO
	# 停止粒子效果
	if hit_particles != null:
		hit_particles.emitting = false

func _respawn() -> void:
	"""重置敌人状态并复活"""
	is_respawning = false
	is_dead = false
	current_health = max_health
	target = null
	
	# 重置状态机
	state = State.PATROL
	state_timer = 0.0
	patrol_idle_timer = 0.0
	out_of_bounds_timer = 0.0
	current_animation_wrapper = null
	
	# 在巡逻范围内随机位置重生（远离所有玩家）
	position = _pick_respawn_position()
	patrol_target = _pick_patrol_target()
	
	# 重置技能冷却
	for skill in skill_controller.cooldowns.keys():
		skill_controller.cooldowns[skill] = 0.0
		skill.current_cooldown = 0.0
	
	# 恢复视觉+物理
	visible = true
	$CollisionShape2D.set_deferred("disabled", false)
	$Area2D/CollisionShape2D.set_deferred("disabled", false)
	
	# 清除闪烁效果
	if animated_sprite.material != null:
		animated_sprite.material.set_shader_parameter("is_hurt", false)
	
	# 播放 idle 动画
	play_animation(AnimationWrapper.new("idle",false))

# ==================== 巡逻 ====================

func _process_patrol(delta: float) -> void:
	# 扫描视野内所有玩家，发现任一即进入追击
	var visible_player = _find_nearest_visible_player()
	if visible_player != null:
		target = visible_player
		_start_chase()
		return
	
	# 等待中（到达巡逻点后的小憩）
	if patrol_idle_timer > 0:
		patrol_idle_timer -= delta
		play_animation(AnimationWrapper.new("idle",false))
		return
	
	# 是否到达当前巡逻目标
	var to_target = patrol_target - position
	if to_target.length() <= stop_distance or to_target.length() <= 0.1:
		# 到达，等待后选新目标
		patrol_idle_timer = patrol_idle_time
		patrol_target = _pick_patrol_target()
		return
	
	# 向巡逻目标移动
	var direction
	if pathfinding!=null:
		direction=pathfinding.find_path(patrol_target).normalized()
	else:
		direction=to_target.normalized()
	velocity=direction*patrol_speed+external_velocity
	move_and_slide()
	play_animation(AnimationWrapper.new("run",false))
	_face_target(to_target)

func _pick_patrol_target() -> Vector2:
	"""在巡逻范围内随机选取一个目标点"""
	var margin = 16.0  # 留一点边距，避免太贴墙
	var x = randf_range(patrol_rect.position.x + margin, patrol_rect.end.x - margin)
	var y = randf_range(patrol_rect.position.y + margin, patrol_rect.end.y - margin)
	return Vector2(x, y)

func _pick_respawn_position() -> Vector2:
	"""在巡逻范围内选取一个远离所有玩家的重生位置"""
	var min_player_distance = 80.0
	var players = _get_alive_players()
	for attempt in range(10):
		var pos = _pick_patrol_target()
		var too_close = false
		for p in players:
			if pos.distance_to(p.position) < min_player_distance:
				too_close = true
				break
		if not too_close:
			return pos
	# 10次都没找到，就用最后一次的结果
	return _pick_patrol_target()

# ==================== 视野检测 ====================

func _is_in_patrol_area() -> bool:
	"""检查当前位置是否在巡逻范围内"""
	return patrol_rect.has_point(position)

# ==================== 追击 ====================

func _start_chase() -> void:
	"""从巡逻/返回切换到追击"""
	chase_origin = position
	out_of_bounds_timer = 0.0
	state = State.CHASE

func _process_chase(delta: float) -> void:
	# 黏性目标切换（CHASE阶段允许换目标）
	_update_target()
	
	# 目标丢失（全部死亡或不在视野）
	if target == null:
		state = State.PATROL
		patrol_target = _pick_patrol_target()
		patrol_idle_timer = 0.0
		return
	
	# 始终面朝目标
	var to_player=target.position-self.position
	_face_target(to_player)
	
	# 检查1：当前位置已满足攻击条件 → 直接攻击
	if _can_hit_target():
		state=State.ATTACK_WINDUP
		state_timer=attack_windup_time
		play_animation(AnimationWrapper.new("idle",false))
		return
	
	# 检查2：到达偏好点 → 攻击
	var attack_pos=_get_attack_position()
	var to_attack_pos=attack_pos-self.position
	if to_attack_pos.length()<=stop_distance:
		state=State.ATTACK_WINDUP
		state_timer=attack_windup_time
		play_animation(AnimationWrapper.new("idle",false))
		return
	
	# 都不满足：继续移向偏好点
	var direction
	if pathfinding!=null:
		direction=pathfinding.find_path(attack_pos).normalized()
	else:
		direction=to_attack_pos.normalized()
	velocity=direction*speed+external_velocity
	move_and_slide()
	play_animation(AnimationWrapper.new("run",false))
	
	# 离开巡逻范围检测
	if not _is_in_patrol_area():
		out_of_bounds_timer += delta
		if out_of_bounds_timer >= out_of_bounds_time:
			state = State.RETURN
			out_of_bounds_timer = 0.0
			return
	else:
		out_of_bounds_timer = 0.0

# ==================== 返回 ====================

func _process_return(delta: float) -> void:
	# 返回途中，如果有任何玩家在视野内且自己仍在巡逻范围内，可重新追击
	var visible_player = _find_nearest_visible_player()
	if visible_player != null and _is_in_patrol_area():
		target = visible_player
		_start_chase()
		return
	
	var to_origin = chase_origin - position
	if to_origin.length() <= stop_distance:
		# 回到巡逻范围，切回巡逻
		state = State.PATROL
		patrol_idle_timer = 0.0  # 立即选新巡逻点
		patrol_target = _pick_patrol_target()
		return
	
	# 向 chase_origin 移动
	var direction
	if pathfinding!=null:
		direction=pathfinding.find_path(chase_origin).normalized()
	else:
		direction=to_origin.normalized()
	velocity=direction*speed+external_velocity
	move_and_slide()
	play_animation(AnimationWrapper.new("run",false))
	_face_target(to_origin)

# ==================== 攻击 ====================

func _get_attack_position()->Vector2:
	"""计算敌人应该移动到的攻击站位（目标左/右偏移点）"""
	if target == null:
		return position
	var to_player=target.position-self.position
	if to_player.x>=0:
		return target.position+Vector2(-attack_position_offset,0)
	else:
		return target.position+Vector2(attack_position_offset,0)

func _can_hit_target()->bool:
	"""检查当前位置+朝向是否满足攻击条件（距离+扇形方向）"""
	if target == null:
		return false
	var to_player=target.position-self.position
	var distance=to_player.length()
	if distance>attack_range:
		return false	
	var face_dir=Vector2(1,0)
	if animated_sprite.flip_h:
		face_dir=Vector2(-1,0)
	var fov=deg_to_rad(attack_fov)
	return to_player.normalized().dot(face_dir)>cos(fov/2)

func _process_windup(delta: float) -> void:
	# 前摇期间锁定目标不切换，但目标死亡则中断
	if target == null or target.is_dead:
		state = State.ATTACK_RECOVERY
		state_timer = attack_recovery_time * 0.5
		if animated_sprite.material != null:
			animated_sprite.material.set_shader_parameter("is_hurt", false)
		play_animation(AnimationWrapper.new("idle",false))
		return
	
	# 前摇期间转向目标，但不能移动
	var to_player=target.position-self.position
	_face_target(to_player)
	
	# 前摇视觉提示：快速闪烁
	if animated_sprite.material != null:
		var flash = sin(state_timer * 20.0) > 0
		animated_sprite.material.set_shader_parameter("is_hurt", flash)
	
	# 推力仍生效
	if external_velocity!=Vector2.ZERO:
		velocity=external_velocity
		move_and_slide()
	
	state_timer -= delta
	if state_timer <= 0:
		# 清除闪烁
		if animated_sprite.material != null:
			animated_sprite.material.set_shader_parameter("is_hurt", false)
		# 前摇结束，释放技能并进入攻击中状态
		skill_controller.trigger_skill_by_idx(0)
		state = State.ATTACKING
		# 用 slash 动画的实际时长（6帧 / speed 8 = 0.75秒）作为攻击中状态计时
		state_timer = 0.75

func _process_attacking(delta: float) -> void:
	# 攻击中状态：锁定目标不切换
	state_timer -= delta
	if state_timer <= 0:
		state = State.ATTACK_RECOVERY
		state_timer = attack_recovery_time
		play_animation(AnimationWrapper.new("idle",false))

func _process_recovery(delta: float) -> void:
	# 后摇期间什么都不做（站着不动）
	play_animation(AnimationWrapper.new("idle",false))
	
	# 推力仍生效
	if external_velocity!=Vector2.ZERO:
		velocity=external_velocity
		move_and_slide()
	
	state_timer -= delta
	if state_timer <= 0:
		# 后摇结束：重新评估目标
		var visible_player = _find_nearest_visible_player()
		if visible_player != null:
			target = visible_player
			state = State.CHASE
		else:
			target = null
			state = State.PATROL
			patrol_target = _pick_patrol_target()
			patrol_idle_timer = 0.0

# ==================== 通用 ====================

func _face_target(velocity):
	if velocity.x>0:
		animated_sprite.flip_h=false
	elif velocity.x<0:
		animated_sprite.flip_h=true

func _on_animated_sprite_2d_animation_finished() -> void:
	if current_animation_wrapper!=null and current_animation_wrapper.name=="die":
		# 死亡动画播完，直接开始重生倒计时（不再 queue_free）
		_start_respawn()
		return
	current_animation_wrapper=null

func _show_damage_taken_effect():
	super._show_damage_taken_effect()
	
	if hit_particles!=null:
		hit_particles.emitting=true
