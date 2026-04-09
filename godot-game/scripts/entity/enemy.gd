class_name Enemy
extends Entity

## 敌人 AI 类
## 继承自 Entity，实现基于状态机的 AI 行为
##
## 状态机流程：
##   PATROL → CHASE → ATTACK_WINDUP → ATTACKING → ATTACK_RECOVERY → (CHASE/PATROL)
##              ↓
##           RETURN（追击超范围时）
##
## 架构位置：
##   - 父类：Entity
##   - 组件：SkillController（攻击）、Pathfinding（避障）
##   - 目标检测：视野检测（距离 + 扇形 FOV）

enum State {
	PATROL,           ## 巡逻：在范围内随机移动
	CHASE,            ## 追击：发现玩家后追击
	ATTACK_WINDUP,    ## 攻击前摇：蓄力阶段（可被打断）
	ATTACKING,        ## 攻击中：技能释放阶段
	ATTACK_RECOVERY,  ## 攻击后摇：恢复阶段
	RETURN            ## 返回：追击超出范围后返回巡逻区
}

@onready var skill_controller: SkillController = $SkillController
@onready var hit_particles: CPUParticles2D = $CpuHitParticles
@onready var pathfinding: Pathfinding = $Pathfinding

@export var speed: float = 30                    ## 追击移动速度
@export var patrol_speed: float = 20             ## 巡逻移动速度
@export var stop_distance: float = 8             ## 到达判定距离
@export var attack_windup_time: float = 0.2      ## 攻击前摇时长
@export var attack_recovery_time: float = 0.6    ## 攻击后摇时长
@export var attack_position_offset: float = 20   ## 攻击站位偏移（目标左右侧）
@export var attack_range: float = 35             ## 攻击距离
@export var attack_fov: float = 100.0            ## 攻击视野角度（扇形）
@export var sight_range: float = 120.0           ## 发现玩家的最大距离
@export var out_of_bounds_time: float = 2.0      ## 离开巡逻范围多久触发返回
@export var patrol_idle_time: float = 1.5        ## 巡逻点到达后等待时长
@export var respawn_time: float = 8.0            ## 重生倒计时

var target: Player = null                        ## 当前追击目标
var state: State = State.PATROL                  ## 当前状态
var state_timer: float = 0.0                     ## 状态计时器（前摇/后摇）
var patrol_target: Vector2 = Vector2.ZERO        ## 当前巡逻目标点
var patrol_idle_timer: float = 0.0               ## 巡逻等待计时
var chase_origin: Vector2 = Vector2.ZERO         ## 开始追击时的位置
var out_of_bounds_timer: float = 0.0             ## 离开巡逻范围计时
var respawn_timer: float = 0.0                   ## 重生倒计时
var is_respawning: bool = false                  ## 重生状态标记
var patrol_rect: Rect2 = Rect2()                 ## 巡逻范围矩形

const STICKY_FACTOR: float = 0.7                 ## 黏性目标系数（越小越不容易换目标）

func _ready() -> void:
	super._ready()
	_init_patrol_rect()
	patrol_target = _pick_patrol_target()

## 初始化巡逻范围
## 从 Road TileMapLayer 动态获取，转换为世界坐标矩形
func _init_patrol_rect() -> void:
	var road_layer = get_tree().get_first_node_in_group("road")
	if road_layer and road_layer is TileMapLayer:
		var used = road_layer.get_used_rect()
		var tile_size = road_layer.tile_set.tile_size
		var scale_v = road_layer.scale
		var pos = road_layer.position
		patrol_rect = Rect2(
			pos.x + used.position.x * tile_size.x * scale_v.x,
			pos.y + used.position.y * tile_size.y * scale_v.y,
			used.size.x * tile_size.x * scale_v.x,
			used.size.y * tile_size.y * scale_v.y
		)
	else:
		## 后备：硬编码范围（Road tiles: -7,-7 to 6,6, scale=1.5, tile_size=16）
		patrol_rect = Rect2(-168, -168, 336, 336)

func _process(delta: float) -> void:
	## 重生倒计时（独立于状态机）
	if is_respawning:
		respawn_timer -= delta
		if respawn_timer <= 0:
			_respawn()
		return
	
	if is_dead:
		return
	
	## 目标死亡处理
	if target != null and target.is_dead:
		target = null
		if state in [State.ATTACK_WINDUP, State.ATTACKING]:
			state = State.ATTACK_RECOVERY
			state_timer = attack_recovery_time * 0.5
			play_animation(AnimationWrapper.new("idle", false))
			return
		elif state == State.CHASE:
			var new_target = _find_nearest_player()
			if new_target != null:
				target = new_target
			else:
				state = State.PATROL
				patrol_target = _pick_patrol_target()
				patrol_idle_timer = 0.0
				return
	
	## 目标丢失处理
	if target == null and state in [State.CHASE, State.ATTACK_WINDUP, State.ATTACKING]:
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
	
	## 外部推力衰减
	if external_velocity != Vector2.ZERO:
		external_velocity = external_velocity.move_toward(
			Vector2.ZERO,
			external_velocity.length() * external_velocity_decay * delta
		)
		if external_velocity.length() < 1.0:
			external_velocity = Vector2.ZERO
	
	## 状态机分发
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

# ==================== 目标感知 ====================

## 获取所有存活玩家
func _get_alive_players() -> Array[Player] -> void:
	var result: Array[Player] = []
	for node in get_tree().get_nodes_in_group("player"):
		if node is Player and not node.is_dead:
			result.append(node)
	return result

## 找到最近的存活玩家
func _find_nearest_player() -> Player:
	var players = _get_alive_players()
	if players.is_empty():
		return null
	
	var nearest: Player = null
	var nearest_dist: float = INF
	for p in players:
		var dist = position.distance_to(p.position)
		if dist < nearest_dist:
			nearest_dist = dist
			nearest = p
	return nearest

## 黏性目标切换
## 策略：新目标距离 < 当前距离 × STICKY_FACTOR 才切换
## 目的：防止目标频繁抖动，保持攻击专注度
func _update_target() -> void:
	var candidate = _find_nearest_player()
	if candidate == null:
		target = null
		return
	
	if target == null or target.is_dead:
		target = candidate
		return
	
	if candidate == target:
		return
	
	var current_dist = position.distance_to(target.position)
	var candidate_dist = position.distance_to(candidate.position)
	if candidate_dist < current_dist * STICKY_FACTOR:
		target = candidate

## 检查视野内是否有玩家
func _can_see_any_player() -> bool:
	for p in _get_alive_players():
		if position.distance_to(p.position) <= sight_range:
			return true
	return false

## 找到视野内最近的玩家
func _find_nearest_visible_player() -> Player:
	var nearest: Player = null
	var nearest_dist: float = INF
	for p in _get_alive_players():
		var dist = position.distance_to(p.position)
		if dist <= sight_range and dist < nearest_dist:
			nearest_dist = dist
			nearest = p
	return nearest

# ==================== 重生 ====================

## 开始重生倒计时
## 操作：隐藏实体、禁用碰撞、清除速度
func _start_respawn() -> void:
	is_respawning = true
	respawn_timer = respawn_time
	
	visible = false
	$CollisionShape2D.set_deferred("disabled", true)
	$Area2D/CollisionShape2D.set_deferred("disabled", true)
	velocity = Vector2.ZERO
	external_velocity = Vector2.ZERO
	
	if hit_particles != null:
		hit_particles.emitting = false

## 复活
## 操作：重置状态、随机位置、清除冷却
func _respawn() -> void:
	is_respawning = false
	is_dead = false
	current_health = max_health
	target = null
	
	state = State.PATROL
	state_timer = 0.0
	patrol_idle_timer = 0.0
	out_of_bounds_timer = 0.0
	current_animation_wrapper = null
	
	position = _pick_respawn_position()
	patrol_target = _pick_patrol_target()
	
	## 重置技能冷却
	for skill in skill_controller.cooldowns.keys():
		skill_controller.cooldowns[skill] = 0.0
		skill.current_cooldown = 0.0
	
	visible = true
	$CollisionShape2D.set_deferred("disabled", false)
	$Area2D/CollisionShape2D.set_deferred("disabled", false)
	
	if animated_sprite.material != null:
		animated_sprite.material.set_shader_parameter("is_hurt", false)
	
	play_animation(AnimationWrapper.new("idle", false))

# ==================== 巡逻 ====================

## 巡逻状态
## 行为：扫描玩家 → 等待 → 移动 → 选新目标
func _process_patrol(delta: float) -> void:
	var visible_player = _find_nearest_visible_player()
	if visible_player != null:
		target = visible_player
		_start_chase()
		return
	
	if patrol_idle_timer > 0:
		patrol_idle_timer -= delta
		play_animation(AnimationWrapper.new("idle", false))
		return
	
	var to_target = patrol_target - position
	if to_target.length() <= stop_distance or to_target.length() <= 0.1:
		patrol_idle_timer = patrol_idle_time
		patrol_target = _pick_patrol_target()
		return
	
	var direction
	if pathfinding != null:
		direction = pathfinding.find_path(patrol_target).normalized()
	else:
		direction = to_target.normalized()
	
	velocity = direction * patrol_speed + external_velocity
	move_and_slide()
	play_animation(AnimationWrapper.new("run", false))
	_face_target(to_target)

## 选取巡逻目标点
func _pick_patrol_target() -> Vector2:
	var margin = 16.0
	var x = randf_range(patrol_rect.position.x + margin, patrol_rect.end.x - margin)
	var y = randf_range(patrol_rect.position.y + margin, patrol_rect.end.y - margin)
	return Vector2(x, y)

## 选取重生位置（远离玩家）
## 尝试 10 次找安全位置
func _pick_respawn_position() -> Vector2:
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
	
	return _pick_patrol_target()

## 检查是否在巡逻范围内
func _is_in_patrol_area() -> bool:
	return patrol_rect.has_point(position)

# ==================== 追击 ====================

## 进入追击状态
func _start_chase() -> void:
	chase_origin = position
	out_of_bounds_timer = 0.0
	state = State.CHASE

## 追击状态
## 逻辑：
##   1. 黏性目标切换
##   2. 目标丢失 → 回巡逻
##   3. 满足攻击条件 → 进入前摇
##   4. 到达攻击站位 → 进入前摇
##   5. 超范围太久 → 返回
func _process_chase(delta: float) -> void:
	_update_target()
	
	if target == null:
		state = State.PATROL
		patrol_target = _pick_patrol_target()
		patrol_idle_timer = 0.0
		return
	
	var to_player = target.position - self.position
	_face_target(to_player)
	
	## 条件1：当前位置可直接攻击
	if _can_hit_target():
		state = State.ATTACK_WINDUP
		state_timer = attack_windup_time
		play_animation(AnimationWrapper.new("idle", false))
		return
	
	## 条件2：到达攻击站位
	var attack_pos = _get_attack_position()
	var to_attack_pos = attack_pos - self.position
	if to_attack_pos.length() <= stop_distance:
		state = State.ATTACK_WINDUP
		state_timer = attack_windup_time
		play_animation(AnimationWrapper.new("idle", false))
		return
	
	## 继续移动
	var direction
	if pathfinding != null:
		direction = pathfinding.find_path(attack_pos).normalized()
	else:
		direction = to_attack_pos.normalized()
	
	velocity = direction * speed + external_velocity
	move_and_slide()
	play_animation(AnimationWrapper.new("run", false))
	
	## 超范围检测
	if not _is_in_patrol_area():
		out_of_bounds_timer += delta
		if out_of_bounds_timer >= out_of_bounds_time:
			state = State.RETURN
			out_of_bounds_timer = 0.0
			return
	else:
		out_of_bounds_timer = 0.0

# ==================== 返回 ====================

## 返回状态（追击超范围后返回）
## 逻辑：返回途中若发现玩家且仍在巡逻范围内，可重新追击
func _process_return(delta: float) -> void:
	var visible_player = _find_nearest_visible_player()
	if visible_player != null and _is_in_patrol_area():
		target = visible_player
		_start_chase()
		return
	
	var to_origin = chase_origin - position
	if to_origin.length() <= stop_distance:
		state = State.PATROL
		patrol_idle_timer = 0.0
		patrol_target = _pick_patrol_target()
		return
	
	var direction
	if pathfinding != null:
		direction = pathfinding.find_path(chase_origin).normalized()
	else:
		direction = to_origin.normalized()
	
	velocity = direction * speed + external_velocity
	move_and_slide()
	play_animation(AnimationWrapper.new("run", false))
	_face_target(to_origin)

# ==================== 攻击 ====================

## 计算攻击站位
## 策略：站在目标左侧或右侧 attack_position_offset 距离处
func _get_attack_position() -> Vector2:
	if target == null:
		return position
	
	var to_player = target.position - self.position
	if to_player.x >= 0:
		return target.position + Vector2(-attack_position_offset, 0)  # 目标在右，站左边
	else:
		return target.position + Vector2(attack_position_offset, 0)   # 目标在左，站右边

## 检查是否满足攻击条件
## 条件：距离在 attack_range 内 + 目标在扇形视野内
func _can_hit_target() -> bool:
	if target == null:
		return false
	
	var to_player = target.position - self.position
	var distance = to_player.length()
	if distance > attack_range:
		return false
	
	var face_dir = Vector2(1, 0)
	if animated_sprite.flip_h:
		face_dir = Vector2(-1, 0)
	
	var fov = deg_to_rad(attack_fov)
	return to_player.normalized().dot(face_dir) > cos(fov / 2)

## 攻击前摇
## 行为：面向目标 + 闪烁提示 + 不可移动
## 可被打断：目标死亡时中断进入后摇
func _process_windup(delta: float) -> void:
	if target == null or target.is_dead:
		state = State.ATTACK_RECOVERY
		state_timer = attack_recovery_time * 0.5
		if animated_sprite.material != null:
			animated_sprite.material.set_shader_parameter("is_hurt", false)
		play_animation(AnimationWrapper.new("idle", false))
		return
	
	var to_player = target.position - self.position
	_face_target(to_player)
	
	## 前摇视觉提示：快速闪烁
	if animated_sprite.material != null:
		var flash = sin(state_timer * 20.0) > 0
		animated_sprite.material.set_shader_parameter("is_hurt", flash)
	
	if external_velocity != Vector2.ZERO:
		velocity = external_velocity
		move_and_slide()
	
	state_timer -= delta
	if state_timer <= 0:
		if animated_sprite.material != null:
			animated_sprite.material.set_shader_parameter("is_hurt", false)
		skill_controller.trigger_skill_by_idx(0)
		state = State.ATTACKING
		state_timer = 0.75  ## slash 动画时长：6帧 / 8fps = 0.75s

## 攻击中
## 行为：播放攻击动画，计时结束后进入后摇
func _process_attacking(delta: float) -> void:
	state_timer -= delta
	if state_timer <= 0:
		state = State.ATTACK_RECOVERY
		state_timer = attack_recovery_time
		play_animation(AnimationWrapper.new("idle", false))

## 攻击后摇
## 行为：站立恢复，结束后重新评估目标
func _process_recovery(delta: float) -> void:
	play_animation(AnimationWrapper.new("idle", false))
	
	if external_velocity != Vector2.ZERO:
		velocity = external_velocity
		move_and_slide()
	
	state_timer -= delta
	if state_timer <= 0:
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

## 调整朝向
func _face_target(velocity: Vector2) -> void:
	if velocity.x > 0:
		animated_sprite.flip_h = false
	elif velocity.x < 0:
		animated_sprite.flip_h = true

## 动画完成回调
func _on_animated_sprite_2d_animation_finished() -> void:
	if current_animation_wrapper != null and current_animation_wrapper.name == "die":
		_start_respawn()
		return
	current_animation_wrapper = null

## 受击特效（添加粒子）
func _show_damage_taken_effect() -> void:
	super._show_damage_taken_effect()
	if hit_particles != null:
		hit_particles.emitting = true
