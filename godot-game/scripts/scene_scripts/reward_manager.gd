extends Node
class_name RewardManager

" 奖励管理器
 统一管理所有 RL 奖励的发放，奖励计算不嵌入各个游戏逻辑代码
 支持 per-player 独立奖励配置（通过 game_config.use_per_player_reward 开关）
 提供奖励变更接口（如 add_reward），各模块调用即可进行奖励更改

 监听的信号：
   - EventBus.entity_damaged  受伤/造成伤害奖励
   - EventBus.enemy_died      击杀敌人奖励
   - EventBus.player_died     击杀玩家/死亡惩罚
   - EventBus.reward_ball_collected → 拾取奖励球奖励"

@export var _sync_node:Sync
@export var game_config:GameConfig
@export var reward_config:RewardConfig                       # 默认奖励配置（所有玩家共享）
@export var player_reward_configs:Array[RewardConfig] = []   # 每个玩家的独立奖励配置（索引0~3对应Player0~3）

var _play_scene: PlayScene = null
var _reward_logger: RewardLogger = null

#每个玩家的"上次获得正奖励"的游戏时间
var _last_reward_time: Dictionary = {}

#纯奖励累计值（不含塑形奖励
var _pure_rewards: Dictionary = {}

#累计游戏时间
var _game_time: float = 0.0

# ── 撞墙计数器 ──
var _wall_collision_counters: Dictionary = {}  # {player_id: 连续撞墙物理帧数}

# ── 势能塑形系统 ──
var _shaping_gamma: float=0.99       # 塑形折扣因子
var _skip_ball_potential_shaping_once: Dictionary = {}
var _prev_ball_potentials: Dictionary = {}
var _prev_wall_potentials: Dictionary = {}
# 方案4需要缓存上帧距离
var _prev_ball_distances: Dictionary = {}  # {player_id: 上帧最近球距离}
var action_repeat_count:int=0
var first_frame:bool=true  #第一帧

# ── 配置路由 ──

# 根据玩家ID获取对应的奖励配置
func _cfg(player_id: int) -> RewardConfig:
	if game_config and game_config.use_per_player_reward:
		if player_id >= 0 and player_id < player_reward_configs.size() and player_reward_configs[player_id] != null:
			return player_reward_configs[player_id]
	return reward_config

# ── 生命周期 ──

func _ready() -> void:
	_play_scene = get_parent() if get_parent() is PlayScene else null
	_connect_signals()

func _exit_tree() -> void:
	_disconnect_signals()
	_disconnect_skill_signals()
	if _reward_logger != null:
		_reward_logger.flush()

func _physics_process(delta: float) -> void:
	_game_time += delta

	#Rewardmanager在同一个物理帧在sync之前执行，让sync获取最新的奖励，之前有一个物理帧的误差，实际获取的是上一个物理帧的奖励
	if action_repeat_count==0 and first_frame==false:
		_process_potential_shaping(delta)
		_process_wall_collision(delta)
		_process_starvation(delta)
		_process_center_shaping(delta)
		for player in _play_scene.players:
			if player.is_moving:
				on_player_moved(player)

	if first_frame==true:
		first_frame=false
	action_repeat_count=(action_repeat_count+1)%_sync_node.action_repeat


#全局信号连接
func _connect_signals() -> void:
	EventBus.entity_damaged.connect(_on_entity_damaged)
	EventBus.enemy_died.connect(_on_enemy_died)
	EventBus.player_died.connect(_on_player_died)
	EventBus.reward_ball_collected.connect(_on_reward_ball_collected)

func _disconnect_signals() -> void:
	if EventBus.entity_damaged.is_connected(_on_entity_damaged):
		EventBus.entity_damaged.disconnect(_on_entity_damaged)
	if EventBus.enemy_died.is_connected(_on_enemy_died):
		EventBus.enemy_died.disconnect(_on_enemy_died)
	if EventBus.player_died.is_connected(_on_player_died):
		EventBus.player_died.disconnect(_on_player_died)
	if EventBus.reward_ball_collected.is_connected(_on_reward_ball_collected):
		EventBus.reward_ball_collected.disconnect(_on_reward_ball_collected)

#初始化
func setup(play_scene: PlayScene) -> void:
	if _play_scene == null:
		_play_scene = play_scene
	_init_starvation_timers()
	_connect_skill_signals()
	# 从 Sync 节点获取 gamma
	_init_shaping_gamma()
	_init_potentials()
	# 根据配置决定是否创建 RewardLogger
	if game_config and game_config.reward_logger_enabled:
		_reward_logger = RewardLogger.new()
		_reward_logger.start_episode()

#初始化饥饿计时器（所有玩家当前游戏时间）
func _init_starvation_timers() -> void:
	_last_reward_time.clear()
	if _play_scene == null:
		return
	for player in _play_scene.players:
		_last_reward_time[player.player_id] = _game_time

#局部信号：连接所有玩家的 SkillController.skill_activated 信号
func _connect_skill_signals() -> void:
	if _play_scene == null:
		return
	for player in _play_scene.players:
		if player.skill_controller != null and not player.skill_controller.skill_activated.is_connected(_on_player_skill_activated):
			player.skill_controller.skill_activated.connect(_on_player_skill_activated)

#断开所有玩家的 SkillController 信号
func _disconnect_skill_signals() -> void:
	if _play_scene == null:
		return
	for player in _play_scene.players:
		if is_instance_valid(player) and player.skill_controller != null and player.skill_controller.skill_activated.is_connected(_on_player_skill_activated):
			player.skill_controller.skill_activated.disconnect(_on_player_skill_activated)

#给指定玩家增加奖励，更新上次正奖励时间
func add_reward(player_id: int, value: float, source: String = "") -> void:
	if _play_scene == null:
		return
	if player_id < 0 or player_id >= _play_scene.players.size():
		return

	var player := _play_scene.players[player_id]
	player.ai_controller.reward += value

	# 累计纯奖励（不含塑形奖励）并发射信号
	if not _pure_rewards.has(player_id):
		_pure_rewards[player_id] = 0.0
	_pure_rewards[player_id] += value
	EventBus.pure_reward_changed.emit(player_id, _pure_rewards[player_id])

	# 正奖励时刷新饥饿计时器
	if value > 0.1:
		_last_reward_time[player_id] = _game_time

	# 记录到纯净奖励日志（排除塑形奖励）
	if _reward_logger != null:
		_reward_logger.log_reward(player_id, source, value, _game_time)

## ── 事件处理 ──

#实体受伤处理
func _on_entity_damaged(entity: Entity, source: Entity) -> void:
	if _play_scene == null:
		return

	# 被击者：如果受伤的是玩家，给予受伤惩罚
	if entity is Player:
		var target_player := entity as Player
		add_reward(target_player.player_id, _cfg(target_player.player_id).bear_damage, "bear_damage")

	# 攻击者：如果攻击者是玩家
	if source is Player:
		var source_player := source as Player
		# 被击者是敌人
		if entity is Enemy:
			add_reward(source_player.player_id, _cfg(source_player.player_id).cause_damage_to_enemy, "cause_damage_to_enemy")
		# 被击者是玩家
		elif entity is Player and entity != source:
			add_reward(source_player.player_id, _cfg(source_player.player_id).cause_damage_to_player, "cause_damage_to_player")

## 敌人死亡处理
## @param enemy 死亡的敌人
func _on_enemy_died(enemy: Enemy) -> void:
	if _play_scene == null:
		return

	# 检查敌人记录的最后伤害来源，如果是玩家则给予击杀奖励
	if enemy.last_damage_source is Player:
		var killer := enemy.last_damage_source as Player
		add_reward(killer.player_id, _cfg(killer.player_id).kill_enemy, "kill_enemy")

## 玩家死亡处理
func _on_player_died(player: Player) -> void:
	if _play_scene == null:
		return

	# 死亡惩罚
	add_reward(player.player_id, _cfg(player.player_id).died, "died")

	# 检查击杀者
	if player.last_damage_source is Player:
		var killer := player.last_damage_source as Player
		add_reward(killer.player_id, _cfg(killer.player_id).kill_player, "kill_player")

#奖励球拾取处理,ball实例由奖励球管理器处理，此处不需要
func _on_reward_ball_collected(player_id: int, ball_type: int, _ball: RewardBall) -> void:
	var cfg := _cfg(player_id)
	if ball_type == RewardBall.BallType.TYPE_A:
		add_reward(player_id, cfg.collect_ball_A, "collect_ball_A")
	elif ball_type == RewardBall.BallType.TYPE_B:
		add_reward(player_id, cfg.collect_ball_B, "collect_ball_B")
	
	_reset_ball_shaping_after_ball_removed(_ball)

#玩家攻击惩罚,不同skill的惩罚力度不一样，但目前只有一个技能
func _on_player_skill_activated(entity: Entity, _skill: Skill) -> void:
	if entity is Player:
		var player := entity as Player
		add_reward(player.player_id, _cfg(player.player_id).attack, "attack")

## ── 每帧执行的持续奖励逻辑 ──

#移动惩罚
func on_player_moved(player: Player) -> void:
	add_reward(player.player_id, _cfg(player.player_id).run, "run")

func _reset_ball_shaping_after_ball_removed(removed_ball: RewardBall) -> void:
	if _play_scene == null:
		return

	for player in _play_scene.players:
		if not is_instance_valid(player) or player.is_dead:
			continue

		var pid := player.player_id
		var cfg := _cfg(pid)
		if not _removed_ball_affects_player_ball_shaping(player, removed_ball, cfg):
			continue
			
		#势能被影响，重置缓存
		_prev_ball_potentials[pid] = _calculate_ball_shaping_potential(player, cfg)

		var current_dist := _get_nearest_ball_distance(player)
		if current_dist == INF:
			_prev_ball_distances.erase(pid)
		else:
			_prev_ball_distances[pid] = current_dist

		_skip_ball_potential_shaping_once[pid] = true

func _removed_ball_affects_player_ball_shaping(player: Player, removed_ball: RewardBall, cfg: RewardConfig) -> bool:
	if not is_instance_valid(removed_ball) or _play_scene == null or _play_scene.reward_ball_manager == null:
		return false

	var vision_radius: float = _play_scene.vision_sensor.vision_radius if _play_scene.vision_sensor else 200.0
	var removed_dist := player.global_position.distance_to(removed_ball.global_position)
	#视野外的球不影响
	if removed_dist > vision_radius:
		return false

	#如果捡到的球在视野内
	if (
		cfg.ball_potential_func != RewardConfig.BallPotentialFunc.DISTANCE_REWARD
		and cfg.ball_potential_mode == RewardConfig.BallPotentialMode.ALL
	):
		return true
	
	#如果只计算最近奖励球的势能
	for ball in _play_scene.reward_ball_manager.reward_balls:
		if ball == removed_ball or not is_instance_valid(ball) or not ball.is_active:
			continue
		var dist := player.global_position.distance_to(ball.global_position)
		if dist <= vision_radius and dist < removed_dist:
			return false #被捡到的球不是离自己最近的

	return true

func _process_starvation(delta: float) -> void:
	if _play_scene == null:
		return

	for player in _play_scene.players:
		if player.is_dead:
			continue
		var starve_duration=compute_starve_duration(player)
		if starve_duration>0.0:
			var cfg := _cfg(player.player_id)
			# 根据增长函数计算衰减倍率
			var multiplier: float = MathUtils.starve_rate_multiplier(starve_duration, cfg.starve_more_func)
			var decrease: float = cfg.starve_reward_decrease * multiplier * delta
			add_reward(player.player_id, -decrease, "starvation")

#计算饥饿时间
func compute_starve_duration(player:Player)->float:
	var pid: int = player.player_id
	var cfg := _cfg(pid)
	var time_since_reward: float = _game_time - _last_reward_time[pid]
	if time_since_reward >= cfg.starve_time:
			# 饥饿时间 = time_since_reward - STARVE_TIME
			var starve_duration:float= min(time_since_reward - cfg.starve_time, cfg.max_starve_duration)
			return starve_duration
	return 0.0

## ── 势能塑形系统（Potential-Based Reward Shaping） ──

# 从 Sync 节点获取 shaping_gamma
func _init_shaping_gamma() -> void:
	if _sync_node != null and _sync_node.args != null:
		_shaping_gamma = _sync_node.args.get("gamma", "0.99").to_float()
		print("[RewardManager] 势能塑形 gamma = ", _shaping_gamma, " (来自命令行参数)")
	else:
		_shaping_gamma = 0.99
		print("[RewardManager] 势能塑形 gamma = ", _shaping_gamma, " (默认值)")

# 初始化所有玩家的上一帧势能缓存
func _init_potentials() -> void:
	_prev_ball_potentials.clear()
	_prev_wall_potentials.clear()
	_prev_ball_distances.clear()
	_skip_ball_potential_shaping_once.clear()
	if _play_scene == null:
		return
	for player in _play_scene.players:
		var pid := player.player_id
		var cfg := _cfg(pid)

		var ball_potential := _calculate_ball_shaping_potential(player, cfg)
		var wall_potential := _calculate_wall_shaping_potential(player, cfg)
		_prev_ball_potentials[pid] = ball_potential
		_prev_wall_potentials[pid] = wall_potential
		_prev_ball_distances[pid] = _get_nearest_ball_distance(player)


func _process_potential_shaping(_delta: float) -> void:
	if _play_scene == null:
		return

	for player in _play_scene.players:
		if player.is_dead:
			continue

		var pid := player.player_id
		var cfg := _cfg(pid)

		if cfg.ball_potential_func == RewardConfig.BallPotentialFunc.DISTANCE_REWARD:
			_process_ball_distance_reward(player, pid, cfg)
		else:
			_process_ball_potential_shaping(player, pid, cfg)
			
		_process_wall_potential_shaping(player, pid, cfg)

func _process_ball_potential_shaping(player: Player, pid: int, cfg: RewardConfig) -> void:
	var current_potential := _calculate_ball_shaping_potential(player, cfg)
	
	#重置势能并跳过奖励发放
	if _skip_ball_potential_shaping_once.get(pid, false):
		_prev_ball_potentials[pid] = current_potential
		_skip_ball_potential_shaping_once.erase(pid)
		return
	
	var prev_potential: float = _prev_ball_potentials.get(pid, current_potential)
	var shaping: float = _shaping_gamma * current_potential - prev_potential
	#if pid==0:
		#print(shaping)
	player.ai_controller.reward += shaping
	_prev_ball_potentials[pid] = current_potential

func _process_wall_potential_shaping(player: Player, pid: int, cfg: RewardConfig) -> void:
	# NONE 模式下无塑形；COLLISION 模式在 _process_wall_collision 中单独处理
	if cfg.wall_potential_mode in [RewardConfig.WallPotentialMode.NONE, RewardConfig.WallPotentialMode.COLLISION]:
		return

	var current_potential := _calculate_wall_shaping_potential(player, cfg)
	var prev_potential: float = _prev_wall_potentials.get(pid, current_potential)
	var shaping: float = _shaping_gamma * current_potential - prev_potential
	#if pid==0:
		#print(shaping)
	player.ai_controller.reward += shaping
	_prev_wall_potentials[pid] = current_potential

#距离奖励
func _process_ball_distance_reward(player: Player, pid: int, cfg: RewardConfig) -> void:
	var current_dist: float = _get_nearest_ball_distance(player)
	#重置距离缓存并过奖励发放
	if _skip_ball_potential_shaping_once.get(pid, false):
		if current_dist == INF:
			_prev_ball_distances.erase(pid)
		else:
			_prev_ball_distances[pid] = current_dist
		_skip_ball_potential_shaping_once.erase(pid)
		return

	# 自身势能未被影响，但是自己走出范围导致看不到奖励球
	if current_dist == INF:
		_prev_ball_distances.erase(pid)
		return

	var prev_dist: float = _prev_ball_distances.get(pid, current_dist)
	if prev_dist == INF:
		_prev_ball_distances[pid] = current_dist
		return

	var reward: float = cfg.distance_reward_scale * (prev_dist - current_dist)
	player.ai_controller.reward += reward

	#if pid == 0:
		#print("[RewardManager] 距离差奖励 = ", reward, " (prev=", prev_dist, ", cur=", current_dist, ") for player ", pid)

	_prev_ball_distances[pid] = current_dist

# 奖励球势能
func _calculate_ball_shaping_potential(player: Player, cfg: RewardConfig) -> float:
	if cfg.ball_potential_func == RewardConfig.BallPotentialFunc.DISTANCE_REWARD:
		return 0.0
	if cfg.ball_potential_mode == RewardConfig.BallPotentialMode.ALL:
		return calculate_ball_potential_all(player)
	#否则计算最近奖励球势能
	return calculate_ball_potential(player)

# 获取玩家到障碍物的碰撞距离列表（已过滤非碰撞射线）
func _get_collision_distances(player: Player, pid: int) -> Array[float]:
	if _play_scene == null:
		return []

	_play_scene.ensure_map_states_current()
	var map_state: Array = _play_scene.map_states.get(pid, [])

	if map_state.size() == 0:
		map_state = _play_scene._build_map_state(player)
		if map_state.size() == 0:
			return []

	var collision_distances: Array[float] = []
	for distance in map_state:
		if distance < 1.0:
			collision_distances.append(distance)

	return collision_distances

func _calculate_wall_shaping_potential(player: Player, cfg: RewardConfig) -> float:
	if cfg.wall_potential_mode in [RewardConfig.WallPotentialMode.LINEAR, RewardConfig.WallPotentialMode.INVERSE]:
		return _calculate_wall_potential(player, cfg)
	return 0.0

# 障碍物势能计算
func _calculate_wall_potential(player: Player, cfg: RewardConfig) -> float:
	var pid: int = player.player_id
	var collision_distances := _get_collision_distances(player, pid)

	if collision_distances.is_empty():
		return 0.0

	var vision_radius: float = _play_scene.vision_sensor.vision_radius if _play_scene.vision_sensor else 250.0

	match cfg.wall_potential_calculate_mode:
		RewardConfig.WallPotentialCalculateMode.MIN:
			return _calculate_min_potential(collision_distances, cfg, vision_radius)
		RewardConfig.WallPotentialCalculateMode.WEIGHTED_AVERAGE:
			return _calculate_weighted_average_potential(collision_distances, cfg, vision_radius)
		RewardConfig.WallPotentialCalculateMode.AVERAGE:
			return _calculate_average_potential(collision_distances, cfg, vision_radius)

	return 0.0

# 计算最小距离的势能
func _calculate_min_potential(distances: Array[float], cfg: RewardConfig, vision_radius: float) -> float:
	var min_distance_normalized: float = distances.reduce(func(a, b): return min(a, b), 1.0)
	var d_min: float = min_distance_normalized * vision_radius
	
	match cfg.wall_potential_mode:
		RewardConfig.WallPotentialMode.LINEAR:
			return (cfg.wall_collision_penalty / vision_radius) * d_min - cfg.wall_collision_penalty
		
		RewardConfig.WallPotentialMode.INVERSE:
			var epsilon: float = 1.0 / cfg.wall_collision_penalty
			return -1.0 / (d_min + epsilon)
		
		# RewardConfig.WallPotentialMode.EXP:
	
	return 0.0

# 计算加权平均势能
func _calculate_weighted_average_potential(distances: Array[float], cfg: RewardConfig, vision_radius: float) -> float:
	if distances.is_empty():
		return 0.0
	
	# 计算权重（1-距离）
	var weights:Array[float] = []
	var total_weight: float = 0.0
	for distance in distances:
		var weight: float = 1.0 - distance
		weights.append(weight)
		total_weight += weight
	
	# 归一化权重
	if total_weight <= 0.0:
		return 0.0
	
	var weighted_sum: float = 0.0
	for i in range(distances.size()):
		var normalized_weight: float = weights[i] / total_weight
		var potential: float = _calculate_single_potential(distances[i], cfg, vision_radius)
		weighted_sum += normalized_weight * potential
	
	return weighted_sum

# 计算平均势能
func _calculate_average_potential(distances: Array[float], cfg: RewardConfig, vision_radius: float) -> float:
	if distances.is_empty():
		return 0.0
	
	var total_potential: float = 0.0
	for distance in distances:
		total_potential += _calculate_single_potential(distance, cfg, vision_radius)
	
	return total_potential / distances.size()

# 计算单个距离的势能值
func _calculate_single_potential(distance_normalized: float, cfg: RewardConfig, vision_radius: float) -> float:
	var d_actual: float = distance_normalized * vision_radius
	
	match cfg.wall_potential_mode:
		RewardConfig.WallPotentialMode.LINEAR:
			return (cfg.wall_collision_penalty / vision_radius) * d_actual - cfg.wall_collision_penalty
		
		RewardConfig.WallPotentialMode.INVERSE:
			var epsilon: float = 1.0 / cfg.wall_collision_penalty
			return -1.0 / (d_actual + epsilon)
	
	return 0.0

# 视野内所有活跃球的势能之和
func calculate_ball_potential_all(player: Player) -> float:
	if _play_scene == null or _play_scene.reward_ball_manager == null:
		return 0.0

	var cfg := _cfg(player.player_id)
	var player_pos := player.global_position
	var ball_manager: RewardBallManager = _play_scene.reward_ball_manager
	var vision_radius: float = _play_scene.vision_sensor.vision_radius
	var total_potential: float = 0.0

	for ball in ball_manager.reward_balls:
		if not is_instance_valid(ball) or not ball.is_active:
			continue

		var dist: float = player_pos.distance_to(ball.global_position)
		if dist > vision_radius:
			continue

		var ball_reward: float
		if ball in ball_manager.type_a_balls:
			ball_reward = cfg.collect_ball_A
		elif ball in ball_manager.type_b_balls:
			ball_reward = cfg.collect_ball_B
		else:
			continue

		total_potential += _calculate_single_ball_potential(dist, ball_reward, vision_radius, cfg.ball_potential_scale, cfg.ball_potential_func)

	return total_potential

# 最近球势能
func calculate_ball_potential(player: Player) -> float:
	if _play_scene == null or _play_scene.reward_ball_manager == null:
		return 0.0

	var cfg := _cfg(player.player_id)
	var player_pos := player.global_position
	var ball_manager: RewardBallManager = _play_scene.reward_ball_manager
	var vision_radius:float=_play_scene.vision_sensor.vision_radius

	var nearest_ball: RewardBall = null
	var min_dist: float = INF
	for ball in ball_manager.reward_balls:
		if not is_instance_valid(ball) or not ball.is_active:
			continue

		var dist: float = player_pos.distance_to(ball.global_position)
		if dist < min_dist and dist<=vision_radius:
			min_dist = dist
			nearest_ball = ball

	if nearest_ball == null:
		return 0.0

	var ball_reward: float
	if nearest_ball in ball_manager.type_a_balls:
		ball_reward = cfg.collect_ball_A
	elif nearest_ball in ball_manager.type_b_balls:
		ball_reward = cfg.collect_ball_B
	else:
		return 0.0

	#if player.player_id==0:
			#print(min_dist)
	return _calculate_single_ball_potential(min_dist, ball_reward, vision_radius, cfg.ball_potential_scale, cfg.ball_potential_func)


# 根据配置的势能函数类型，计算单个球的势能值
func _calculate_single_ball_potential(dist: float, ball_reward: float, vision_radius: float, scale: float, func_type: RewardConfig.BallPotentialFunc) -> float:
	match func_type:
		RewardConfig.BallPotentialFunc.LINEAR:
			# Φ = R_ball - (R_ball / r_vision) * d
			return scale * maxf(0.0, ball_reward - ball_reward / vision_radius * dist)
		RewardConfig.BallPotentialFunc.EXPONENTIAL:
			# Φ = R_ball * exp(-d / r_vision)
			return scale * ball_reward * exp(-dist / vision_radius)
		RewardConfig.BallPotentialFunc.INVERSE:
			# Φ = R_ball * r_vision / (d + r_vision)
			return scale * ball_reward * vision_radius / (dist + vision_radius)
	return 0.0


# 获取玩家到最近活跃球的距离（用于方案4距离差计算）
func _get_nearest_ball_distance(player: Player) -> float:
	if _play_scene == null or _play_scene.reward_ball_manager == null:
		return INF

	var player_pos := player.global_position
	var ball_manager: RewardBallManager = _play_scene.reward_ball_manager
	var vision_radius: float = _play_scene.vision_sensor.vision_radius if _play_scene.vision_sensor else 250.0

	var min_dist: float = INF
	for ball in ball_manager.reward_balls:
		if not is_instance_valid(ball) or not ball.is_active:
			continue
		var dist: float = player_pos.distance_to(ball.global_position)
		if dist <= vision_radius and dist < min_dist:
			min_dist = dist

	return min_dist

## 中央区域塑形奖励：鼓励智能体进入竞技场中心
func _process_center_shaping(delta: float) -> void:
	if _play_scene == null:
		return

	var arena_center: Vector2 = Vector2.ZERO

	for player in _play_scene.players:
		if player.is_dead:
			continue

		var cfg := _cfg(player.player_id)
		var dist_to_center: float = player.global_position.distance_to(arena_center)
		#乘2，在边界奖励为0
		var center_reward: float = cfg.center_reward_scale * maxf(0.0, 1.0 - 2*dist_to_center / _play_scene.arena_length)
		# 直接修改 AIController 的 reward
		player.ai_controller.reward += center_reward * delta

## ── 撞墙惩罚 ──
func _process_wall_collision(_delta: float) -> void:
	if _play_scene == null:
		return

	for player:Player in _play_scene.players:
		if player.is_dead:
			continue

		var pid: int = player.player_id
		var cfg := _cfg(pid)

		# 撞墙惩罚
		var is_wall_collision := false
		if player.last_collison_data and player.is_moving:
			if player.last_collison_data.get_collider() is TileMapLayer:
				add_reward(pid, -cfg.wall_collision_penalty, "wall_collision")
				is_wall_collision = true

		# 撞墙计数器与 reset_on_wall
		if game_config != null and game_config.reset_on_wall:
			if is_wall_collision:
				_wall_collision_counters[pid] = _wall_collision_counters.get(pid, 0) + 1
				if _wall_collision_counters[pid] >= game_config.wall_reset_threshold:
					player.ai_controller.needs_reset = true
			else:
				_wall_collision_counters[pid] = 0

		# 方案三距离惩罚
		if cfg.wall_potential_mode == RewardConfig.WallPotentialMode.COLLISION:
			var collision_distances := _get_collision_distances(player, pid)
			if collision_distances.is_empty():
				continue

			var min_distance_normalized: float = collision_distances.reduce(func(a, b): return min(a, b), 1.0)
			var vision_radius: float = _play_scene.vision_sensor.vision_radius if _play_scene.vision_sensor else 250.0
			var d_min: float = min_distance_normalized * vision_radius
			var epsilon: float = 1.0 / cfg.wall_collision_penalty
			var rd: float = -1.0 / (d_min + epsilon)
			add_reward(pid, rd, "wall_distance")

## ── 重置 ──

#游戏重置时调用
func reset() -> void:
	if _reward_logger != null:
		_reward_logger.end_episode()
		_reward_logger.start_episode()

	action_repeat_count=0
	first_frame=true
	_init_starvation_timers()
	_init_potentials()
	_pure_rewards.clear()
	_prev_ball_distances.clear()
	_wall_collision_counters.clear()
