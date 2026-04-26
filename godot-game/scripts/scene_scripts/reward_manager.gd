extends Node
class_name RewardManager

" 奖励管理器
 统一管理所有 RL 奖励的发放，奖励计算不嵌入各个游戏逻辑代码
 读取 configs/reward_config.tres 了解各个事件的奖励数值
 提供奖励变更接口（如 add_reward），各模块调用即可进行奖励更改

 监听的信号：
   - EventBus.entity_damaged  受伤/造成伤害奖励
   - EventBus.enemy_died      击杀敌人奖励
   - EventBus.player_died     击杀玩家/死亡惩罚
   - EventBus.reward_ball_collected → 拾取奖励球奖励"

@export var _sync_node:Sync
@export var game_config:GameConfig
@export var reward_config:RewardConfig
#奖励常量
var COLLECT_BALL_A: float 
var COLLECT_BALL_B: float
var BEAR_DAMAGE: float 
var CAUSE_DAMAGE_TO_ENEMY: float 
var CAUSE_DAMAGE_TO_PLAYER: float 
var KILL_ENEMY: float 
var KILL_PLAYER: float
var RUN: float 
var ATTACK: float
var DIED: float
var STARVE_TIME: float 
var MAX_STARVE_DURATION:float
var STARVE_REWARD_DECREASE: float 
var STARVE_MORE_FUNC: int

#塑形奖励常量
var BALL_POTENTIAL_SCALE: float      #球吸引势能缩放系数
var CENTER_REWARD_SCALE: float        #离竞技场中心越近奖励越大（每帧）
var BALL_POTENTIAL_MODE: int = reward_config.BallPotentialMode.NEAREST  # 球势能计算模式枚举

#撞墙惩罚常量
var WALL_COLLISION_PENALTY: float  #撞墙时移动惩罚

# 墙壁势能塑形模式枚举：NONE, LINEAR, INVERSE, COLLISION
var WALL_POTENTIAL_MODE: int = reward_config.WallPotentialMode.NONE

var _play_scene: PlayScene = null
var _reward_logger: RewardLogger = null

#每个玩家的"上次获得正奖励"的游戏时间,用于饥饿机制
#key: player_id, value: 游戏时间（秒，受 Engine.time_scale 影响）
var _last_reward_time: Dictionary = {}

#纯奖励累计值（不含塑形奖励），用于分数榜展示
#key: player_id, value: 累计纯奖励值
var _pure_rewards: Dictionary = {}

#累计游戏时间（受 Engine.time_scale 影响），用于饥饿计时
var _game_time: float = 0.0

# ── 势能塑形系统 ──
var _prev_potentials: Dictionary = {}  # {player_id: 上帧总势能}
var _shaping_gamma: float=0.99       # 塑形折扣因子
var action_repeat_count:int=0

func _ready() -> void:
	_play_scene = get_parent() if get_parent() is PlayScene else null
	_load_reward_config()
	_connect_signals()

func _exit_tree() -> void:
	_disconnect_signals()
	_disconnect_skill_signals()
	if _reward_logger != null:
		_reward_logger.flush()

func _physics_process(delta: float) -> void:
	_game_time += delta
	_process_starvation(delta)
	
	action_repeat_count=(action_repeat_count+1)%_sync_node.action_repeat
	if action_repeat_count==0:
		_process_potential_shaping(delta)
		_process_wall_collision(delta)
		for player in _play_scene.players:
			if player.is_moving:
				on_player_moved(player)
		
	_process_center_shaping(delta)
	
#从 JSON 文件加载奖励配置
func _load_reward_config() -> void:
	COLLECT_BALL_A = reward_config.collect_ball_A
	COLLECT_BALL_B = reward_config.collect_ball_B
	BEAR_DAMAGE = reward_config.bear_damage
	CAUSE_DAMAGE_TO_ENEMY = reward_config.cause_damage_to_enemy
	CAUSE_DAMAGE_TO_PLAYER = reward_config.cause_damage_to_player
	KILL_ENEMY = reward_config.kill_enemy
	KILL_PLAYER = reward_config.kill_player
	RUN = reward_config.run
	ATTACK = reward_config.attack
	DIED = reward_config.died
	STARVE_TIME = reward_config.starve_time
	MAX_STARVE_DURATION = reward_config.max_starve_duration
	STARVE_REWARD_DECREASE = reward_config.starve_reward_decrease
	STARVE_MORE_FUNC = reward_config.starve_more_func
	BALL_POTENTIAL_SCALE = reward_config.ball_potential_scale
	BALL_POTENTIAL_MODE = reward_config.ball_potential_mode
	CENTER_REWARD_SCALE = reward_config.center_reward_scale
	WALL_COLLISION_PENALTY = reward_config.wall_collision_penalty
	WALL_POTENTIAL_MODE = reward_config.wall_potential_mode
	

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
	# 从 Sync 节点获取 gamma（由 Python --gamma 参数传入）
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
		# bear_damage
		add_reward(target_player.player_id, BEAR_DAMAGE, "bear_damage")

	# 攻击者：如果攻击者是玩家
	if source is Player:
		var source_player := source as Player
		# 被击者是敌人
		if entity is Enemy:
			add_reward(source_player.player_id, CAUSE_DAMAGE_TO_ENEMY, "cause_damage_to_enemy")
		# 被击者是玩家
		elif entity is Player and entity != source:
			add_reward(source_player.player_id, CAUSE_DAMAGE_TO_PLAYER, "cause_damage_to_player")

## 敌人死亡处理
## @param enemy 死亡的敌人
func _on_enemy_died(enemy: Enemy) -> void:
	if _play_scene == null:
		return

	# 检查敌人记录的最后伤害来源，如果是玩家则给予击杀奖励
	if enemy.last_damage_source is Player:
		var killer := enemy.last_damage_source as Player
		add_reward(killer.player_id, KILL_ENEMY, "kill_enemy")

## 玩家死亡处理
## @param player 死亡的玩家
func _on_player_died(player: Player) -> void:
	if _play_scene == null:
		return

	# 死亡惩罚
	add_reward(player.player_id, DIED, "died")

	# 检查击杀者
	if player.last_damage_source is Player:
		var killer := player.last_damage_source as Player
		add_reward(killer.player_id, KILL_PLAYER, "kill_player")

#奖励球拾取处理,ball实例由奖励球管理器处理，此处不需要
func _on_reward_ball_collected(player_id: int, ball_type: int, _ball: RewardBall) -> void:
	if ball_type == RewardBall.BallType.TYPE_A:
		add_reward(player_id, COLLECT_BALL_A, "collect_ball_A")
	elif ball_type == RewardBall.BallType.TYPE_B:
		add_reward(player_id, COLLECT_BALL_B, "collect_ball_B")

#玩家攻击惩罚,不同skill的惩罚力度不一样，但目前只有一个技能
func _on_player_skill_activated(entity: Entity, _skill: Skill) -> void:
	if entity is Player:
		var player := entity as Player
		add_reward(player.player_id, ATTACK, "attack")

## ── 每帧执行的持续奖励逻辑 ──

#移动惩罚
func on_player_moved(player: Player) -> void:
	add_reward(player.player_id, RUN, "run")

#饥饿机制：长时间未获得正奖励的玩家，奖励逐渐减少
func _process_starvation(delta: float) -> void:
	if _play_scene == null:
		return

	for player in _play_scene.players:
		if player.is_dead:
			continue
		var starve_duration=compute_starve_duration(player)
		if starve_duration>0.0:
			# 根据增长函数计算衰减倍率
			var multiplier: float = MathUtils.starve_rate_multiplier(starve_duration, STARVE_MORE_FUNC)
			var decrease: float = STARVE_REWARD_DECREASE * multiplier * delta
			#if player.player_id==0:
				#print(decrease)
			add_reward(player.player_id, -decrease, "starvation")

#计算饥饿时间
func compute_starve_duration(player:Player)->float:
	var pid: int = player.player_id
	var time_since_reward: float = _game_time - _last_reward_time[pid]
	if time_since_reward >= STARVE_TIME:
			# 饥饿时间 = time_since_reward - STARVE_TIME
			var starve_duration:float= min(time_since_reward - STARVE_TIME,MAX_STARVE_DURATION)
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
	_prev_potentials.clear()
	if _play_scene == null:
		return
	for player in _play_scene.players:
		_prev_potentials[player.player_id] = calculate_total_potential(player)

# 计算玩家的总势能
# 根据 BALL_POTENTIAL_MODE 选择使用最近球势能或所有球势能之和
func calculate_total_potential(player: Player) -> float:
	var total_potential:float=0
	var ball_potential: float = 0.0
	
	# 根据模式计算球势能
	if BALL_POTENTIAL_MODE == reward_config.BallPotentialMode.ALL:
		ball_potential = calculate_ball_potential_all(player)
	else:
		#print("nearest")
		ball_potential = calculate_ball_potential(player)
	
	total_potential+=ball_potential
	
	# 添加墙壁势能（方案一和方案二）
	if WALL_POTENTIAL_MODE in [reward_config.WallPotentialMode.LINEAR, reward_config.WallPotentialMode.INVERSE]:
		var wall_potential: float = calculate_wall_potential(player)
		total_potential += wall_potential
	
	return total_potential

# 墙壁势能计算,根据最小射线距离计算墙壁势能
func calculate_wall_potential(player: Player) -> float:
	if _play_scene == null:
		return 0.0
	
	var pid: int = player.player_id
	
	# 从 PlayScene 的 last_map_states 缓存中获取上一帧的 map_state
	# 注意：这里获取的是上一帧的 map_state，用于势能计算
	var map_state:Array = _play_scene.last_map_states.get(pid, [])
	
	if map_state.size() == 0:
		map_state=_play_scene._build_map_state(player)
		if map_state.size()==0:
			print("map empty")
			return 0.0
	
	# 找到最小的射线距离（归一化，1.0表示无碰撞）
	var min_distance_normalized: float = 1.0
	for distance in map_state:
		if distance < min_distance_normalized:
			min_distance_normalized = distance
	
	# 如果最小距离为1.0（无碰撞），势能为0
	if min_distance_normalized >= 1.0:
		return 0.0
	
	# 将归一化距离转换为实际距禿
	var vision_radius: float = _play_scene.vision_sensor.vision_radius if _play_scene.vision_sensor else 250.0
	var d_min: float = min_distance_normalized * vision_radius
	
	if WALL_POTENTIAL_MODE == reward_config.WallPotentialMode.LINEAR:
		# 方案一 线性函数 
		#print("linear")
		var potential: float = (WALL_COLLISION_PENALTY / vision_radius) * d_min - WALL_COLLISION_PENALTY
		return potential
	
	elif WALL_POTENTIAL_MODE == reward_config.WallPotentialMode.INVERSE:
		# 方案二：反比例函数 
		#print("invp")
		var epsilon: float = 1.0 / WALL_COLLISION_PENALTY
		var potential: float = -1.0 / (d_min + epsilon)
		return potential
	
	return 0.0

# 球吸引势能计算视野内所有活跃球的势能之和
func calculate_ball_potential_all(player: Player) -> float:
	if _play_scene == null or _play_scene.reward_ball_manager == null:
		return 0.0
	
	var player_pos := player.global_position
	var ball_manager: RewardBallManager = _play_scene.reward_ball_manager
	var vision_radius: float = _play_scene.vision_sensor.vision_radius
	var total_potential: float = 0.0
	
	for ball in ball_manager.reward_balls:
		if not is_instance_valid(ball) or not ball.is_active:
			continue
		
		var dist: float = player_pos.distance_to(ball.global_position)
		# 只计算视野内的球
		if dist > vision_radius:
			continue
		
		# 线性势函数
		if ball in ball_manager.type_a_balls:
			var potential: float = BALL_POTENTIAL_SCALE * maxf(0.0, COLLECT_BALL_A - COLLECT_BALL_A / vision_radius * dist)
			total_potential += potential
		elif ball in ball_manager.type_b_balls:
			var potential: float = BALL_POTENTIAL_SCALE * maxf(0.0, COLLECT_BALL_B - COLLECT_BALL_B / vision_radius * dist)
			total_potential += potential
	
	return total_potential

# 球吸引势能：距离最近活跃球越近，势能越高 
func calculate_ball_potential(player: Player) -> float:
	if _play_scene == null or _play_scene.reward_ball_manager == null:
		return 0.0
	
	var player_pos := player.global_position
	var ball_manager: RewardBallManager = _play_scene.reward_ball_manager
	
	var nearest_ball: RewardBall = null
	var min_dist: float = INF
	var vision_radius:float=_play_scene.vision_sensor.vision_radius
	for ball in ball_manager.reward_balls:
		if not is_instance_valid(ball) or not ball.is_active:
			continue
		
		var dist: float = player_pos.distance_to(ball.global_position)
		if dist < min_dist and dist<=vision_radius:
			min_dist = dist
			nearest_ball = ball
	
	if nearest_ball == null:
		return 0.0

	#线性函数，终止非0
	if nearest_ball in ball_manager.type_a_balls:
		return BALL_POTENTIAL_SCALE * maxf(0.0, COLLECT_BALL_A - COLLECT_BALL_A / vision_radius*min_dist)
	elif nearest_ball in ball_manager.type_b_balls:
		return BALL_POTENTIAL_SCALE * maxf(0.0, COLLECT_BALL_B - COLLECT_BALL_B / vision_radius*min_dist)
	
	return 0.0

func _process_potential_shaping(_delta: float) -> void:
	if _play_scene == null:
		return
	
	for player in _play_scene.players:
		if player.is_dead:
			continue
		
		var pid := player.player_id
		var current_potential := calculate_total_potential(player)
		var prev_potential :float= _prev_potentials.get(pid, 0.0)
		
		var shaping :float= _shaping_gamma * current_potential - prev_potential
		
		# 直接写入 AIController.reward
		player.ai_controller.reward += shaping
		
		# 缓存当前势能为下一帧使用
		_prev_potentials[pid] = current_potential

## 中央区域塑形奖励：鼓励智能体进入竞技场中心
func _process_center_shaping(delta: float) -> void:
	if _play_scene == null:
		return
	
	var arena_center: Vector2 = Vector2.ZERO
	
	for player in _play_scene.players:
		if player.is_dead:
			continue
		
		var dist_to_center: float = player.global_position.distance_to(arena_center)
		#乘2，在边界奖励为0
		var center_reward: float = CENTER_REWARD_SCALE * maxf(0.0, 1.0 - 2*dist_to_center / _play_scene.arena_length)
		# 直接修改 AIController 的 reward
		player.ai_controller.reward += center_reward * delta

## ── 撞墙惩罚 ──
func _process_wall_collision(_delta: float) -> void:
	#print(WALL_POTENTIAL_MODE == reward_config.WallPotentialMode.COLLISION)
	if _play_scene == null:
		return

	for player:Player in _play_scene.players:
		if player.is_dead:
			continue

		var pid: int = player.player_id

		# 撞墙惩罚
		if player.last_collison_data and player.is_moving:
			if player.last_collison_data.get_collider() is TileMapLayer:
				add_reward(pid, -WALL_COLLISION_PENALTY, "wall_collision")  
		
		# 方案三距离惩罚
		if WALL_POTENTIAL_MODE == reward_config.WallPotentialMode.COLLISION:
			var map_state:Array = _play_scene.last_map_states.get(pid, [])
			if map_state.size()==0:
				map_state=_play_scene._build_map_state(player)
				if map_state.size()==0:
					print("map_state empty")
					break
			# 找到最小的射线距离（归一化，1.0=无碰撞）
			var min_distance_normalized: float = 1.0
			for distance in map_state:
				if distance < min_distance_normalized:
					min_distance_normalized = distance
			
			var epsilon: float = 1.0 / WALL_COLLISION_PENALTY
			var d_min: float = min_distance_normalized * (_play_scene.vision_sensor.vision_radius if _play_scene.vision_sensor else 250.0)
			var rd: float = -1.0 / (d_min + epsilon)  
			
			#if pid==0:
				#print("add ",rd)
			add_reward(pid, rd, "wall_distance")

## ── 重置 ──

#游戏重置时调用
func reset() -> void:
	if _reward_logger != null:
		_reward_logger.end_episode()
		_reward_logger.start_episode()
	
	action_repeat_count=0
	_init_starvation_timers()
	_init_potentials()
	_pure_rewards.clear()
