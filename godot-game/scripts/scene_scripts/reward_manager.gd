extends Node
class_name RewardManager

" 奖励管理器
 统一管理所有 RL 奖励的发放，奖励计算不嵌入各个游戏逻辑代码
 读取 configs/reward.json 了解各个事件的奖励数值
 提供奖励变更接口（如 add_reward），各模块调用即可进行奖励更改

 监听的信号：
   - EventBus.entity_damaged  受伤/造成伤害奖励
   - EventBus.enemy_died      击杀敌人奖励
   - EventBus.player_died     击杀玩家/死亡惩罚
   - EventBus.reward_ball_collected → 拾取奖励球奖励"

@export var _sync_node:Sync
@export var game_config:GameConfig
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
var STARVE_MORE_FUNC: String

#塑形奖励常量
var BALL_POTENTIAL_SCALE: float      #球吸引势能缩放系数
var CENTER_REWARD_SCALE: float        #离竞技场中心越近奖励越大（每帧）
var BALL_POTENTIAL_MODE: String = "nearest"  # 球势能计算模式："nearest"(最近球) 或 "all"(所有视野内球)

#撞墙惩罚常量
var WALL_COLLISION_PENALTY: float  #撞墙时移动惩罚

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

#奖励配置文件路径
const REWARD_CONFIG_PATH: String = "res://configs/reward.json"

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
		
	_process_center_shaping(delta)
	
	for player in _play_scene.players:
		if player.is_moving:
			on_player_moved(player)

#从 JSON 文件加载奖励配置
func _load_reward_config() -> void:
	if not FileAccess.file_exists(REWARD_CONFIG_PATH):
		print("[RewardManager] 奖励配置文件不存在: %s, 使用默认值" % REWARD_CONFIG_PATH)
		return

	var file := FileAccess.open(REWARD_CONFIG_PATH, FileAccess.READ)
	if file == null:
		print("[RewardManager] 无法打开奖励配置文件: %s" % REWARD_CONFIG_PATH)
		return

	var json_text := file.get_as_text()
	file.close()

	var json := JSON.new()
	var error := json.parse(json_text)
	if error != OK:
		print("[RewardManager] JSON 解析错误: %s (行 %d)" % [json.get_error_message(), json.get_error_line()])
		return

	var data: Dictionary = json.data

	# 按键映射到常量
	if data.has("collect_ball_A"):
		COLLECT_BALL_A = float(data["collect_ball_A"])
	if data.has("collect_ball_B"):
		COLLECT_BALL_B = float(data["collect_ball_B"])
	if data.has("bear_damage"):
		BEAR_DAMAGE = float(data["bear_damage"])
	if data.has("cause_damage_to_enemy"):
		CAUSE_DAMAGE_TO_ENEMY = float(data["cause_damage_to_enemy"])
	if data.has("cause_damage_to_player"):
		CAUSE_DAMAGE_TO_PLAYER = float(data["cause_damage_to_player"])
	if data.has("kill_enemy"):
		KILL_ENEMY = float(data["kill_enemy"])
	if data.has("kill_player"):
		KILL_PLAYER = float(data["kill_player"])
	if data.has("run"):
		RUN = float(data["run"])
	if data.has("attack"):
		ATTACK = float(data["attack"])
	if data.has("died"):
		DIED = float(data["died"])
	if data.has("starve_time"):
		STARVE_TIME = float(data["starve_time"])
	if data.has("max_starve_duration"):
		MAX_STARVE_DURATION=float(data["max_starve_duration"])
	if data.has("starve_reward_decrease"):
		STARVE_REWARD_DECREASE = float(data["starve_reward_decrease"])
	if data.has("starve_more_func"):
		STARVE_MORE_FUNC = str(data["starve_more_func"])
	if data.has("proximity_to_ball_scale"):
		BALL_POTENTIAL_SCALE = float(data["proximity_to_ball_scale"])
	if data.has("ball_potential_scale"):
		BALL_POTENTIAL_SCALE = float(data["ball_potential_scale"])
	if data.has("ball_potential_mode"):
		BALL_POTENTIAL_MODE = str(data["ball_potential_mode"])
	if data.has("center_reward_scale"):
		CENTER_REWARD_SCALE = float(data["center_reward_scale"])
	if data.has("wall_collision_penalty"):
		WALL_COLLISION_PENALTY = float(data["wall_collision_penalty"])

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
	var ball_potential: float
	if BALL_POTENTIAL_MODE == "all":
		ball_potential = calculate_ball_potential_all(player)
	else:
		ball_potential = calculate_ball_potential(player)
	
	total_potential+=ball_potential
	return total_potential

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
	# 指数函数
	if nearest_ball in ball_manager.type_a_balls:
		return BALL_POTENTIAL_SCALE * COLLECT_BALL_A * exp(-min_dist / vision_radius)
	elif nearest_ball in ball_manager.type_b_balls:
		return BALL_POTENTIAL_SCALE * COLLECT_BALL_B * exp(-min_dist / vision_radius)
	
	#线性函数
	#if nearest_ball in ball_manager.type_a_balls:
		#return BALL_POTENTIAL_SCALE * maxf(0.0, COLLECT_BALL_A - COLLECT_BALL_A / vision_radius*min_dist)
	#elif nearest_ball in ball_manager.type_b_balls:
		#return BALL_POTENTIAL_SCALE * maxf(0.0, COLLECT_BALL_B - COLLECT_BALL_B / vision_radius*min_dist)
	
	#反比例函数
	#if nearest_ball in ball_manager.type_a_balls:
		#return BALL_POTENTIAL_SCALE * COLLECT_BALL_A *min_dist/(min_dist+vision_radius)
	#elif nearest_ball in ball_manager.type_b_balls:
		#return BALL_POTENTIAL_SCALE * COLLECT_BALL_B *min_dist/(min_dist+vision_radius)
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
# 每帧检测玩家撞墙并应用惩罚
func _process_wall_collision(_delta: float) -> void:
	if _play_scene == null:
		return

	for player in _play_scene.players:
		if player.is_dead:
			continue

		var pid: int = player.player_id

		# 检测是否撞墙（包括左右墙、地板、天花板），并判断移动方向是否导致碰撞
		var should_penalize: bool = false
		var player_movement: Vector2 = player.movement
		
		if player.is_on_wall_only():
			# 撞左右墙：只有水平移动（向左/右）才惩罚
			if abs(player_movement.x) > 0.0:  # 有水平输入
				should_penalize = true
		elif player.is_on_floor_only():
			# 撞地板（下边界）：只有向下移动才惩罚
			if player_movement.y > 0.0:  # 向下输入
				should_penalize = true
		elif player.is_on_ceiling_only():
			# 撞天花板（上边界）：只有向上移动才惩罚
			if player_movement.y < -0.0:  # 向上输入
				should_penalize = true

		if should_penalize and player.is_moving:
			# 撞墙
			var wall_penalty: float = WALL_COLLISION_PENALTY
			add_reward(pid, -wall_penalty, "wall_collision")

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
