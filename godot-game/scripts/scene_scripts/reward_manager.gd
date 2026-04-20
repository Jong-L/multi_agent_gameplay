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
var PROXIMITY_TO_BALL_SCALE: float   #距离奖励：离球越近奖励越大（每帧）
var VELOCITY_TOWARD_BALL_SCALE: float   #朝向球移动时额外奖励（每帧）
var CENTER_REWARD_SCALE: float        #离竞技场中心越近奖励越大（每帧）

var _play_scene: PlayScene = null

#每个玩家的"上次获得正奖励"的游戏时间,用于饥饿机制
#key: player_id, value: 游戏时间（秒，受 Engine.time_scale 影响）
var _last_reward_time: Dictionary = {}

#累计游戏时间（受 Engine.time_scale 影响），用于饥饿计时
var _game_time: float = 0.0

#奖励配置文件路径
const REWARD_CONFIG_PATH: String = "res://configs/reward.json"

func _ready() -> void:
	_load_reward_config()
	_connect_signals()

func _exit_tree() -> void:
	_disconnect_signals()
	_disconnect_skill_signals()

func _process(delta: float) -> void:
	_game_time += delta
	_process_starvation(delta)
	_process_proximity_shaping(delta)
	_process_center_shaping(delta)

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
		PROXIMITY_TO_BALL_SCALE = float(data["proximity_to_ball_scale"])
	if data.has("velocity_toward_ball_scale"):
		VELOCITY_TOWARD_BALL_SCALE = float(data["velocity_toward_ball_scale"])
	if data.has("center_reward_scale"):
		CENTER_REWARD_SCALE = float(data["center_reward_scale"])

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

#初始化（由 PlayScene 调用）
func setup(play_scene: PlayScene) -> void:
	_play_scene = play_scene
	_init_starvation_timers()
	_connect_skill_signals()

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
func add_reward(player_id: int, value: float) -> void:
	if _play_scene == null:
		return
	if player_id < 0 or player_id >= _play_scene.players.size():
		return

	var player := _play_scene.players[player_id]
	player.ai_controller.reward += value

	# 正奖励时刷新饥饿计时器
	if value > 0.1:
		_last_reward_time[player_id] = _game_time

## ── 事件处理 ──

#实体受伤处理
func _on_entity_damaged(entity: Entity, source: Entity) -> void:
	if _play_scene == null:
		return

	# 被击者：如果受伤的是玩家，给予受伤惩罚
	if entity is Player:
		var target_player := entity as Player
		# bear_damage
		add_reward(target_player.player_id, BEAR_DAMAGE)

	# 攻击者：如果攻击者是玩家
	if source is Player:
		var source_player := source as Player
		# 被击者是敌人
		if entity is Enemy:
			add_reward(source_player.player_id, CAUSE_DAMAGE_TO_ENEMY)
		# 被击者是玩家
		elif entity is Player and entity != source:
			add_reward(source_player.player_id, CAUSE_DAMAGE_TO_PLAYER)

## 敌人死亡处理
## @param enemy 死亡的敌人
func _on_enemy_died(enemy: Enemy) -> void:
	if _play_scene == null:
		return

	# 检查敌人记录的最后伤害来源，如果是玩家则给予击杀奖励
	if enemy.last_damage_source is Player:
		var killer := enemy.last_damage_source as Player
		add_reward(killer.player_id, KILL_ENEMY)

## 玩家死亡处理
## @param player 死亡的玩家
func _on_player_died(player: Player) -> void:
	if _play_scene == null:
		return

	# 死亡惩罚
	add_reward(player.player_id, DIED)

	# 检查击杀者
	if player.last_damage_source is Player:
		var killer := player.last_damage_source as Player
		add_reward(killer.player_id, KILL_PLAYER)

#奖励球拾取处理,ball实例由奖励球管理器处理，此处不需要
func _on_reward_ball_collected(player_id: int, ball_type: int, _ball: RewardBall) -> void:
	if ball_type == RewardBall.BallType.TYPE_A:
		add_reward(player_id, COLLECT_BALL_A)
	elif ball_type == RewardBall.BallType.TYPE_B:
		add_reward(player_id, COLLECT_BALL_B)

#玩家攻击惩罚,不同skill的惩罚力度不一样，但目前只有一个技能
func _on_player_skill_activated(entity: Entity, _skill: Skill) -> void:
	if entity is Player:
		var player := entity as Player
		add_reward(player.player_id, ATTACK)

## ── 每帧执行的持续奖励逻辑 ──

#移动惩罚
func on_player_moved(player: Player) -> void:
	add_reward(player.player_id, RUN)

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
			add_reward(player.player_id, -decrease)

#计算饥饿时间
func compute_starve_duration(player:Player)->float:
	var pid: int = player.player_id
	var time_since_reward: float = _game_time - _last_reward_time[pid]
	if time_since_reward >= STARVE_TIME:
			# 饥饿时间 = time_since_reward - STARVE_TIME
			var starve_duration:float= min(time_since_reward - STARVE_TIME,MAX_STARVE_DURATION)
			return starve_duration
	return 0.0

## ── 塑形奖励（持续引导） ──

## 每帧计算塑形奖励：靠近奖励球 + 速度方向奖励
## 设计原则：Potential-Based Reward Shaping + 行为塑形
##   1. 弱距离奖励：仅作为位置引导，系数降低以避免鼓励待机
##   2. 速度方向奖励：朝向最近球移动时额外奖励，鼓励主动接近行为
##   球被清完后，塑形归零，智能体自然往中心游荡 → 触发战斗事件奖励
func _process_proximity_shaping(delta: float) -> void:
	if _play_scene == null:
		return

	var ball_manager: RewardBallManager = _play_scene.reward_ball_manager
	if ball_manager == null:
		return

	for player:Player in _play_scene.players:
		if player.is_dead:
			continue

		var player_pos: Vector2 = player.global_position
		#var player_vel: Vector2 = player.get_real_velocity()#采用“实际速度”，避免撞墙时还得到奖励
		var player_vel: Vector2=player.velocity #在逻辑帧中如果被挡住velocity会是0,因此用这个也行，被挡住了不会获得奖励

		# 找到最近活跃球 
		var nearest_ball: RewardBall = null
		var min_dist: float = INF
		for ball in ball_manager.reward_balls:
			var dist: float = player_pos.distance_to(ball.global_position)
			if not is_instance_valid(ball) or not ball.is_active or dist>_play_scene.vision_sensor.vision_radius:
				continue
			if dist < min_dist:
				min_dist = dist
				nearest_ball = ball

		var total_shaping: float = 0.0

		if nearest_ball != null:
			# 距离奖励
			var dist_reward: float = PROXIMITY_TO_BALL_SCALE * maxf(0.0, 1.0 - 4*min_dist / _play_scene.arena_length)
			total_shaping += dist_reward

			#速度方向奖励
			var to_ball_dir: Vector2 = (nearest_ball.global_position - player_pos).normalized()
			var vel_dir: Vector2 = player_vel.normalized()
			var dot: float = vel_dir.dot(to_ball_dir)  # -1~1
			if dot > 0.0:  # 朝向球移动
				var velocity_reward: float = VELOCITY_TOWARD_BALL_SCALE * dot
				total_shaping += velocity_reward
			#elif dot<0.0:#离开球时等量惩罚，防止反复靠近刷分
				#var velocity_reward: float = VELOCITY_TOWARD_BALL_SCALE * dot
				#total_shaping += velocity_reward

		# 直接修改reward,不走 add_reward 以避免刷新饥饿计时器
		player.ai_controller.reward += total_shaping * delta

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

## ── 重置 ──

#游戏重置时调用
func reset() -> void:
	_init_starvation_timers()
