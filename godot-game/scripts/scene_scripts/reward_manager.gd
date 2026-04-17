extends Node
class_name RewardManager

" 奖励管理器
 统一管理所有 RL 奖励的发放，将原本嵌入游戏代码中的奖励计算分离出来
 读取 configs/reward.json 了解各个事件的奖励数值
 提供奖励变更接口（如 add_reward），各模块调用即可进行奖励更改

 架构位置：PlayScene 子节点（与 RewardBallManager 同级）
 数据流：
   事件触发 → RewardManager 对应处理方法 → add_reward() → AIController2D.reward

 监听的信号：
   - EventBus.entity_damaged  → 受伤/造成伤害奖励
   - EventBus.enemy_died      → 击杀敌人奖励
   - EventBus.player_died     → 击杀玩家/死亡惩罚
   - EventBus.reward_ball_collected → 拾取奖励球奖励"

## ── 奖励常量（从 reward.json 加载） ──
var COLLECT_BALL_A: float = 1.0
var COLLECT_BALL_B: float = 1.5
var BEAR_DAMAGE: float = -1.0
var CAUSE_DAMAGE_TO_ENEMY: float = 1.0
var CAUSE_DAMAGE_TO_PLAYER: float = 1.5
var KILL_ENEMY: float = 3.0
var KILL_PLAYER: float = 4.5
var RUN: float = -0.01
var ATTACK: float = -0.5
var DIED: float = -2.0
var STARVE_TIME: float = 10.0
var STARVE_REWARD_DECREASE: float = 0.01
var STARVE_MORE_FUNC: String = "linear"

## PlayScene 引用
var _play_scene: PlayScene = null

## 每个玩家的"上次获得正奖励"时间戳（用于饥饿机制）
## key: player_id, value: 时间戳（秒）
var _last_reward_time: Dictionary = {}

## 奖励配置文件路径
const REWARD_CONFIG_PATH: String = "res://configs/reward.json"


func _ready() -> void:
	_load_reward_config()
	_connect_signals()


func _exit_tree() -> void:
	_disconnect_signals()
	_disconnect_skill_signals()


## ── 配置加载 ──

## 从 JSON 文件加载奖励配置
func _load_reward_config() -> void:
	if not FileAccess.file_exists(REWARD_CONFIG_PATH):
		push_warning("[RewardManager] 奖励配置文件不存在: %s, 使用默认值" % REWARD_CONFIG_PATH)
		return

	var file := FileAccess.open(REWARD_CONFIG_PATH, FileAccess.READ)
	if file == null:
		push_warning("[RewardManager] 无法打开奖励配置文件: %s" % REWARD_CONFIG_PATH)
		return

	var json_text := file.get_as_text()
	file.close()

	var json := JSON.new()
	var error := json.parse(json_text)
	if error != OK:
		push_warning("[RewardManager] JSON 解析错误: %s (行 %d)" % [json.get_error_message(), json.get_error_line()])
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
	if data.has("starve_reward_decrease"):
		STARVE_REWARD_DECREASE = float(data["starve_reward_decrease"])
	if data.has("starve_more_func"):
		STARVE_MORE_FUNC = str(data["starve_more_func"])


## ── 信号连接 ──

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


## ── 初始化（由 PlayScene 调用） ──

func setup(play_scene: PlayScene) -> void:
	_play_scene = play_scene
	_init_starvation_timers()
	_connect_skill_signals()


## 初始化饥饿计时器（所有玩家当前时间）
func _init_starvation_timers() -> void:
	_last_reward_time.clear()
	if _play_scene == null:
		return
	var current_time := Time.get_ticks_msec() / 1000.0
	for player in _play_scene.players:
		_last_reward_time[player.player_id] = current_time


## 连接所有玩家的 SkillController.skill_activated 信号
func _connect_skill_signals() -> void:
	if _play_scene == null:
		return
	for player in _play_scene.players:
		if player.skill_controller != null and not player.skill_controller.skill_activated.is_connected(_on_player_skill_activated):
			player.skill_controller.skill_activated.connect(_on_player_skill_activated)


## 断开所有玩家的 SkillController 信号
func _disconnect_skill_signals() -> void:
	if _play_scene == null:
		return
	for player in _play_scene.players:
		if is_instance_valid(player) and player.skill_controller != null and player.skill_controller.skill_activated.is_connected(_on_player_skill_activated):
			player.skill_controller.skill_activated.disconnect(_on_player_skill_activated)


## ── 核心接口：统一奖励发放 ──

## 给指定玩家增加奖励
## @param player_id 玩家 ID (0-3)
## @param value 奖励数值（正=奖励，负=惩罚）
func add_reward(player_id: int, value: float) -> void:
	if _play_scene == null:
		return
	if player_id < 0 or player_id >= _play_scene.players.size():
		return

	var player := _play_scene.players[player_id]
	player.ai_controller.reward += value

	# 正奖励时刷新饥饿计时器
	if value > 0.0:
		_last_reward_time[player_id] = Time.get_ticks_msec() / 1000.0


## ── 事件处理 ──

## 实体受伤处理
## @param entity 受击实体
## @param damage 伤害值
## @param source 伤害来源（可能为 null）
func _on_entity_damaged(entity: Entity, damage: float, source: Entity) -> void:
	if _play_scene == null:
		return

	# 被击者：如果受伤的是玩家，给予受伤惩罚
	if entity is Player:
		var target_player := entity as Player
		# bear_damage 奖励按伤害值比例缩放
		add_reward(target_player.player_id, BEAR_DAMAGE * damage / target_player.max_health)

	# 攻击者：如果攻击者是玩家
	if source is Player:
		var source_player := source as Player
		# 被击者是敌人 → cause_damage_to_enemy
		if entity is Enemy:
			add_reward(source_player.player_id, CAUSE_DAMAGE_TO_ENEMY)
		# 被击者是玩家 → cause_damage_to_player
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
		# 不给自己击杀奖励（自杀场景）
		if killer.player_id != player.player_id:
			add_reward(killer.player_id, KILL_PLAYER)


## 奖励球拾取处理
## @param player_id 拾取者玩家 ID
## @param ball_type RewardBall.BallType 枚举值
## @param reward_value 奖励数值
## @param ball 被拾取的奖励球实例
func _on_reward_ball_collected(player_id: int, ball_type: int, reward_value: float, ball: RewardBall) -> void:
	if ball_type == RewardBall.BallType.TYPE_A:
		add_reward(player_id, COLLECT_BALL_A)
	elif ball_type == RewardBall.BallType.TYPE_B:
		add_reward(player_id, COLLECT_BALL_B)


## 玩家技能成功激活处理（攻击惩罚）
## @param entity 激活技能的实体
## @param skill 被激活的技能
func _on_player_skill_activated(entity: Entity, skill: Skill) -> void:
	if entity is Player:
		var player := entity as Player
		add_reward(player.player_id, ATTACK)


## ── 持续奖励逻辑（每帧执行） ──

## 移动惩罚：玩家移动时每帧扣减
## @param player 移动的玩家
func on_player_moved(player: Player) -> void:
	add_reward(player.player_id, RUN)


## 饥饿机制：长时间未获得正奖励的玩家，奖励逐渐减少
## 在 _process 中每帧调用
func _process_starvation(delta: float) -> void:
	if _play_scene == null:
		return

	var current_time := Time.get_ticks_msec() / 1000.0

	for player in _play_scene.players:
		if player.is_dead:
			continue

		var pid: int = player.player_id
		if not _last_reward_time.has(pid):
			_last_reward_time[pid] = current_time
			continue

		var time_since_reward: float = current_time - _last_reward_time[pid]
		if time_since_reward >= STARVE_TIME:
			# 饥饿时间 = time_since_reward - STARVE_TIME
			var starve_duration: float = time_since_reward - STARVE_TIME
			# 根据增长函数计算衰减倍率
			var multiplier: float = MathUtils.starve_rate_multiplier(starve_duration, STARVE_MORE_FUNC)
			var decrease: float = STARVE_REWARD_DECREASE * multiplier * delta
			add_reward(pid, -decrease)


func _process(delta: float) -> void:
	_process_starvation(delta)


## ── 重置 ──

## 游戏重置时调用
func reset() -> void:
	_init_starvation_timers()
