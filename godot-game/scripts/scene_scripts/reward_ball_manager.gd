extends Node
class_name RewardBallManager

## 奖励球管理器
## 负责 A/B 类球的生成、状态追踪、奖励发放、观测数据
## 由 PlayScene 在 _ready() 中创建并添加为子节点
##
## A类球：4个玩家各3个=12个，出生点所在角的子区域内生成，不重生
## B类球：巡逻区域内最多5个，8秒后重生（位置由 Manager 决定）

const BALL_A_PER_PLAYER: int = 3
const BALL_B_MAX_COUNT: int = 5
const BALL_A_REWARD: float = 1.0
const BALL_B_REWARD: float = 1.5
const BALL_B_RESPAWN_DELAY: float = 8.0
const BALL_A_SPAWN_MARGIN: float = 30.0   ## A类球距子区域边缘的最小距离
const BALL_A_EXTENT_RATIO: float = 0.2    ## A类球子区域占竞技场尺寸的比例
const BALL_A_MIN_SPAWN_DIST: float = 40.0 ## A类球与玩家出生点的最小距离
const BALL_B_SPAWN_MARGIN: float = 10.0   ## B类球距巡逻区边缘的最小距离
const BALL_B_MIN_PLAYER_DIST: float = 60.0 ## B类球与任何玩家的最小距离
const BALL_B_MIN_DECO_DIST: float = 30.0   ## B类球与障碍物的最小距离
const _SAFE_POS_MAX_ATTEMPTS: int = 30     ## 安全位置生成的最大重试次数

## 所有奖励球引用（A+B）
var reward_balls: Array[RewardBall] = []
## A类球引用（不重生）
var type_a_balls: Array[RewardBall] = []
## B类球引用（可重生）
var type_b_balls: Array[RewardBall] = []

## PlayScene 引用
var _play_scene: PlayScene = null
## 奖励球场景
var _ball_scene: PackedScene = preload("res://assets/scenes/RewardBall.tscn")
## B类球重生队列：[{"ball": RewardBall, "timer": float}]
var _respawn_queue: Array[Dictionary] = []

func _ready() -> void:
	EventBus.reward_ball_collected.connect(_on_reward_ball_collected)

func _exit_tree() -> void:
	if EventBus.reward_ball_collected.is_connected(_on_reward_ball_collected):
		EventBus.reward_ball_collected.disconnect(_on_reward_ball_collected)

#初始化：由 PlayScene._ready() 调用
func setup(play_scene: PlayScene) -> void:
	_play_scene = play_scene
	_spawn_type_a_balls()
	_spawn_type_b_balls()


## 生成 A 类球：在每个玩家出生点所在角的子区域内随机生成3个
## 子矩形从竞技场角向内延伸，避免球生成到地图外
func _spawn_type_a_balls() -> void:
	if _play_scene == null:
		return
	
	var arena := _play_scene.arena_bounds
	# 子矩形尺寸：竞技场尺寸的一定比例
	var extent := arena.size * BALL_A_EXTENT_RATIO
	
	# 收集所有玩家出生点，用于距离检查
	var spawn_positions: Array[Vector2] = []
	for player in _play_scene.players:
		spawn_positions.append(player.spawn_position)
	
	for player in _play_scene.players:
		var spawn_pos := player.spawn_position
		var center := arena.get_center()
		var dir_x := 1 if spawn_pos.x >= center.x else -1
		var dir_y := 1 if spawn_pos.y >= center.y else -1
		# 从竞技场角向内延伸的子矩形
		var quadrant := MathUtils.quadrant_rect(arena, extent, dir_x, dir_y)
		
		for i in range(BALL_A_PER_PLAYER):
			var pos := _random_pos_avoiding_spawn(quadrant, BALL_A_SPAWN_MARGIN, spawn_positions, BALL_A_MIN_SPAWN_DIST)
			var ball := _create_ball(RewardBall.BallType.TYPE_A, BALL_A_REWARD, pos)
			type_a_balls.append(ball)
			reward_balls.append(ball)


## 生成 B 类球：在巡逻区域内随机位置（避开玩家和障碍物）
func _spawn_type_b_balls() -> void:
	if _play_scene == null:
		return
	
	var patrol := _play_scene.patrol_rect
	if patrol.size == Vector2.ZERO:
		return
	
	for i in range(BALL_B_MAX_COUNT):
		var pos := _safe_pos_in_patrol()
		var ball := _create_ball(RewardBall.BallType.TYPE_B, BALL_B_REWARD, pos)
		type_b_balls.append(ball)
		reward_balls.append(ball)


# 创建单个奖励球并添加到场景
func _create_ball(type: RewardBall.BallType, reward: float, pos: Vector2) -> RewardBall:
	var ball: RewardBall = _ball_scene.instantiate()
	ball.ball_type = type
	ball.reward_value = reward
	ball.position = pos
	_play_scene.add_child(ball)
	ball.set_owner(_play_scene)
	return ball


# 拾取信号处理：给对应玩家的 AIController 加奖励，B类球启动重生计时
func _on_reward_ball_collected(player_id: int, ball_type: int, reward_value: float, ball: RewardBall) -> void:
	if _play_scene == null:
		return
	
	# 找到对应玩家，增加奖励
	if player_id >= 0 and player_id < _play_scene.players.size():
		var player := _play_scene.players[player_id]
		player.ai_controller.reward += reward_value
	
	# B类球加入重生队列
	if ball_type == RewardBall.BallType.TYPE_B:
		_respawn_queue.append({"ball": ball, "timer": BALL_B_RESPAWN_DELAY})


# 每帧检查B类球重生队列
func _process(delta: float) -> void:
	var to_remove: Array[int] = []
	for i in range(_respawn_queue.size()):
		var entry: Dictionary = _respawn_queue[i]
		entry.timer -= delta
		if entry.timer <= 0.0:
			var ball: RewardBall = entry.ball
			if is_instance_valid(ball):
				# 在巡逻区域内安全位置重生（避开玩家和障碍物）
				var new_pos := _safe_pos_in_patrol()
				ball.global_position = new_pos
				ball.activate()
			to_remove.append(i)
	# 从后往前移除已重生的条目
	for i in to_remove:
		_respawn_queue.remove_at(i)


## 在巡逻区域内生成安全位置（避开玩家和障碍物）
## 检查：离所有玩家 >= BALL_B_MIN_PLAYER_DIST, 离所有障碍物 >= BALL_B_MIN_DECO_DIST
## 最多重试 _SAFE_POS_MAX_ATTEMPTS 次，退避返回最后一次随机位置
func _safe_pos_in_patrol() -> Vector2:
	var patrol := _play_scene.patrol_rect
	var deco_positions := _play_scene.collision_decoration_positions
	var last_pos := MathUtils.random_pos_in_rect(patrol, BALL_B_SPAWN_MARGIN)
	
	for attempt in range(_SAFE_POS_MAX_ATTEMPTS):
		var pos := MathUtils.random_pos_in_rect(patrol, BALL_B_SPAWN_MARGIN) if attempt > 0 else last_pos
		var safe := true
		
		# 检查与所有玩家的距离
		for player in _play_scene.players:
			if is_instance_valid(player) and pos.distance_to(player.global_position) < BALL_B_MIN_PLAYER_DIST:
				safe = false
				break
		
		if not safe:
			last_pos = MathUtils.random_pos_in_rect(patrol, BALL_B_SPAWN_MARGIN)
			continue
		
		# 检查与障碍物的距离
		for deco_pos in deco_positions:
			if pos.distance_to(deco_pos) < BALL_B_MIN_DECO_DIST:
				safe = false
				break
		
		if safe:
			return pos
		last_pos = pos
	
	# 退避：返回最后一次位置
	return last_pos


## 在矩形内生成随机位置，避开指定的出生点
## @param rect 目标矩形区域
## @param margin 距矩形边缘的最小距离
## @param spawn_positions 需要避开的出生点列表
## @param min_dist 与出生点的最小距离
## @return 符合条件的随机位置（最多重试 _SAFE_POS_MAX_ATTEMPTS 次）
func _random_pos_avoiding_spawn(rect: Rect2, margin: float, spawn_positions: Array[Vector2], min_dist: float) -> Vector2:
	for attempt in range(_SAFE_POS_MAX_ATTEMPTS):
		var pos := MathUtils.random_pos_in_rect(rect, margin)
		var too_close := false
		for sp in spawn_positions:
			if pos.distance_to(sp) < min_dist:
				too_close = true
				break
		if not too_close:
			return pos
	# 退避策略：返回最后一次随机位置
	return MathUtils.random_pos_in_rect(rect, margin)


## 游戏重置时调用
func reset_all() -> void:
	_respawn_queue.clear()
	for ball in reward_balls:
		if is_instance_valid(ball):
			ball.reset_ball()


## @deprecated 奖励球观测已迁移到 VisionSensor，此方法保留备用
## 获取指定玩家的奖励球观测数据
## 返回字典：
##   nearest_a: Vector2 — 到最近活跃A类球的相对位置（归一化）
##   nearest_b: Vector2 — 到最近活跃B类球的相对位置（归一化）
##   remaining_a: int   — 剩余A类球数量
##   active_b: int      — 活跃B类球数量
func get_obs_for_player(player_id: int) -> Dictionary:
	if _play_scene == null or player_id >= _play_scene.players.size():
		return {"nearest_a": Vector2.ZERO, "nearest_b": Vector2.ZERO, "remaining_a": 0, "active_b": 0}
	
	var player := _play_scene.players[player_id]
	var player_pos := player.global_position
	var half_arena := _play_scene.arena_length / 2.0
	
	# 最近A类球
	var nearest_a := Vector2.ZERO
	var min_dist_a := INF
	var remaining_a := 0
	for ball in type_a_balls:
		if not is_instance_valid(ball) or not ball.is_active:
			continue
		remaining_a += 1
		var dist := player_pos.distance_squared_to(ball.global_position)
		if dist < min_dist_a:
			min_dist_a = dist
			nearest_a = (ball.global_position - player_pos) / half_arena
	
	# 最近B类球
	var nearest_b := Vector2.ZERO
	var min_dist_b := INF
	var active_b := 0
	for ball in type_b_balls:
		if not is_instance_valid(ball) or not ball.is_active:
			continue
		active_b += 1
		var dist := player_pos.distance_squared_to(ball.global_position)
		if dist < min_dist_b:
			min_dist_b = dist
			nearest_b = (ball.global_position - player_pos) / half_arena
	
	if remaining_a == 0:
		nearest_a = Vector2.ZERO
	if active_b == 0:
		nearest_b = Vector2.ZERO
	
	return {
		"nearest_a": nearest_a,
		"nearest_b": nearest_b,
		"remaining_a": remaining_a,
		"active_b": active_b,
	}
