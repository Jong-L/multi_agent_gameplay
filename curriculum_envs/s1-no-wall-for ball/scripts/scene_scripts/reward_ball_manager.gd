extends Node
class_name RewardBallManager

## 奖励球管理器
## 负责 A/B 类球的生成、状态追踪、奖励发放、观测数据
## 由 PlayScene 在 _ready() 中创建并添加为子节点
##
## A类球：4个玩家各3个=12个，出生点所在角的子区域内生成，不重生
## B类球：巡逻区域内最多5个，5秒后重生（位置由 Manager 决定）

const BALL_A_PER_PLAYER: int = 3
const BALL_B_MAX_COUNT: int = 5
const BALL_A_REWARD: float = 1.0
const BALL_B_REWARD: float = 1.5
const BALL_B_RESPAWN_DELAY: float = 5.0
const BALL_A_SPAWN_MARGIN: float = 10.0   ## A类球距子区域边缘的最小距离
const BALL_A_MIN_DECO_DIST: float = 20.0   ## A类球与障碍物的最小距离
const BALL_A_EXTENT_RATIO: float = 0.2    ## A类球子区域占竞技场尺寸的比例
const BALL_A_MIN_SPAWN_DIST: float = 30.0 ## A类球与玩家出生点的最小距离
const BALL_B_SPAWN_MARGIN: float = 10.0   ## B类球距巡逻区边缘的最小距离
const BALL_B_MIN_PLAYER_DIST: float = 60.0 ## B类球与任何玩家的最小距离
const BALL_B_MIN_DECO_DIST: float = 30.0   ## B类球与障碍物的最小距离
const _SAFE_POS_MAX_ATTEMPTS: int = 50     ## 安全位置生成的最大重试次数

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
	var collision_decoration_positions := _play_scene.collision_decoration_positions
	
	for player in _play_scene.players:
		var spawn_pos := player.spawn_position
		var center := arena.get_center()
		var dir_x := 1 if spawn_pos.x >= center.x else -1
		var dir_y := 1 if spawn_pos.y >= center.y else -1
		# 从竞技场角向内延伸的子矩形
		var quadrant := MathUtils.quadrant_rect(arena, extent, dir_x, dir_y)
		var candidate_positions := _grid_pos_candidates_avoiding_spawn_and_deco(quadrant, BALL_A_SPAWN_MARGIN, spawn_positions, BALL_A_MIN_SPAWN_DIST, collision_decoration_positions, BALL_A_MIN_DECO_DIST)
		
		for i in range(BALL_A_PER_PLAYER):
			var pos: Vector2
			if candidate_positions.is_empty():
				pos = _random_pos_avoiding_spawn_and_deco(quadrant, BALL_A_SPAWN_MARGIN, spawn_positions, BALL_A_MIN_SPAWN_DIST, collision_decoration_positions, BALL_A_MIN_DECO_DIST)
			else:
				pos = _pop_random_pos(candidate_positions)
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


# 拾取信号处理：B类球启动重生计时,player_id由奖励管理器获取，此处不需要
func _on_reward_ball_collected(_player_id: int, ball_type: int, ball: RewardBall) -> void:
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
	for i in range(to_remove.size() - 1, -1, -1):
		_respawn_queue.remove_at(to_remove[i])

## 在巡逻区域内生成安全位置（避开玩家和障碍物）
## 检查：离所有玩家 >= BALL_B_MIN_PLAYER_DIST, 离所有障碍物 >= BALL_B_MIN_DECO_DIST
## 最多重试 _SAFE_POS_MAX_ATTEMPTS 次，退避返回最后一次随机位置
func _safe_pos_in_patrol() -> Vector2:
	var patrol := _play_scene.patrol_rect
	var deco_positions := _play_scene.collision_decoration_positions
	var last_pos := _rng_pos_in_rect(patrol, BALL_B_SPAWN_MARGIN)
	
	for attempt in range(_SAFE_POS_MAX_ATTEMPTS):
		var pos := _rng_pos_in_rect(patrol, BALL_B_SPAWN_MARGIN) if attempt > 0 else last_pos
		var safe := true
		
		# 检查与所有玩家的距离
		for player in _play_scene.players:
			if is_instance_valid(player) and pos.distance_to(player.global_position) < BALL_B_MIN_PLAYER_DIST:
				safe = false
				break
		
		if not safe:
			last_pos = _rng_pos_in_rect(patrol, BALL_B_SPAWN_MARGIN)
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
		var pos := _rng_pos_in_rect(rect, margin)
		var too_close := false
		for sp in spawn_positions:
			if pos.distance_to(sp) < min_dist:
				too_close = true
				break
		if not too_close:
			return pos
	# 保底返回
	return _rng_pos_in_rect(rect, margin)

func _grid_pos_candidates_avoiding_spawn_and_deco(rect: Rect2, margin: float, spawn_positions: Array[Vector2], min_dist: float, deco_positions: Array[Vector2], min_deco_dist: float) -> Array[Vector2]:
	var candidates: Array[Vector2] = []
	if _play_scene == null or _play_scene.arena_tile_positions.is_empty():
		return candidates

	var usable_rect := Rect2(
		rect.position + Vector2(margin, margin),
		rect.size - Vector2(margin * 2.0, margin * 2.0)
	)
	if usable_rect.size.x <= 0.0 or usable_rect.size.y <= 0.0:
		return candidates

	for tile_pos in _play_scene.arena_tile_positions:
		if not usable_rect.has_point(tile_pos):
			continue
		if _is_too_close_to_any(tile_pos, spawn_positions, min_dist):
			continue
		if _is_too_close_to_any(tile_pos, deco_positions, min_deco_dist):
			continue
		candidates.append(tile_pos)
	return candidates

func _pop_random_pos(positions: Array[Vector2]) -> Vector2:
	var index := randi_range(0, positions.size() - 1)
	var pos := positions[index]
	positions.remove_at(index)
	return pos

func _is_too_close_to_any(pos: Vector2, positions: Array[Vector2], min_dist: float) -> bool:
	var min_dist_sq := min_dist * min_dist
	for other_pos in positions:
		if pos.distance_squared_to(other_pos) < min_dist_sq:
			return true
	return false

func _random_pos_avoiding_spawn_and_deco(rect: Rect2, margin: float, spawn_positions: Array[Vector2], min_dist: float, deco_positions: Array[Vector2], min_deco_dist: float) -> Vector2:
	
	var pos: Vector2
	for attempt in range(_SAFE_POS_MAX_ATTEMPTS):
		pos = _rng_pos_in_rect(rect, margin)
		var too_close := false
		for sp in spawn_positions:
			if pos.distance_to(sp) < min_dist:
				too_close = true
				break
		if not too_close:
			for deco_pos in deco_positions:
				if pos.distance_to(deco_pos) < min_deco_dist:
					too_close = true
					break
			if not too_close:
				return pos
	# 保底返回
	print("Failed to find a safe position, returning the last position")
	return pos

## 在矩形内生成随机位置
func _rng_pos_in_rect(rect: Rect2, margin: float) -> Vector2:
	var x := randf_range(rect.position.x + margin, rect.end.x - margin)
	var y := randf_range(rect.position.y + margin, rect.end.y - margin)
	return Vector2(x, y)


## 游戏重置时调用：重新随机化所有球的位置
func reset_all() -> void:
	_respawn_queue.clear()
	
	# 重新随机化 A 类球位置
	var arena := _play_scene.arena_bounds
	var extent := arena.size * BALL_A_EXTENT_RATIO
	var spawn_positions: Array[Vector2] = []
	for player in _play_scene.players:
		spawn_positions.append(player.spawn_position)
	var collision_decoration_positions := _play_scene.collision_decoration_positions
	
	var a_idx := 0
	for player in _play_scene.players:
		var spawn_pos := player.spawn_position
		var center := arena.get_center()
		var dir_x := 1 if spawn_pos.x >= center.x else -1
		var dir_y := 1 if spawn_pos.y >= center.y else -1
		var quadrant := MathUtils.quadrant_rect(arena, extent, dir_x, dir_y)
		var candidate_positions := _grid_pos_candidates_avoiding_spawn_and_deco(quadrant, BALL_A_SPAWN_MARGIN, spawn_positions, BALL_A_MIN_SPAWN_DIST, collision_decoration_positions, BALL_A_MIN_DECO_DIST)
		
		for i in range(BALL_A_PER_PLAYER):
			if a_idx < type_a_balls.size() and is_instance_valid(type_a_balls[a_idx]):
				var ball := type_a_balls[a_idx]
				ball.reset_ball()
				if candidate_positions.is_empty():
					ball.position = _random_pos_avoiding_spawn_and_deco(quadrant, BALL_A_SPAWN_MARGIN, spawn_positions, BALL_A_MIN_SPAWN_DIST, collision_decoration_positions, BALL_A_MIN_DECO_DIST)
				else:
					ball.position = _pop_random_pos(candidate_positions)
			a_idx += 1
	
	# 重新随机化 B 类球位置
	for ball in type_b_balls:
		if is_instance_valid(ball):
			ball.reset_ball()
			ball.position = _safe_pos_in_patrol()
