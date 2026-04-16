class_name VisionSensor
extends Node

## 视野传感器（独立工具类）
## 负责计算以指定玩家为圆心的视野范围内所有实体信息
## 将视野内实体信息编码为固定维度的字典观测数据，配合 SB3 MultiInputPolicy 使用
##
## 使用方式：
## - 作为 PlayScene 的子节点添加一个实例
## - PlayScene 调用 VisionSensor.scan() 静态方法，传入当前帧数据
## - 无需 Player 持有 PlayScene 引用

@export var vision_radius: float = 250.0  ## 视野半径（像素）

## ---- 槽位常量 ----
## 与场景中实际实体数量对应，修改时需同步更新 controller.gd 的 get_obs_space()
const MAX_NEARBY_PLAYERS: int = 3   ## 最多观测其他3个玩家
const MAX_NEARBY_BALLS: int = 17    ## A球12 + B球5
const MAX_NEARBY_ENEMIES: int = 5   ## 场景中5个敌人

## ---- 每个槽位的维度 ----
const PLAYER_SLOT_DIM: int = 6  ## [rel_x, rel_y, is_alive, hp_ratio, flip_h, dist_norm]
const BALL_SLOT_DIM: int = 5    ## [rel_x, rel_y, ball_type, is_active, dist_norm]
const ENEMY_SLOT_DIM: int = 6   ## [rel_x, rel_y, is_alive, is_respawning, hp_ratio, dist_norm]

## ---- 自身状态维度 ----
const SELF_STATE_DIM: int = 6   ## [player_id, pos_x, pos_y, flip_h, hp_ratio, max_health]


## 扫描指定玩家视野内所有实体（静态方法，可独立调用）
## PlayScene 调用此方法为每个玩家生成观测数据
## @param player 观测的主体玩家
## @param all_players 所有玩家列表（由 PlayScene 提供）
## @param all_enemies 所有敌人列表（由 PlayScene 提供）
## @param all_balls 所有奖励球列表（由 PlayScene 提供）
## @param arena_length 竞技场边长（用于归一化，由 PlayScene 提供）
## @param radius 视野半径（像素），默认 250.0
## @return Dictionary 格式的观测数据，每个 key 对应一个固定长度的数组
static func scan(
	player: Player,
	all_players: Array[Player],
	all_enemies: Array[Enemy],
	all_balls: Array[RewardBall],
	arena_length: float,
	radius: float = 250.0
) -> Dictionary:
	var half_arena := arena_length / 2.0
	var player_pos := player.global_position

	# ---- 自身状态 ----
	var self_state: Array = [
		player.player_id,
		player_pos.x / half_arena,
		player_pos.y / half_arena,
		float(player.animated_sprite.flip_h),
		player.current_health / player.max_health,
		player.max_health,
	]

	# ---- 视野内其他玩家 ----
	var nearby_players_data: Array = []
	for other in all_players:
		if other.player_id == player.player_id:
			continue  # 跳过自己
		if not is_instance_valid(other):
			continue
		if other.is_dead:
			continue
		var dist := player_pos.distance_to(other.global_position)
		if dist > radius:
			continue
		var rel := (other.global_position - player_pos) / half_arena
		nearby_players_data.append({
			"dist": dist,
			"slot": [
				rel.x,
				rel.y,
				1.0,  # is_alive (已经过滤了 is_dead)
				other.current_health / other.max_health,
				float(other.animated_sprite.flip_h),
				dist / half_arena,
			]
		})

	# ---- 视野内奖励球 ----
	var nearby_balls_data: Array = []
	for ball in all_balls:
		if not is_instance_valid(ball):
			continue
		if not ball.is_active:
			continue
		var dist := player_pos.distance_to(ball.global_position)
		if dist > radius:
			continue
		var rel := (ball.global_position - player_pos) / half_arena
		nearby_balls_data.append({
			"dist": dist,
			"slot": [
				rel.x,
				rel.y,
				1.0 if ball.ball_type == RewardBall.BallType.TYPE_B else 0.0,  # 0=A, 1=B
				1.0,  # is_active (已经过滤了 !is_active)
				dist / half_arena,
			]
		})

	# ---- 视野内敌人 ----
	var nearby_enemies_data: Array = []
	for enemy in all_enemies:
		if not is_instance_valid(enemy):
			continue
		if enemy.is_dead:
			continue
		if enemy.is_respawning:
			continue
		var dist := player_pos.distance_to(enemy.global_position)
		if dist > radius:
			continue
		var rel := (enemy.global_position - player_pos) / half_arena
		nearby_enemies_data.append({
			"dist": dist,
			"slot": [
				rel.x,
				rel.y,
				1.0,  # is_alive (已经过滤了 is_dead)
				0.0,  # is_respawning (已经过滤了 is_respawning)
				enemy.current_health / enemy.max_health,
				dist / half_arena,
			]
		})

	# ---- 按距离排序 + 填充固定维度 slot ----
	var nearby_players := _fill_slots(nearby_players_data, MAX_NEARBY_PLAYERS, PLAYER_SLOT_DIM)
	var nearby_balls := _fill_slots(nearby_balls_data, MAX_NEARBY_BALLS, BALL_SLOT_DIM)
	var nearby_enemies := _fill_slots(nearby_enemies_data, MAX_NEARBY_ENEMIES, ENEMY_SLOT_DIM)

	return {
		"self_state": self_state,
		"nearby_players": nearby_players,
		"nearby_balls": nearby_balls,
		"nearby_enemies": nearby_enemies,
	}


## 将视野内实体数据按距离排序后填入固定维度 slot
## @param data_list 包含 {"dist": float, "slot": Array} 的列表
## @param max_slots 最大槽位数
## @param slot_dim 每个槽位的维度
## @return 固定长度的扁平数组 (max_slots × slot_dim)
static func _fill_slots(data_list: Array, max_slots: int, slot_dim: int) -> Array:
	# 按距离从近到远排序
	data_list.sort_custom(func(a, b): return a["dist"] < b["dist"])

	var result: Array = []
	result.resize(max_slots * slot_dim)
	result.fill(0.0)

	var count := mini(data_list.size(), max_slots)
	for i in count:
		var slot: Array = data_list[i]["slot"]
		for j in slot.size():
			result[i * slot_dim + j] = slot[j]

	return result
