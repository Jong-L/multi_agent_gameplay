class_name VisionSensor
extends Node

## 视野传感器
## 负责计算以指定玩家为圆心的视野范围内所有实体信息
## 将视野内实体信息编码为固定维度的字典观测数据

@export var vision_radius: float  ## 视野半径（像素）

## ---- 槽位常量 ----
## 与场景中实际实体数量对应，修改时需同步更新 controller.gd 的 get_obs_space()
const MAX_NEARBY_PLAYERS: int = 3   ## 最多观测其他3个玩家
const MAX_NEARBY_BALLS: int = 8    ## A球3 + B球5,实际上有12个A球，但是A球只出现在四角。
const MAX_NEARBY_ENEMIES: int = 5   ## 场景中5个敌人

## ---- 每个槽位的维度 ----
const PLAYER_SLOT_DIM: int = 9     #rel_x, rel_y, hp_ratio, flip_h, dist_ratio, is_attack_animating, skill_cooldown_ratio, vel_x, vel_y
const BALL_SLOT_DIM: int = 4    
const ENEMY_SLOT_DIM: int = 9   #rel_x, rel_y, hp_ratio, flip_h, dist_ratio, is_attack_animating, skill_cooldown_ratio, vel_x, vel_y

## 玩家额外观测维度
const PLAYER_EXTRA_DIM: int = 1  #player_id

## ---- 自身状态维度 ----
## pos_x, pos_y, hp_ratio, flip_h, is_attack_animating, skill_cooldown_ratio, vel_x, vel_y
const SELF_STATE_DIM: int = 8


## 扫描指定玩家视野内所有实体
func scan(
	player: Player,
	all_players: Array[Player],
	all_enemies: Array[Enemy],
	all_balls: Array[RewardBall],
	arena_length: float,
	arena_center: Vector2 = Vector2.ZERO,
	use_valid_mask: bool = true,
) -> Dictionary:
	var half_arena := arena_length / 2.0
	var player_pos := player.global_position

	# 自身状态 — 相对于竞技场中心归一化到[-1,1]
	var self_vel := player.get_normalized_velocity()
	var self_state: Array = [
		#player.player_id, #调试时用，但智能体不需要管"我是谁"
		(player_pos.x - arena_center.x) / half_arena,
		(player_pos.y - arena_center.y) / half_arena,
		player.current_health / player.max_health,#归一化到[0,1]
		int(player.animated_sprite.flip_h),#是否翻转影响攻击范围
		int(player.is_attack_animating()),#自身攻击动画状态 [0,1]
		player.get_skill_cooldown_ratio(),#自身技能冷却比例 [0,1]
		clampf(self_vel.x,-1.0,1.0),#自身归一化速度 [-1,1]
		clampf(self_vel.y,-1.0,1.0),
	]
	# 视野内其他玩家 — 按 player_id 固定槽位
	var nearby_players := _fill_player_slots_fixed(player, all_players, use_valid_mask)

	# 视野内敌人 — 按数组索引固定槽位
	var nearby_enemies := _fill_enemy_slots_fixed(player, all_enemies, use_valid_mask)

	# 视野内奖励球
	var nearby_balls_data: Array = []
	for ball in all_balls:
		if not is_instance_valid(ball):
			continue
		if not ball.is_active:
			continue
		var dist := player_pos.distance_to(ball.global_position)
		if dist > vision_radius:
			continue
		var rel := (ball.global_position - player_pos) / vision_radius
		nearby_balls_data.append({
			"dist": dist,
			"slot": [
				rel.x,
				rel.y,
				1.0 if ball.ball_type == RewardBall.BallType.TYPE_B else 0.0,  # 0=A, 1=B
				dist / vision_radius,
			]
		})

	# 奖励球按距离排序填充
	var nearby_balls := _fill_slots(nearby_balls_data, MAX_NEARBY_BALLS, BALL_SLOT_DIM, use_valid_mask)

	var obs := {
		"self_state": self_state,
		"nearby_players": nearby_players,
		"nearby_balls": nearby_balls,
		"nearby_enemies": nearby_enemies,
	}
	return obs


#将视野内实体数据按距离排序后填入固定维度 slot
static func _fill_slots(data_list: Array, max_slots: int, slot_dim: int, use_valid_mask: bool = false) -> Array:
	# 按距离从近到远排序
	data_list.sort_custom(func(a, b): return a["dist"] < b["dist"])

	var output_slot_dim := slot_dim + (1 if use_valid_mask else 0)
	var result: Array = []
	result.resize(max_slots * output_slot_dim)
	result.fill(0.0)

	var count := mini(data_list.size(), max_slots)#奖励球，但正常情况视野内的奖励球不会超过8个
	for i in count:
		var slot: Array = data_list[i]["slot"]
		for j in slot.size():
			result[i * output_slot_dim + j] = slot[j]
		if use_valid_mask:
			result[i * output_slot_dim + slot_dim] = 1.0#有效

	return result


# 玩家槽位按 player_id 固定填充
func _fill_player_slots_fixed(player: Player, all_players: Array,  use_valid_mask: bool) -> Array:
	var player_pos := player.global_position
	var n_players := maxf(1.0, float(all_players.size()))
	var slot_dim := PLAYER_SLOT_DIM + PLAYER_EXTRA_DIM
	var output_slot_dim := slot_dim + (1 if use_valid_mask else 0)
	var result: Array = []
	result.resize(MAX_NEARBY_PLAYERS * output_slot_dim)
	result.fill(0.0)

	for other in all_players:
		if other.player_id == player.player_id:
			continue
		if not is_instance_valid(other) or other.is_dead:
			continue

		var dist := player_pos.distance_to(other.global_position)
		if dist > vision_radius:
			continue

		var slot_idx :int= other.player_id if other.player_id < player.player_id else other.player_id - 1
		var rel :Vector2= (other.global_position - player_pos) / vision_radius
		var other_vel :Vector2= other.get_normalized_velocity()
		var slot_data: Array = [
			other.player_id / (n_players - 1.0),  # 归一化玩家ID [0,1]
			rel.x,#相对位置
			rel.y,
			other.current_health / other.max_health,
			int(other.animated_sprite.flip_h),#翻转决定攻击范围
			dist / vision_radius,
			int(other.is_attack_animating()),
			other.get_skill_cooldown_ratio(),
			clampf(other_vel.x, -1.0, 1.0),
			clampf(other_vel.y, -1.0, 1.0),
		]

		var base_idx := slot_idx * output_slot_dim #起始索引
		for j in slot_data.size():
			result[base_idx + j] = slot_data[j]
		if use_valid_mask:
			result[base_idx + slot_dim] = 1.0

	return result


# 敌人槽位按数组索引固定填充
func _fill_enemy_slots_fixed(player: Player, all_enemies: Array, use_valid_mask: bool) -> Array:
	var player_pos := player.global_position
	var output_slot_dim := ENEMY_SLOT_DIM + (1 if use_valid_mask else 0)
	var result: Array = []
	result.resize(MAX_NEARBY_ENEMIES * output_slot_dim)
	result.fill(0.0)

	var num_enemies := mini(all_enemies.size(), MAX_NEARBY_ENEMIES) #场景中敌人数量
	for i in num_enemies:
		var enemy = all_enemies[i]
		if not is_instance_valid(enemy) or enemy.is_dead or enemy.is_respawning:
			continue

		var dist := player_pos.distance_to(enemy.global_position)
		if dist > vision_radius:
			continue

		var rel :Vector2= (enemy.global_position - player_pos) / vision_radius #相对距离
		var enemy_vel :Vector2= enemy.get_normalized_velocity() #速度
		var slot_data: Array = [
			rel.x,
			rel.y,
			enemy.current_health / enemy.max_health,
			float(enemy.animated_sprite.flip_h),
			dist / vision_radius,
			int(enemy.is_attack_animating()),
			enemy.get_skill_cooldown_ratio(),
			clampf(enemy_vel.x, -1.0, 1.0),
			clampf(enemy_vel.y, -1.0, 1.0),
		]

		var base_idx := i * output_slot_dim
		for j in slot_data.size():
			result[base_idx + j] = slot_data[j]
		if use_valid_mask:
			result[base_idx + ENEMY_SLOT_DIM] = 1.0

	return result
