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
const PLAYER_SLOT_DIM: int = 5     #相比于自身增加距离
const BALL_SLOT_DIM: int = 4    
const ENEMY_SLOT_DIM: int = 5   #

## ---- 自身状态维度 ----
const SELF_STATE_DIM: int = 4   


## 扫描指定玩家视野内所有实体
func scan(
	player: Player,
	all_players: Array[Player],
	all_enemies: Array[Enemy],
	all_balls: Array[RewardBall],
	arena_length: float,
	use_valid_mask: bool = false,
) -> Dictionary:
	var half_arena := arena_length / 2.0
	var player_pos := player.global_position
	# 自身状态 
	var self_state: Array = [
		#player.player_id, #调试时用，但智能体不需要管"我是谁"
		player_pos.x / half_arena,#归一化到[-1,1]
		player_pos.y / half_arena,
		player.current_health / player.max_health,#归一化到[0,1]
		int(player.animated_sprite.flip_h),#是否翻转影响攻击范围
	]
	# 视野内其他玩家
	var nearby_players_data: Array = []
	for other in all_players:
		if other.player_id == player.player_id:
			continue  # 跳过自己
		if not is_instance_valid(other):
			continue
		if other.is_dead:
			continue
		
		var dist :float= player_pos.distance_to(other.global_position)#距离
		if dist > vision_radius:
			continue
		
		var rel := (other.global_position - player_pos) / vision_radius#向量距离
		nearby_players_data.append({
			"dist": dist,#填充并排序slot时用到，不进入最终观测
			"slot": [#实际观测特征
				rel.x,#[-1,1]
				rel.y,
				other.current_health / other.max_health,#归一化到[0,1]
				int(other.animated_sprite.flip_h),
				dist / vision_radius,
			]
		})

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

	# 视野内敌人
	var nearby_enemies_data: Array = []
	for enemy in all_enemies:
		if not is_instance_valid(enemy):
			continue
		if enemy.is_dead:
			continue
		if enemy.is_respawning:
			continue
		var dist := player_pos.distance_to(enemy.global_position)
		if dist > vision_radius:
			continue
		var rel := (enemy.global_position - player_pos) / vision_radius
		nearby_enemies_data.append({
			"dist": dist,
			"slot": [
				rel.x,
				rel.y,
				enemy.current_health / enemy.max_health,#归一化到[0,1]
				float(enemy.animated_sprite.flip_h),
				dist / vision_radius,
			]
		})

	# 按距离排序并填充固定维度 slot
	var nearby_players := _fill_slots(nearby_players_data, MAX_NEARBY_PLAYERS, PLAYER_SLOT_DIM, use_valid_mask)
	var nearby_balls := _fill_slots(nearby_balls_data, MAX_NEARBY_BALLS, BALL_SLOT_DIM, use_valid_mask)
	var nearby_enemies := _fill_slots(nearby_enemies_data, MAX_NEARBY_ENEMIES, ENEMY_SLOT_DIM, use_valid_mask)

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
