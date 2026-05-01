extends AIController2D

@onready var play_scene :PlayScene=$"../.."
var move_action:int
var n_time_step:int=0

func _ready():
	super._ready()
	# 根据 training_player_id 动态设置 policy_name
	# 必须在 Sync._get_agents() 读取 policy_name 之前完成
	if play_scene != null and play_scene.game_config != null:
		var cfg := play_scene.game_config
		if cfg.training_player_id >= 0:
			if _player is Player and _player.player_id == cfg.training_player_id:
				policy_name = "learning_policy"
			else:
				policy_name = "idle_policy"

func _physics_process(_delta):
	n_steps += 1
	if n_steps > reset_after:
		needs_reset = true
		done=true
	if needs_reset:
		needs_reset=false
		reset()
		


func reset():
	super.reset()
	if _player is Player:#加这个判断用于代码提示，写在同一行条件语句没有代码提示
		if _player.player_id==0:#只用重置一次
			play_scene._handle_reset()

func get_obs() -> Dictionary:
	# PlayScene分发
	if play_scene == null:
		return {}
	var obs=play_scene.get_obs_for_player(_player)
	#if _player.player_id==0:
		#print(obs)
	return obs

func get_reward() -> float:
	#if _player.player_id==0:
		#print("{0} get reward {1}".format([_player.player_id,reward]))
	return reward

func get_action_space() -> Dictionary:
	return {
		"move_action": {
			"size": 6,
			"action_type": "discrete",
		},
	}

## overide ,多 key 字典观测空间
func get_obs_space() -> Dictionary:
	var ray_count := 32
	var use_valid_mask := false
	if play_scene != null and play_scene.game_config != null:
		ray_count = play_scene.game_config.ray_count
		use_valid_mask = play_scene.game_config.use_observation_valid_mask

	var player_slot_dim := VisionSensor.PLAYER_SLOT_DIM + (1 if use_valid_mask else 0)
	var ball_slot_dim := VisionSensor.BALL_SLOT_DIM + (1 if use_valid_mask else 0)
	var enemy_slot_dim := VisionSensor.ENEMY_SLOT_DIM + (1 if use_valid_mask else 0)
	var obs_space := {
		"self_state": {"size": [VisionSensor.SELF_STATE_DIM], "space": "box"},
		"nearby_players": {"size": [VisionSensor.MAX_NEARBY_PLAYERS * player_slot_dim], "space": "box"},
		"nearby_balls": {"size": [VisionSensor.MAX_NEARBY_BALLS * ball_slot_dim], "space": "box"},
		"nearby_enemies": {"size": [VisionSensor.MAX_NEARBY_ENEMIES * enemy_slot_dim], "space": "box"},
		"map_state": {"size": [ray_count], "space": "box"}
	}
	return obs_space

func set_action(action) -> void:
	# 如果启用了单智能体训练且当前玩家不是训练玩家，强制 IDLE
	if play_scene != null and play_scene.game_config != null:
		var cfg := play_scene.game_config
		if cfg.training_player_id >= 0:
			if _player is Player and _player.player_id != cfg.training_player_id:
				move_action = Player.Action.IDLE
				_player.pending_action = move_action as Player.Action
				return
	move_action = action["move_action"]
	#if _player is Player:
		#if _player.player_id==0:
			#print(move_action)
	_player.pending_action = move_action as Player.Action
