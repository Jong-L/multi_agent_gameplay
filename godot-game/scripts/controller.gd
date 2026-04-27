extends AIController2D

@onready var play_scene :PlayScene=$"../.."
var move_action:int
var n_time_step:int=0

func _physics_process(delta):
	super._physics_process(delta)
	if needs_reset:
		done=true
	
	#n_time_step=(n_time_step+1)%8
	#if n_time_step==0:
		#var obs=get_obs()
		#if _player.player_id == 1:  
			#print("=== Player %d Observations ===" % _player.player_id)
			#print("  self_state: ", obs.self_state)
			#print("  nearby_players: ", obs.nearby_players)
			#print("  nearby_balls: ", obs.nearby_balls)
			#print("  nearby_enemies: ", obs.nearby_enemies)
			#print("  map_state: ", obs.map_state)
	

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

## 覆写 get_obs_space() 以支持多 key 字典观测空间
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
	move_action = action["move_action"]
	# 0=上, 1=下, 2=左, 3=右, 4=攻击，5=待机
	_player.pending_action = move_action as Player.Action
