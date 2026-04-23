extends AIController2D

@onready var play_scene :PlayScene=$"../.."
var move_action:int
var n_time_step:int=0

func _physics_process(delta):
	super._physics_process(delta)
	if needs_reset:
		done=true
	
	#n_time_step=(n_time_step+1)%60
	#if n_time_step==0:
		#var obs=get_obs()
		#if _player.player_id == 0:  
			#print("=== Player %d Observations ===" % _player.player_id)
			#print("  self_state: ", obs.self_state)
			#print("  nearby_players: ", obs.nearby_players)
			#print("  nearby_balls: ", obs.nearby_balls)
			#print("  nearby_enemies: ", obs.nearby_enemies)
			#print("  map_state: ", obs.map_state,)
	

func reset():
	super.reset()
	_player as Player
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
	return {
		"self_state": {"size": [VisionSensor.SELF_STATE_DIM], "space": "box"},
		"nearby_players": {"size": [VisionSensor.MAX_NEARBY_PLAYERS * VisionSensor.PLAYER_SLOT_DIM], "space": "box"},
		"nearby_balls": {"size": [VisionSensor.MAX_NEARBY_BALLS * VisionSensor.BALL_SLOT_DIM], "space": "box"},
		"nearby_enemies": {"size": [VisionSensor.MAX_NEARBY_ENEMIES * VisionSensor.ENEMY_SLOT_DIM], "space": "box"},
		"map_state": {"size": [52], "space": "box"}
	}

func set_action(action) -> void:
	move_action = action["move_action"]
	# 0=上, 1=下, 2=左, 3=右, 4=攻击，5=待机
	_player.pending_action = move_action as Player.Action
