extends AIController2D

var move_action:int

func _physics_process(delta):
	super._physics_process(delta)
	if needs_reset:
		done=true

func reset():
	super.reset()
	var play_scene := _player.get_parent() as PlayScene
	play_scene._handle_reset()

func get_obs() -> Dictionary:
	# PlayScene分发
	var play_scene := _player.get_parent() as PlayScene
	if play_scene == null:
		return {}
	var obs=play_scene.get_obs_for_player(_player)
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
