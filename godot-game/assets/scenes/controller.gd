extends AIController2D

var move_action:int

func get_obs() -> Dictionary:
	var obs=[_player.global_position.x,_player.global_position.y]
	print(obs)
	return {"obs":obs}
func get_reward() -> float:	
	return reward
	
func get_action_space() -> Dictionary:
	return {
		"move_action": {          
			"size": 6,            
			"action_type": "discrete",  
		},
	}
	
func set_action(action) -> void:
	move_action = action["move_action"]
	# 0=上, 1=下, 2=左, 3=右, 4=攻击，5=待机
	print()
	_player.pending_action = move_action as Player.Action
