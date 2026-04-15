extends AIController2D

var move_action:int

func _physics_process(delta):
	super._physics_process(delta)
	if needs_reset:
		done=true

func reset():
	super.reset()
	_player.play_scene._handle_reset()

func get_obs() -> Dictionary:
	var obs=_player.get_obs()
	return obs
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
	_player.pending_action = move_action as Player.Action
