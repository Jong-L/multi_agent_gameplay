extends AIController2D

@onready var play_scene :PlayScene=$"../.."

@export var cfg:GameConfig=null
var move_action:int
var n_time_step:int=0
var _prev_action: int = Player.Action.IDLE  # 上一帧动作，用于观测

func _ready():
	super._ready()
	
func _physics_process(_delta):
	n_steps += 1
	if n_steps > reset_after:
		needs_reset = true
		done=true
	if needs_reset:
		needs_reset=false
		reset()

	#if _player.player_id==0:
		#get_obs()


func reset():
	super.reset()
	_prev_action = Player.Action.IDLE
	if _player is Player:#加这个判断用于代码提示，写在同一行条件语句没有代码提示
		if _player.player_id==0:#只用重置一次
			play_scene._handle_reset()

func get_obs() -> Dictionary:
	# PlayScene分发
	if play_scene == null:
		return {}
	var obs_dict = play_scene.get_obs_for_player(_player)
	
	# 追加 prev_action one-hot (6 维) 到 self_state
	var prev_action_onehot: Array = []
	prev_action_onehot.resize(6)
	prev_action_onehot.fill(0.0)
	prev_action_onehot[_prev_action] = 1.0
	obs_dict["self_state"].append_array(prev_action_onehot)
	
	# 追加 episode_progress (1 维) 到 self_state
	var ep_progress := clampf(float(n_steps) / float(reset_after), 0.0, 1.0)
	obs_dict["self_state"].append(ep_progress)
	
	# 展平所有键为单一 "obs" 向量, Python 端可按偏移切片分别处理
	var flat: Array = []
	flat.append_array(obs_dict["self_state"])
	flat.append_array(obs_dict["nearby_players"])
	flat.append_array(obs_dict["nearby_balls"])
	flat.append_array(obs_dict["nearby_enemies"])
	flat.append_array(obs_dict["map_state"])
	#if _player.player_id==0:
		#var output_arr=obs_dict["self_state"].slice(6,8)
		#print(output_arr)
	return {"obs": flat}

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

### 展平后的单键观测空间, Python 端按偏移量切片可恢复各段语义
func get_obs_space() -> Dictionary:
	var ray_count := 36
	var use_valid_mask := true
	if play_scene != null and play_scene.game_config != null:
		ray_count = play_scene.game_config.ray_count
		use_valid_mask = play_scene.game_config.use_observation_valid_mask

	var valid_dims := 1 if use_valid_mask else 0
	var player_slot_dim := VisionSensor.PLAYER_SLOT_DIM + VisionSensor.PLAYER_EXTRA_DIM + valid_dims
	var ball_slot_dim := VisionSensor.BALL_SLOT_DIM + valid_dims
	var enemy_slot_dim := VisionSensor.ENEMY_SLOT_DIM + valid_dims
	var total_self_dim := VisionSensor.SELF_STATE_DIM + 6 + 1  # +6(prev_action) +1(episode_progress)
	var total_dim := (
		total_self_dim
		+ VisionSensor.MAX_NEARBY_PLAYERS * player_slot_dim
		+ VisionSensor.MAX_NEARBY_BALLS * ball_slot_dim
		+ VisionSensor.MAX_NEARBY_ENEMIES * enemy_slot_dim
		+ ray_count
	)
	return {"obs": {"size": [total_dim], "space": "box"}}

func set_action(action) -> void:
	# 如果启用了单智能体训练且当前玩家不是训练玩家，强制 IDLE
	if play_scene != null and play_scene.game_config != null:
		if cfg.training_player_id != GameConfig.TrainingPlayer.ALL:
			if _player is Player and _player.player_id != cfg.training_player_id:
				move_action = Player.Action.IDLE
				_player.pending_action = move_action as Player.Action
				_prev_action = move_action
				return
	move_action = action["move_action"]
	#if _player is Player:
		#if _player.player_id==0:
			#print(move_action)
	_player.pending_action = move_action as Player.Action
	_prev_action = move_action
