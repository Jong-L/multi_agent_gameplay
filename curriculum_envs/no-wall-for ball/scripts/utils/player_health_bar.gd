extends TextureProgressBar

@export var player: Player                    ## 绑定的玩家实例

func _process(_delta: float) -> void:
	if player == null:
		return
	
	var health_percent = player.current_health / player.max_health
	self.value = health_percent * self.max_value
	
	# 根据相机状态控制可见性
	visible = CameraManager.should_show_player_ui(player.player_id)
