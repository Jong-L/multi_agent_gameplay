extends TextureProgressBar

@export var player: Player

func _process(delta: float) -> void:
	if player == null:
		return
	self.value = player.current_health / player.max_health * self.max_value
	# 根据相机状态决定可见性
	visible = CameraManager.should_show_player_ui(player.player_id)
