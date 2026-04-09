extends TextureProgressBar

@export var player:Player

func _process(delta: float) -> void:
	self.value=player.current_health/player.max_health*self.max_value
