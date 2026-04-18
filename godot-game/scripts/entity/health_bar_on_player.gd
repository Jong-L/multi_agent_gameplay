extends TextureProgressBar

@onready var _player:Player=self.get_parent()
# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	update_value()


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(_delta: float) -> void:
	if _player==null:
		return
	if _player.is_dead:
		visible=false
		return
	
	visible=(CameraManager.current_camera_id==-1)
	if visible==false:
		return
	
	update_value()

func update_value():
	value=_player.current_health /_player.max_health *max_value
