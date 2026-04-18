extends Button


# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	modulate=Color(1.0, 1.0, 1.0)

func _on_pressed() -> void:
	if modulate==Color(0.5, 1.0, 0.7):
		modulate=Color(1.0, 1.0, 1.0)
	else:
		modulate=Color(0.5, 1.0, 0.7)
