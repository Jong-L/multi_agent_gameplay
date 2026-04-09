extends VBoxContainer
class_name PauseMenu

@export var resume_btn:Button
@export var title_btn:Button

func _ready() -> void:
	process_mode=Node.PROCESS_MODE_ALWAYS

func _on_resume_button_pressed() -> void:
	EventBus.game_paused.emit(false)
	get_tree().paused=false
	self.hide()


func _on_title_button_pressed() -> void:
	get_tree().paused=false
	get_tree().change_scene_to_file("res://assets/scenes/home_scene.tscn")
