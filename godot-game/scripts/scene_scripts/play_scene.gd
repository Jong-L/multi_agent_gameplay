extends Node
class_name PlayScene

@export var screen_transition:ColorRect
@export var pause_menu:PauseMenu

func _ready() -> void:
	EventBus.game_paused.connect(_handle_pause)
func _handle_game_over(player:Player):
	var tween=fade_in()
	await tween.finished
	player.current_animation_wrapper=null
	player.is_dead=false
	player.position=player.spawn_position
	player.current_health=player.max_health
	
	tween=fade_out()
	await tween.finished

func _on_player_player_died(player: Player) -> void:
	_handle_game_over(player)

func fade_out():
	var tween=create_tween()
	tween.tween_property(
		screen_transition,
		"color:a",
		0.0,
		0.4
	).set_trans(Tween.TRANS_LINEAR).set_ease(Tween.EASE_OUT)
	
	return tween

func fade_in():
	var tween=create_tween()
	tween.tween_property(
		screen_transition,
		"color:a",
		1.0,
		0.5
	).set_trans(Tween.TRANS_LINEAR).set_ease(Tween.EASE_IN)
	
	return tween

func _on_pause_button_pressed() -> void:
	EventBus.game_paused.emit(true)
	pause_menu.show()
	get_tree().paused=true
	
func _handle_pause(paused:bool):
	if paused:
		screen_transition.color=Color(0,0,0,0.5)
	else:
		screen_transition.color=Color(0,0,0,0)
	
