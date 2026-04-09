class_name SkillComponent
extends Node

@export var execution_delay_time:float=0
# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	pass

func activate(context:SkillContext):
	if execution_delay_time>0:
		await get_tree().create_timer(execution_delay_time).timeout
	_activate(context)
	
	
func _activate(context:SkillContext):
	print("activate componnet:",self.name)
	
	
