class_name SkillComponent
extends Node

@export var execution_delay_time: float = 0

#func _ready() -> void:
	#pass
#
#func _process(delta: float) -> void:
	#pass

func activate(context: SkillContext) -> void:
	if execution_delay_time > 0:
		await get_tree().create_timer(execution_delay_time).timeout
	_activate(context)

#实际执行逻辑
func _activate(_context: SkillContext) -> void:
	push_warning("SkillComponent._activate() not implemented: %s" % name)
	
	
