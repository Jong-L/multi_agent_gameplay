class_name Skill
extends Node

@export var cooldown:float=1.5
@export var icon_texture:Texture2D

var current_cooldown:float

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	pass

func activate(entity:Entity):
	var context=SkillContext.new(entity,self)
	_activate(context)

func _activate(context:SkillContext):
	_activate_component(context)

func _activate_component(context:SkillContext):
	for child in self.get_children():
		if child is SkillComponent:
			child.activate(context)
