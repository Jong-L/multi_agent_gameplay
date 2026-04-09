extends SkillComponent
class_name DealDamageGroup

var sub_components:Array[SkillComponent]=[]

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	for child in get_children():
		if child is SkillComponent:
			sub_components.push_back(child)

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	pass

func _activate(context:SkillContext):
	if execution_delay_time>0:
		await get_tree().create_timer(execution_delay_time).timeout
	for component in sub_components:
		component.activate(context)
	
