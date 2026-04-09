extends SkillComponent
class_name SkillDealDamage

@export var damage:float=10.0
# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	pass # Replace with function body.

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	pass
	
func _activate(context:SkillContext):
	var targets=context.targets
	for target in targets:
		if target is Entity:
			target.bear_damage(damage)
	

	
	
	
	
	
	
