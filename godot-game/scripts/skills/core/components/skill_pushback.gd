extends SkillComponent
class_name SkillPushback

@export var pushback_distance:float=10
@export var duration:float=0.1

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	pass

func _activate(context:SkillContext):
	var caster=context.caster
	var face_dir
	if caster.animated_sprite.flip_h:
		face_dir=1
	else:
		face_dir=-1
	var push_dir=Vector2(face_dir,0).normalized()
	# 用 external_velocity 代替 tween position，与 move_and_slide 兼容
	var pushback_speed=pushback_distance/duration
	caster.external_velocity=push_dir*pushback_speed
	caster.external_velocity_decay=1.0/duration
	
