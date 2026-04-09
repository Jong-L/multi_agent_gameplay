extends SkillComponent
class_name SkillGetTarget

@export var _radius=28

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	pass


func _activate(context:SkillContext):
	var targets=check_colliders_around_position(context.caster,self._radius)
	context.targets=targets

func check_colliders_around_position(caster:Entity,radius:float)->Array:
	var shape=CircleShape2D.new()
	shape.radius=radius
	
	var query=PhysicsShapeQueryParameters2D.new()
	query.shape=shape
	query.transform.origin=caster.position
	query.collide_with_areas=true

	var space_state=caster.get_world_2d().direct_space_state
	var results=space_state.intersect_shape(query)
	var targets:Array[Entity]=[]
	
	var face_dir:Vector2=Vector2(0,0)
	if caster.animated_sprite!=null:
		if caster.animated_sprite.flip_h:
			face_dir=Vector2(-1,0)
		else:
			face_dir=Vector2(1,0)
	for result in results:
		var collider=result.collider
		var parent=collider.get_parent()
		if parent is Entity and not parent.is_dead:
			var to_target=(parent.position-caster.position).normalized()
			var fov=deg_to_rad(100)
			if to_target.dot(face_dir)>cos(fov/2):
				targets.push_back(parent)
	return targets

func create_debug_circle(radius:float):
	var points_num=32
	var line=Line2D.new()
	radius=radius
	line.width=1
	line.default_color=Color(1,0,0)
	var angle=TAU/points_num

	for i in range(points_num+1):
		var point=Vector2(cos(angle*i),sin(angle*i))
		line.add_point(point*radius)
	return line

func destroy_line(line:Line2D,time:float):
	await get_tree().create_timer(time).timeout
	if line!=null:
		line.queue_free()
