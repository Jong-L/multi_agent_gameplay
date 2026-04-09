extends SkillComponent
class_name SkillTargetPlayer

@export var check_range:float=25
@export var attack_fov:float=100.0

func _activate(context:SkillContext):
	var caster=context.caster
	var targets:Array[Entity]=[]
	
	# 遍历所有活着的玩家，距离+扇形校验
	for node in get_tree().get_nodes_in_group("player"):
		if not node is Player:
			continue
		if node.is_dead:
			continue
		
		var distance=caster.position.distance_to(node.position)
		# 距离检查
		if distance>check_range:
			continue
		
		# 扇形方向检查：玩家必须在敌人前方扇形内
		var face_dir=Vector2(1,0)
		if caster.animated_sprite!=null:
			if caster.animated_sprite.flip_h:
				face_dir=Vector2(-1,0)
		
		var to_target=(node.position-caster.position).normalized()
		var fov=deg_to_rad(attack_fov)
		if to_target.dot(face_dir)>cos(fov/2):
			targets.push_back(node)
	
	context.targets=targets
