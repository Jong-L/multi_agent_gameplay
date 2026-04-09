extends Node2D
class_name Pathfinding

@export var neigbour_check_radius=10
@export var separation_force:float=300

func find_path(target_position:Vector2):
	var shape=CircleShape2D.new()
	shape.radius=neigbour_check_radius
	
	var query=PhysicsShapeQueryParameters2D.new()
	query.shape=shape
	query.collide_with_areas=true
	query.collide_with_bodies=false
	query.transform.origin=global_position
	
	var space_state=get_world_2d().direct_space_state
	var results=space_state.intersect_shape(query)
	var neigbours:Array[Enemy]=[]
	if results.size()>0:
		for result in results:
			var collider=result.collider
			var parent=collider.get_parent()
			if parent is Enemy and parent !=self.get_parent():
				neigbours.push_back(parent)
	var separation_direction=separation(neigbours)
	return separation_direction*separation_force+(target_position-global_position)

func separation(neigbours:Array[Enemy]):
	var separation_vector=Vector2.ZERO
	
	for neigbour in neigbours:
		var to_me=global_position-neigbour.global_position
		var distance=to_me.length()
		
		if distance>0:
			separation_vector+=to_me.normalized()/distance
	
	return separation_vector
