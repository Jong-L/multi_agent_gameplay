extends SkillComponent
class_name SkillGetTarget

## 目标获取组件
## 在施法者周围圆形范围内检测敌人，结合扇形视野（FOV）过滤
##
## 检测条件：
##   1. 必须是 Entity
##   2. 必须存活（not is_dead）
##   3. 必须在施法者前方扇形视野内（默认 FOV = 100°）
##
## 检测结果：填充到 context.targets，供后续组件使用

@export var _radius: float = 18           ## 检测半径（像素）
@export var _fov_degrees: float = 100.0   ## 视野角度（度）

func _activate(context: SkillContext) -> void:
	var targets = _check_colliders_around_position(context.caster, _radius)
	context.targets = targets

## 圆形范围 + 扇形视野检测
## @param caster: 施法者
## @param radius: 检测半径
## @return: 符合条件的 Entity 数组
func _check_colliders_around_position(caster: Entity, radius: float) -> Array[Entity]:
	var shape = CircleShape2D.new()
	shape.radius = radius
	
	var query = PhysicsShapeQueryParameters2D.new()
	query.shape = shape
	query.collide_with_areas = true
	query.transform.origin = caster.position
	
	var space_state = caster.get_world_2d().direct_space_state
	var results = space_state.intersect_shape(query)
	
	var targets: Array[Entity] = []
	
	## 确定施法者朝向
	var face_dir = Vector2.RIGHT
	if caster.animated_sprite != null:
		face_dir = Vector2.LEFT if caster.animated_sprite.flip_h else Vector2.RIGHT
	
	var fov = deg_to_rad(_fov_degrees)
	var cos_half_fov = cos(fov / 2)
	
	for result in results:
		var collider = result.collider
		var parent = collider.get_parent()
		if parent is Entity and not parent.is_dead:
			var to_target = (parent.position - caster.position).normalized()
			## 点积 > cos(fov/2) 表示在扇形范围内
			if to_target.dot(face_dir) > cos_half_fov:
				targets.push_back(parent)
	
	return targets

## 创建调试圆形（可视化检测范围）
## 用于开发调试时显示技能范围
func create_debug_circle(radius: float) -> Line2D:
	var points_num = 32
	var line = Line2D.new()
	line.width = 1
	line.default_color = Color.RED
	var angle = TAU / points_num
	
	for i in range(points_num + 1):
		var point = Vector2(cos(angle * i), sin(angle * i))
		line.add_point(point * radius)
	return line

## 销毁调试圆形
func destroy_line(line: Line2D, time: float) -> void:
	await get_tree().create_timer(time).timeout
	if line != null:
		line.queue_free()  ## 延迟指定时间后销毁调试线条，用于临时显示检测范围后自动清理
