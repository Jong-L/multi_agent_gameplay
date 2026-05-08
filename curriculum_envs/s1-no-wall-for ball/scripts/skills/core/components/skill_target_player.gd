class_name SkillTargetPlayer
extends SkillComponent

## 目标筛选组件（玩家专用）
## 在施法者周围检测存活玩家，用于敌人 AI 的攻击目标获取
##
## 检测条件：
##   1. 属于 "player" 组
##   2. 存活状态（not is_dead）
##   3. 距离在 check_range 内
##   4. 在施法者前方扇形视野内（attack_fov）
##
## 与 SkillGetTarget 的区别：
##   - SkillGetTarget：物理检测（PhysicsShapeQuery），检测所有 Entity
##   - SkillTargetPlayer：遍历分组，专用于敌人检测玩家

@export var check_range: float          ## 检测半径（像素）
@export var attack_fov: float        ## 攻击视野角度（度）

func _activate(context: SkillContext) -> void:
	var caster = context.caster
	var targets: Array[Entity] = []

	for node in get_tree().get_nodes_in_group("player"):
		if not node is Player:
			continue
		if node.is_dead:
			continue

		var distance = caster.position.distance_to(node.position)
		if distance > check_range:
			continue

		## 扇形视野检查
		var face_dir = Vector2.RIGHT
		if caster.animated_sprite != null:
			face_dir = Vector2.LEFT if caster.animated_sprite.flip_h else Vector2.RIGHT

		var to_target = (node.position - caster.position).normalized()
		var fov = deg_to_rad(attack_fov)
		if to_target.dot(face_dir) > cos(fov / 2):
			targets.push_back(node)

	context.targets = targets  ## 将检测到的玩家目标填充到技能上下文中，供后续伤害组件使用
