class_name SkillPushback
extends SkillComponent

## 击退组件
## 对施法者施加瞬时推力（后坐力效果）
## 使用 Entity.external_velocity 实现，与 move_and_slide 兼容
##
## 物理原理：
##   - 初速度 = pushback_distance / duration
##   - 衰减率 = 1.0 / duration（duration 秒后速度归零）
##   - 位移 ≈ pushback_distance（近似匀速减速运动）

@export var pushback_distance: float = 10    ## 击退距离（像素）
@export var duration: float = 0.1            ## 击退持续时间（秒）

func _activate(context: SkillContext) -> void:
	var caster = context.caster
	
	## 击退方向 = 施法者朝向的反方向
	var face_dir = 1 if caster.animated_sprite.flip_h else -1
	var push_dir = Vector2(face_dir, 0).normalized()
	
	var pushback_speed = pushback_distance / duration
	caster.external_velocity = push_dir * pushback_speed
	caster.external_velocity_decay = 1.0 / duration
