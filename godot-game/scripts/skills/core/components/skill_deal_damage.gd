extends SkillComponent
class_name SkillDealDamage

## 伤害组件
## 对 SkillContext.targets 中的所有 Entity 造成伤害
##
## 典型使用：配合 SkillGetTarget 组件，形成"检测目标→造成伤害"的技能链
##
## 示例技能链：
##   Skill (攻击技能)
##   ├── SkillGetTarget (检测前方敌人)
##   ├── SkillAnimationRunner (播放攻击动画)
##   └── SkillDealDamage (造成伤害)

@export var damage: float = 10.0          ## 基础伤害值

func _activate(context: SkillContext) -> void:
	var targets = context.targets
	for target in targets:
		if target is Entity:
			target.bear_damage(damage)  ## 调用 Entity.bear_damage() 处理伤害、特效、死亡
