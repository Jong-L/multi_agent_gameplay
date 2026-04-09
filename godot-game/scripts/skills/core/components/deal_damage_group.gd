class_name DealDamageGroup
extends SkillComponent

## 伤害组件组（容器组件）
## 依次激活所有子 SkillComponent，用于组合多个效果
##
## 典型使用场景：
##   DealDamageGroup
##   ├── SkillDealDamage (造成伤害)
##   ├── SkillPushback (施加击退)
##   └── SkillSpawnEffect (生成特效)

var sub_components: Array[SkillComponent] = []    ## 子组件数组（自动收集）

func _ready() -> void:
	for child in get_children():
		if child is SkillComponent:
			sub_components.push_back(child)

func _activate(context: SkillContext) -> void:
	if execution_delay_time > 0:
		await get_tree().create_timer(execution_delay_time).timeout
	for component in sub_components:
		component.activate(context)
