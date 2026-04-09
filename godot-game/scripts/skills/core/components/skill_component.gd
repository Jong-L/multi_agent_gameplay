class_name SkillComponent
extends Node

## 技能组件基类
## 所有技能效果的基类，采用组合模式实现技能系统
##
## 使用方式：
##   1. 继承此类创建具体组件（如 SkillDealDamage、SkillGetTarget）
##   2. 重写 _activate() 实现具体逻辑
##   3. 将组件作为 Skill 的子节点
##
## 执行流程：
##   Skill.activate() → SkillComponent.activate() → [delay] → _activate()

@export var execution_delay_time: float = 0    ## 执行延迟（秒），用于技能连招的时间差

func _ready() -> void:
	pass

func _process(delta: float) -> void:
	pass

## 激活组件（由 Skill 调用）
## 处理延迟逻辑后执行 _activate()
## @param context: 技能上下文
func activate(context: SkillContext) -> void:
	if execution_delay_time > 0:
		await get_tree().create_timer(execution_delay_time).timeout
	_activate(context)

## 实际执行逻辑（子类必须重写）
## @param context: 技能上下文
func _activate(context: SkillContext) -> void:
	push_warning("SkillComponent._activate() not implemented: %s" % name)
	
	
