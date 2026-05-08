class_name SkillComponent
extends Node

## 技能组件基类
## Skill 由多个 SkillComponent 组合而成
## 每个组件在技能激活时依次执行，负责具体效果（伤害、位移、特效等）

## 执行延迟时间（秒），0 表示立即执行
@export var execution_delay_time: float = 0

## 激活组件（入口）
## 支持延迟执行
## @param context: 技能执行上下文
func activate(context: SkillContext) -> void:
	if execution_delay_time > 0:
		await get_tree().create_timer(execution_delay_time).timeout
	_activate(context)

## 实际执行逻辑（子类重写此函数）
func _activate(_context: SkillContext) -> void:
	print("activate component:", self.name)
