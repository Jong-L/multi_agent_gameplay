class_name SkillManifest
extends Node2D

## 技能 Manifest 基类
## 由 SkillSpawnManifest 组件实例化并激活
## 负责具体的技能效果呈现（如攻击判定、范围特效、持续时间等）
##
## 与 SkillComponent 的区别：
##   - Manifest 是独立节点，可拥有 _process 生命周期
##   - 适合需要动画、持续判定、延迟效果的技能
##   - 需要自行管理生命周期（通常动画结束后 queue_free）
##
## 生命周期：
##   SkillSpawnManifest._activate() → instantiate() → activate() → _activate()
##   → [子类逻辑] → [动画/效果结束] → queue_free()

## 激活入口
## @param context: 技能上下文
func activate(context: SkillContext) -> void:
	_activate(context)

## 实际执行逻辑（子类必须重写）
## @param context: 技能上下文
func _activate(context: SkillContext) -> void:
	push_warning("SkillManifest._activate() not implemented: %s" % name)
