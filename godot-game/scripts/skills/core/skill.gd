class_name Skill
extends Node

## 技能基类
## 采用组合模式，由多个 SkillComponent 组成
## 激活时依次执行所有子组件
##
## 架构位置：
##   - 父节点：SkillController（管理冷却和触发）
##   - 子节点：SkillComponent（具体效果组件）
##   - 运行时：通过 SkillContext 传递上下文
##
## 典型组件链：
##   SkillGetTarget → SkillAnimationRunner → SkillDealDamage

@export var cooldown: float = 1.5           ## 技能冷却时间（秒）
@export var icon_texture: Texture2D         ## 技能图标（SpellBar 显示）

var current_cooldown: float                 ## 当前剩余冷却（SkillController 管理）

func _ready() -> void:
	pass

func _process(delta: float) -> void:
	pass

## 激活技能（外部入口）
## @param entity: 施法者（Player 或 Enemy）
func activate(entity: Entity) -> void:
	var context = SkillContext.new(entity, self)
	_activate(context)

## 内部激活（子类可重写添加前置逻辑）
func _activate(context: SkillContext) -> void:
	_activate_component(context)

## 执行所有子组件
func _activate_component(context: SkillContext) -> void:
	for child in self.get_children():
		if child is SkillComponent:
			child.activate(context)
