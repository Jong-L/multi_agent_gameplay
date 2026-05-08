class_name Skill
extends Node

@export var cooldown: float = 1.5           ## 技能冷却时间（秒）
@export var icon_texture: Texture2D         ## 技能图标（SpellBar 显示）
@export var dmg_multiplier:float=1.0             #伤害倍率
var current_cooldown: float                 ## 当前剩余冷却（SkillController 管理）

#func _ready() -> void:
	#pass
#
#func _process(delta: float) -> void:
	#pass


func activate(entity: Entity) -> void:
	var context = SkillContext.new(entity, self)
	_activate(context)


func _activate(context: SkillContext) -> void:
	_activate_component(context)

func _activate_component(context: SkillContext) -> void:
	for child in self.get_children():
		if child is SkillComponent:
			child.activate(context)
