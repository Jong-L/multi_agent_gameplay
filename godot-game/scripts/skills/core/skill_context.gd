class_name SkillContext
extends RefCounted

var caster:Entity
var skill:Skill
var targets:Array[Variant]=[]
func _init(caster:Entity,skill:Skill):
	self.caster=caster
	self.skill=skill

func get_target_position(idx:int)->Vector2:
	var target=targets[idx]
	if target is Entity:
		return target.global_position
	elif target is Vector2:
		return target
	else:
		return Vector2.ZERO
