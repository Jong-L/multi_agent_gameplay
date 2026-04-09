class_name SkillController
extends Node

var skills:Array[Skill]=[]
var cooldowns:Dictionary={}
var entity:Entity

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	entity=get_parent()
	for child in get_children():
		if child is Skill: 
			skills.push_back(child)

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	for skill in cooldowns.keys():
		if cooldowns[skill]>0.0:
			cooldowns[skill]=max(0.0,cooldowns[skill]-delta)
			skill.current_cooldown=cooldowns[skill]

func trigger_skill_by_idx(idx:int):
	if skills.size()==0:return
	var skill=skills.get(idx)
	trigger_skill(skill)

func trigger_skill(skill:Skill):
	if skill==null:return
	
	if cooldowns.get(skill,0.0)>0.0:
		return
	skill.activate(entity)
	cooldowns[skill]=skill.cooldown
