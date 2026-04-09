extends PanelContainer
class_name SpellBar

var spell_buttons:Array[SpellButton]=[]

@export var button_container:Node

func _enter_tree() -> void:
	var i=1
	for button in button_container.get_children():
		if button is SpellButton:
			spell_buttons.push_back(button)
			button.binded_key=str(i)
			i+=1

func register_skill(skill:Skill,idx:int):
	if idx>=0 and idx<=spell_buttons.size():
		var spell_button=spell_buttons[idx]
		spell_button.set_skill(skill)
