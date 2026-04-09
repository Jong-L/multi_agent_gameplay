extends PanelContainer
class_name SpellBar

var spell_buttons: Array[SpellButton] = []

@export var button_container: Node

## 绑定的玩家ID，-1表示未绑定
var bound_player_id: int = -1

func _enter_tree() -> void:
	var i = 1
	for button in button_container.get_children():
		if button is SpellButton:
			spell_buttons.push_back(button)
			button.binded_key = str(i)
			i += 1

func _process(delta: float) -> void:
	# 根据相机状态决定可见性
	if bound_player_id >= 0:
		visible = CameraManager.should_show_player_ui(bound_player_id)
	else:
		visible = false

func register_skill(skill: Skill, idx: int):
	if idx >= 0 and idx < spell_buttons.size():
		var spell_button = spell_buttons[idx]
		spell_button.set_skill(skill)
