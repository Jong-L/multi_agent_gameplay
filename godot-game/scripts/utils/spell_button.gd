extends TextureButton
class_name SpellButton

var skill:Skill=null

@export var icon:TextureRect
@export var progress_bar:TextureProgressBar
@export var cooldown_label:Label
@export var keybind_label:Label

var binded_key:String="":
	set(key):
		binded_key=key
		shortcut=Shortcut.new()
		var input_key=InputEventKey.new()
		input_key.keycode=key.unicode_at(0)
		shortcut.events=[input_key]
		cooldown_label.text=""
		keybind_label.text=key
		
func _process(delta: float) -> void:
	if skill ==null:return
	disabled=skill.current_cooldown>0
	progress_bar.value=skill.current_cooldown/skill.cooldown*progress_bar.max_value
	if disabled:
		cooldown_label.text="%3.1f" % skill.current_cooldown
	else:
		cooldown_label.text=""
	
func _on_pressed() -> void:
	if disabled:return
	
	disabled=true
	EventBus.player_cast_skill.emit(skill)

func set_skill(skill:Skill):
	disabled=false
	self.skill=skill
	icon.texture=skill.icon_texture
	
