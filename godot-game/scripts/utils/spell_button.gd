extends TextureButton
class_name SpellButton

## 技能按钮 UI
## 显示：技能图标、冷却进度、剩余时间、按键绑定
##
## 交互方式：
##   - 鼠标点击
##   - 键盘快捷键（通过 binded_key 设置）
##
## 状态显示：
##   - 冷却中：进度条 + 时间数字
##   - 可用：正常显示

var skill: Skill = null                       ## 绑定的技能

@export var icon: TextureRect                 ## 技能图标
@export var progress_bar: TextureProgressBar  ## 冷却进度条
@export var cooldown_label: Label             ## 冷却时间文字
@export var keybind_label: Label              ## 按键绑定文字

var binded_key: String = "":
	set(key):
		binded_key = key
		shortcut = Shortcut.new()
		var input_key = InputEventKey.new()
		input_key.keycode = key.unicode_at(0)
		shortcut.events = [input_key]
		cooldown_label.text = ""
		keybind_label.text = key

func _process(delta: float) -> void:
	if skill == null:
		return
	
	## 更新冷却状态
	disabled = skill.current_cooldown > 0
	progress_bar.value = skill.current_cooldown / skill.cooldown * progress_bar.max_value
	
	if disabled:
		cooldown_label.text = "%3.1f" % skill.current_cooldown
	else:
		cooldown_label.text = ""

func _on_pressed() -> void:
	if disabled:
		return
	
	disabled = true  ## 防止连点
	EventBus.player_cast_skill.emit(skill)

## 设置绑定的技能
## @param s: 技能实例
func set_skill(s: Skill) -> void:
	disabled = false
	skill = s
	icon.texture = skill.icon_texture if skill else null

## 获取绑定的技能
func get_skill() -> Skill:
	return skill

## 清除技能绑定
func clear_skill() -> void:
	skill = null
	icon.texture = null
	disabled = true

## 检查是否有技能
func has_skill() -> bool:
	return skill != null

## 检查是否在冷却中
func is_on_cooldown() -> bool:
	if skill == null:
		return false
	return skill.current_cooldown > 0

## 获取剩余冷却时间
## @return: 冷却时间（秒）
func get_remaining_cooldown() -> float:
	if skill == null:
		return 0.0
	return skill.current_cooldown

## 获取冷却进度
## @return: 0.0 ~ 1.0
func get_cooldown_progress() -> float:
	if skill == null or skill.cooldown <= 0:
		return 0.0
	return skill.current_cooldown / skill.cooldown

## 获取按键绑定
func get_binded_key() -> String:
	return binded_key

## 更新按钮状态（手动刷新）
func update_state() -> void:
	if skill == null:
		return
	disabled = skill.current_cooldown > 0
	progress_bar.value = get_cooldown_progress() * progress_bar.max_value

## 重置按钮
func reset() -> void:
	disabled = false
	if skill != null:
		skill.current_cooldown = 0.0
	progress_bar.value = 0.0
	cooldown_label.text = ""

## 播放冷却完成动画
func play_cooldown_finished_animation() -> void:
	var tween = create_tween()
	tween.tween_property(self, "modulate", Color(1.5, 1.5, 1.5), 0.1)
	tween.tween_property(self, "modulate", Color.WHITE, 0.2)

## 播放点击反馈动画
func play_click_feedback() -> void:
	var tween = create_tween()
	tween.tween_property(self, "scale", Vector2(0.95, 0.95), 0.05)
	tween.tween_property(self, "scale", Vector2.ONE, 0.05)

## 设置图标
## @param texture: 图标纹理
func set_icon(texture: Texture2D) -> void:
	icon.texture = texture

## 获取图标
func get_icon() -> Texture2D:
	return icon.texture

## 设置冷却进度条颜色
## @param color: 颜色
func set_progress_color(color: Color) -> void:
	progress_bar.tint_progress = color

## 获取技能名称
## @return: 技能名称，无技能返回空字符串
func get_skill_name() -> String:
	if skill == null:
		return ""
	return skill.name

## 检查是否可用（有技能且不在冷却）
func is_available() -> bool:
	return skill != null and skill.current_cooldown <= 0

## 触发技能（模拟点击）
func trigger() -> void:
	_on_pressed()

## 设置冷却标签可见性
## @param visible: 是否可见
func set_cooldown_label_visible(visible: bool) -> void:
	cooldown_label.visible = visible

## 设置进度条可见性
## @param visible: 是否可见
func set_progress_bar_visible(visible: bool) -> void:
	progress_bar.visible = visible

## 获取技能冷却时间
## @return: 总冷却时间（秒）
func get_skill_cooldown() -> float:
	if skill == null:
		return 0.0
	return skill.cooldown

## 检查是否是同一技能
## @param other_skill: 另一个技能
## @return: true 表示是同一技能
func is_same_skill(other_skill: Skill) -> bool:
	return skill == other_skill
