extends PanelContainer
class_name SpellBar

" 玩家技能栏 UI
 显示多个技能按钮（SpellButton），包含图标、冷却、按键绑定

 可见性规则：
   - 主相机模式：隐藏
   - 玩家相机模式：只显示对应玩家的技能栏

 使用方式：
   - PlayScene._setup_player_uis() 动态创建
   - register_skill() 绑定技能"

var spell_buttons: Array[SpellButton] = []  
@export var button_container: Node           
var bound_player_id: int = -1                 

func _enter_tree() -> void:
	#收集按钮并分配按键绑定（1, 2, 3...）
	var i = 1
	for button in button_container.get_children():
		if button is SpellButton:
			spell_buttons.push_back(button)
			button.binded_key = str(i)
			i += 1

func _process(delta: float) -> void:
	## 根据相机状态控制可见性
	visible = bound_player_id >= 0 and CameraManager.should_show_player_ui(bound_player_id)

#注册技能到按钮
func register_skill(skill: Skill, idx: int) -> void:
	if idx >= 0 and idx < spell_buttons.size():
		spell_buttons[idx].set_skill(skill)

## 获取绑定的玩家 ID
func get_bound_player_id() -> int:
	return bound_player_id

## 设置绑定的玩家
## @param player_id: 玩家 ID
func bind_to_player(player_id: int) -> void:
	bound_player_id = player_id

## 获取技能按钮数量
func get_button_count() -> int:
	return spell_buttons.size()

## 获取指定索引的按钮
## @param idx: 按钮索引
func get_button(idx: int) -> SpellButton:
	if idx >= 0 and idx < spell_buttons.size():
		return spell_buttons[idx]
	return null

## 更新所有按钮状态
func update_all_buttons() -> void:
	for button in spell_buttons:
		button.update_state()

## 检查是否有技能在冷却中
## @return: true 表示有技能冷却中
func has_skill_on_cooldown() -> bool:
	for button in spell_buttons:
		if button.is_on_cooldown():
			return true
	return false

## 获取可用技能数量
## @return: 不在冷却中的技能数
func get_available_skill_count() -> int:
	var count = 0
	for button in spell_buttons:
		if not button.is_on_cooldown():
			count += 1
	return count

## 触发指定索引的技能（模拟按键）
## @param idx: 按钮索引
func trigger_skill_at(idx: int) -> void:
	var button = get_button(idx)
	if button != null:
		button._on_pressed()

## 设置技能栏透明度
## @param alpha: 透明度（0-1）
func set_bar_alpha(alpha: float) -> void:
	self_modulate = Color(1, 1, 1, alpha)

## 显示技能栏
func show_bar() -> void:
	visible = true

## 隐藏技能栏
func hide_bar() -> void:
	visible = false

## 检查技能栏是否可见
func is_bar_visible() -> bool:
	return visible

## 清空所有技能绑定
func clear_all_skills() -> void:
	for button in spell_buttons:
		button.clear_skill()

## 重置所有按钮状态
func reset() -> void:
	for button in spell_buttons:
		button.reset()

## 获取第一个可用技能的索引
## @return: 索引，无可用返回 -1
func get_first_available_skill_index() -> int:
	for i in range(spell_buttons.size()):
		if not spell_buttons[i].is_on_cooldown():
			return i
	return -1

## 检查指定技能是否可用
## @param idx: 按钮索引
## @return: true 表示可用
func is_skill_available(idx: int) -> bool:
	var button = get_button(idx)
	if button == null:
		return false
	return not button.is_on_cooldown()

## 获取技能冷却信息
## @return: 各技能冷却进度数组
func get_cooldown_info() -> Array[float]:
	var info: Array[float] = []
	for button in spell_buttons:
		info.append(button.get_cooldown_progress())
	return info

## 设置按钮交互状态
## @param enabled: true 表示可交互
func set_buttons_enabled(enabled: bool) -> void:
	for button in spell_buttons:
		button.disabled = not enabled

## 高亮指定按钮
## @param idx: 按钮索引
func highlight_button(idx: int) -> void:
	for i in range(spell_buttons.size()):
		if i == idx:
			spell_buttons[i].modulate = Color(1.2, 1.2, 1.2)  ## 高亮
		else:
			spell_buttons[i].modulate = Color.WHITE  ## 恢复

## 清除所有高亮
func clear_highlight() -> void:
	for button in spell_buttons:
		button.modulate = Color.WHITE

## 播放技能栏显示动画
func play_show_animation() -> void:
	var tween = create_tween()
	tween.tween_property(self, "modulate:a", 1.0, 0.2)

## 播放技能栏隐藏动画
func play_hide_animation() -> void:
	var tween = create_tween()
	tween.tween_property(self, "modulate:a", 0.0, 0.2)

## 检查是否绑定玩家
func is_bound() -> bool:
	return bound_player_id >= 0

## 解绑玩家
func unbind() -> void:
	bound_player_id = -1
	visible = false

## 获取已绑定技能数量
## @return: 已绑定技能的按钮数
func get_registered_skill_count() -> int:
	var count = 0
	for button in spell_buttons:
		if button.has_skill():
			count += 1
	return count

## 检查是否所有技能都在冷却
## @return: true 表示全部冷却中
func are_all_skills_on_cooldown() -> bool:
	if spell_buttons.is_empty():
		return false
	for button in spell_buttons:
		if not button.is_on_cooldown():
			return false
	return true

## 获取最短剩余冷却时间
## @return: 最短冷却时间（秒），无技能返回 0
func get_shortest_cooldown() -> float:
	var shortest = INF
	for button in spell_buttons:
		var cd = button.get_remaining_cooldown()
		if cd < shortest:
			shortest = cd
	return 0.0 if shortest == INF else shortest

## 设置按钮大小
## @param size: 新大小
func set_button_size(size: Vector2) -> void:
	for button in spell_buttons:
		button.custom_minimum_size = size

## 设置按钮间距
## @param separation: 间距（像素）
func set_button_separation(separation: int) -> void:
	if button_container is BoxContainer:
		button_container.add_theme_constant_override("separation", separation)
