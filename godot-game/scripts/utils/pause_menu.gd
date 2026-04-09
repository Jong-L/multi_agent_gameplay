extends VBoxContainer
class_name PauseMenu

## 暂停菜单 UI
## 游戏暂停时显示的菜单，包含继续游戏和返回标题选项
##
## 注意：
##   - process_mode = ALWAYS，确保暂停时仍可响应
##   - 通过 EventBus 发送暂停状态变更

@export var resume_btn: Button                ## 继续游戏按钮
@export var title_btn: Button                 ## 返回标题按钮

func _ready() -> void:
	process_mode = Node.PROCESS_MODE_ALWAYS

func _on_resume_button_pressed() -> void:
	EventBus.game_paused.emit(false)
	get_tree().paused = false
	self.hide()

func _on_title_button_pressed() -> void:
	get_tree().paused = false
	get_tree().change_scene_to_file("res://assets/scenes/home_scene.tscn")

## 显示暂停菜单
func show_menu() -> void:
	show()
	if resume_btn != null:
		resume_btn.grab_focus()

## 隐藏暂停菜单
func hide_menu() -> void:
	hide()

## 检查菜单是否显示
func is_menu_visible() -> bool:
	return visible

## 设置继续按钮文本
## @param text: 新文本
func set_resume_text(text: String) -> void:
	if resume_btn != null:
		resume_btn.text = text

## 设置返回标题按钮文本
## @param text: 新文本
func set_title_text(text: String) -> void:
	if title_btn != null:
		title_btn.text = text

## 禁用返回标题按钮
func disable_title_button() -> void:
	if title_btn != null:
		title_btn.disabled = true

## 启用返回标题按钮
func enable_title_button() -> void:
	if title_btn != null:
		title_btn.disabled = false

## 播放显示动画
func play_show_animation() -> void:
	modulate = Color(1, 1, 1, 0)
	var tween = create_tween()
	tween.tween_property(self, "modulate:a", 1.0, 0.2)

## 播放隐藏动画
func play_hide_animation() -> void:
	var tween = create_tween()
	tween.tween_property(self, "modulate:a", 0.0, 0.2)
	tween.tween_callback(hide)

## 切换菜单显示状态
func toggle() -> void:
	if visible:
		hide_menu()
	else:
		show_menu()

## 获取继续按钮
func get_resume_button() -> Button:
	return resume_btn

## 获取返回标题按钮
func get_title_button() -> Button:
	return title_btn

## 设置按钮焦点
## @param button_name: "resume" 或 "title"
func set_focus(button_name: String) -> void:
	match button_name:
		"resume":
			if resume_btn != null:
				resume_btn.grab_focus()
		"title":
			if title_btn != null:
				title_btn.grab_focus()

## 重置菜单状态
func reset() -> void:
	modulate = Color.WHITE
	if resume_btn != null:
		resume_btn.disabled = false
	if title_btn != null:
		title_btn.disabled = false

## 添加自定义按钮
## @param button: 按钮实例
## @param index: 插入位置（-1 表示末尾）
func add_custom_button(button: Button, index: int = -1) -> void:
	if index < 0:
		add_child(button)
	else:
		add_child(button)
		move_child(button, index)

## 移除自定义按钮
## @param button: 按钮实例
func remove_custom_button(button: Button) -> void:
	remove_child(button)

## 清除所有自定义按钮
func clear_custom_buttons() -> void:
	for child in get_children():
		if child != resume_btn and child != title_btn:
			remove_child(child)

## 设置菜单背景透明度
## @param alpha: 透明度（0-1）
func set_background_alpha(alpha: float) -> void:
	modulate = Color(1, 1, 1, alpha)

## 获取菜单尺寸
func get_menu_size() -> Vector2:
	return size

## 设置菜单位置
## @param pos: 新位置
func set_menu_position(pos: Vector2) -> void:
	position = pos

## 居中菜单
func center_menu() -> void:
	if get_viewport() != null:
		var viewport_size = get_viewport().get_visible_rect().size
		position = (viewport_size - size) / 2

## 检查是否有子菜单打开
## @return: true 表示有子菜单
func has_submenu_open() -> bool:
	for child in get_children():
		if child is Popup or child is Window:
			if child.visible:
				return true
	return false

## 关闭所有子菜单
func close_all_submenus() -> void:
	for child in get_children():
		if child is Popup or child is Window:
			child.hide()

## 设置按钮样式
## @param style: 样式资源
func set_button_style(style: StyleBox) -> void:
	if resume_btn != null:
		resume_btn.add_theme_stylebox_override("normal", style)
	if title_btn != null:
		title_btn.add_theme_stylebox_override("normal", style)

## 恢复默认按钮样式
func reset_button_styles() -> void:
	if resume_btn != null:
		resume_btn.remove_theme_stylebox_override("normal")
	if title_btn != null:
		title_btn.remove_theme_stylebox_override("normal")  ## 移除继续和返回按钮的自定义样式覆盖，恢复默认外观
