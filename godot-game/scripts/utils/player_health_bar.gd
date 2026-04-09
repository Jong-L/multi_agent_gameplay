extends TextureProgressBar

## 玩家血条 UI
## 实时同步玩家血量，并根据相机状态控制可见性
##
## 可见性规则：
##   - 主相机模式：隐藏所有血条
##   - 玩家相机模式：只显示当前相机对应玩家的血条
##
## 使用方式：
##   - PlayScene._setup_player_uis() 动态创建并绑定 player
##   - 每帧自动更新

@export var player: Player                    ## 绑定的玩家实例

func _process(delta: float) -> void:
	if player == null:
		return
	
	## 同步血量百分比
	var health_percent = player.current_health / player.max_health
	self.value = health_percent * self.max_value
	
	## 根据相机状态控制可见性
	visible = CameraManager.should_show_player_ui(player.player_id)

## 获取当前血量百分比
## @return: 0.0 ~ 1.0
func get_health_percent() -> float:
	if player == null:
		return 0.0
	return player.current_health / player.max_health

## 检查是否低血量
## @param threshold: 低血量阈值（默认 0.25）
## @return: true 表示低血量
func is_low_health(threshold: float = 0.25) -> bool:
	return get_health_percent() < threshold

## 获取绑定的玩家
func get_player() -> Player:
	return player

## 设置绑定的玩家
## @param p: 玩家实例
func set_player(p: Player) -> void:
	player = p

## 更新血条颜色（根据血量百分比）
## 可以扩展为动态改变 progress 纹理色调
func update_color() -> void:
	var percent = get_health_percent()
	if percent > 0.5:
		modulate = Color.WHITE  ## 健康状态
	elif percent > 0.25:
		modulate = Color.YELLOW  ## 警告状态
	else:
		modulate = Color.RED     ## 危险状态

## 显示血条（强制可见）
func show_bar() -> void:
	visible = true

## 隐藏血条（强制隐藏）
func hide_bar() -> void:
	visible = false

## 检查血条是否可见
func is_bar_visible() -> bool:
	return visible

## 获取玩家 ID
## @return: 玩家 ID，未绑定返回 -1
func get_player_id() -> int:
	if player == null:
		return -1
	return player.player_id

## 检查是否已绑定玩家
func is_bound() -> bool:
	return player != null

## 获取玩家当前血量
## @return: 当前血量，未绑定返回 0
func get_current_health() -> float:
	if player == null:
		return 0.0
	return player.current_health

## 获取玩家最大血量
## @return: 最大血量，未绑定返回 0
func get_max_health() -> float:
	if player == null:
		return 0.0
	return player.max_health

## 检查玩家是否死亡
## @return: true 表示已死亡
func is_player_dead() -> bool:
	if player == null:
		return true
	return player.is_dead

## 检查玩家是否存活
## @return: true 表示存活
func is_player_alive() -> bool:
	if player == null:
		return false
	return not player.is_dead

## 刷新显示（手动调用更新）
func refresh() -> void:
	if player == null:
		return
	var health_percent = player.current_health / player.max_health
	self.value = health_percent * self.max_value
	update_color()  ## 手动刷新血条显示，包括数值更新和颜色调整，用于初始化或特殊事件后的强制更新

## 播放受伤动画效果
## 可以扩展为缩放、闪烁等视觉反馈
func play_damage_effect() -> void:
	var tween = create_tween()
	tween.tween_property(self, "scale", Vector2(1.1, 1.1), 0.05)
	tween.tween_property(self, "scale", Vector2(1.0, 1.0), 0.05)  ## 播放血条受到攻击时的视觉反馈动画，通过短暂的缩放效果提示玩家受到伤害

## 播放治疗动画效果
func play_heal_effect() -> void:
	var tween = create_tween()
	tween.tween_property(self, "modulate", Color.GREEN, 0.1)
	tween.tween_property(self, "modulate", Color.WHITE, 0.2)  ## 播放血条受到治疗时的视觉反馈动画，通过颜色变化提示玩家恢复生命值

## 获取血条在世界空间的位置
## @return: 世界坐标
func get_world_position() -> Vector2:
	return global_position  ## 返回血条当前在世界空间中的位置，用于特效定位或其他需要知道血条位置的场景

## 设置血条偏移
## @param offset: 偏移量
func set_bar_offset(offset: Vector2) -> void:
	position += offset  ## 调整血条相对于锚点的位置偏移，用于适配不同分辨率或布局需求

## 重置血条状态
func reset() -> void:
	if player != null:
		self.value = self.max_value
	modulate = Color.WHITE
	scale = Vector2.ONE  ## 重置血条的所有状态到初始值，包括血量显示、颜色和缩放，用于游戏重新开始或玩家复活时

## 检查是否需要警告（低血量闪烁）
## @return: true 需要警告
func needs_warning() -> bool:
	return is_low_health(0.25) and not is_player_dead()  ## 检查血条是否应该显示低血量警告，当血量低于 25% 且玩家存活时返回 true，用于触发闪烁或其他警告效果

## 更新警告效果（低血量闪烁）
func update_warning_effect(delta: float) -> void:
	if not needs_warning():
		modulate = Color.WHITE
		return
	
	## 简单的闪烁效果
	var alpha = 0.5 + 0.5 * sin(Time.get_time_dict_from_system()["second"] * 10)
	modulate = Color(1, 0, 0, alpha)  ## 在低血量状态下更新血条的闪烁警告效果，通过随时间变化的透明度创建脉冲视觉效果，提醒玩家注意血量

## 获取血条尺寸
## @return: 尺寸向量
func get_bar_size() -> Vector2:
	return size  ## 返回血条控件的尺寸大小，用于布局计算或与其他 UI 元素对齐

## 设置血条尺寸
## @param new_size: 新尺寸
func set_bar_size(new_size: Vector2) -> void:
	size = new_size  ## 设置血条控件的新尺寸，用于动态调整血条大小或适配不同屏幕分辨率

## 检查是否满血
## @return: true 表示满血
func is_full_health() -> bool:
	return get_health_percent() >= 1.0  ## 检查玩家当前是否处于满血状态，用于判断是否需要治疗或显示满血特效

## 获取血量文字表示
## @return: "当前/最大" 格式
func get_health_text() -> String:
	if player == null:
		return "0/0"
	return "%d/%d" % [int(player.current_health), int(player.max_health)]  ## 返回格式化的血量文字，显示当前血量和最大血量，用于 UI 文字显示或调试信息

## 设置血条样式（通过主题）
## @param style: 样式资源
func set_bar_style(style: StyleBox) -> void:
	add_theme_stylebox_override("fill", style)  ## 通过主题覆盖设置血条的填充样式，允许自定义血条的外观风格

## 清除血条样式
func clear_bar_style() -> void:
	remove_theme_stylebox_override("fill")  ## 移除血条的自定义样式覆盖，恢复默认外观

## 获取血条中心点
## @return: 中心点位置
func get_center() -> Vector2:
	return global_position + size / 2  ## 返回血条的中心点世界坐标，用于特效定位或作为参考点
