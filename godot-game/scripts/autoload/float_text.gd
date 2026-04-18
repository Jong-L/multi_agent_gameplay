extends Node

## 浮动文字管理器（Autoload 单例）
## 使用方式：
##   FloatText.show_damage_text("100", global_position, Color.RED)

@onready var damage_font = preload("res://resources/damage_font.tres")

const Z_INDEX := 12                         ## 确保在最上层显示
const INITIAL_SCALE := Vector2(0.15, 0.15)  ## 初始缩放
const FINAL_SCALE := Vector2(0.07, 0.07)    ## 最终缩放
const ANIMATION_DURATION := 0.5             ## 动画时长（秒）
const FLOAT_HEIGHT := 12.0                  ## 飘升高度（像素）

## 显示伤害文字
## @param damage: 伤害数值（字符串）
## @param position: 世界坐标位置
## @param color: 文字颜色
func show_damage_text(damage: String, position: Vector2, color: Color) -> void:
	var label = Label.new()
	label.text = damage
	label.z_index = Z_INDEX
	label.scale = INITIAL_SCALE
	
	## 随机偏移，避免多个数字重叠
	var x_random = randf_range(-10, 10)
	var y_random = randf_range(10, 15)
	var vector_random = Vector2(x_random, y_random)
	
	## 配置标签样式
	label.label_settings = LabelSettings.new()
	label.label_settings.font = damage_font
	label.label_settings.font_color = color
	label.label_settings.font_size = 100
	label.label_settings.outline_size = 1
	label.label_settings.outline_color = Color.BLACK  ## 黑色描边
	
	add_child(label)
	
	## 居中定位
	var offset = label.size / 2
	label.position = position - offset - vector_random
	label.pivot_offset = label.size / 2
	
	## 创建动画：飘升 + 缩放淡出
	var tween = create_tween()
	tween.tween_property(label, "position:x", label.position.x + randf_range(-5, 5), ANIMATION_DURATION)\
		.set_trans(Tween.TRANS_EXPO).set_ease(Tween.EASE_OUT)
	tween.parallel()
	tween.tween_property(label, "position:y", label.position.y - FLOAT_HEIGHT, ANIMATION_DURATION)\
		.set_trans(Tween.TRANS_EXPO).set_ease(Tween.EASE_OUT)
	tween.parallel()
	tween.tween_property(label, "scale", FINAL_SCALE, ANIMATION_DURATION)\
		.set_ease(Tween.EASE_IN)
	
	await tween.finished
	label.queue_free()

	
	
	
	
	
	
	
