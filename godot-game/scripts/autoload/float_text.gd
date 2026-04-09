extends Node

## 浮动文字管理器（Autoload 单例）
## 负责在场景中显示伤害数字、治疗数值等浮动文本
##
## 视觉效果：
##   - 初始缩放 0.15，淡出至 0.07
##   - 飘升 12 像素 + 随机水平偏移
##   - 黑色描边增强可读性
##   - 动画时长 0.5 秒
##
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

## 显示治疗文字（绿色）
## @param heal: 治疗数值
## @param position: 世界坐标位置
func show_heal_text(heal: String, position: Vector2) -> void:
	show_damage_text(heal, position, Color.GREEN)

## 显示暴击文字（黄色，更大）
## @param damage: 伤害数值
## @param position: 世界坐标位置
func show_crit_text(damage: String, position: Vector2) -> void:
	var label = Label.new()
	label.text = damage + "!"
	label.z_index = Z_INDEX + 1  ## 暴击显示在最上层
	label.scale = Vector2(0.2, 0.2)  ## 更大
	
	var x_random = randf_range(-10, 10)
	var y_random = randf_range(10, 15)
	var vector_random = Vector2(x_random, y_random)
	
	label.label_settings = LabelSettings.new()
	label.label_settings.font = damage_font
	label.label_settings.font_color = Color.YELLOW
	label.label_settings.font_size = 120  ## 更大字体
	label.label_settings.outline_size = 2
	label.label_settings.outline_color = Color.RED  ## 红色描边
	
	add_child(label)
	
	var offset = label.size / 2
	label.position = position - offset - vector_random
	label.pivot_offset = label.size / 2
	
	var tween = create_tween()
	tween.tween_property(label, "position:x", label.position.x + randf_range(-5, 5), ANIMATION_DURATION)\
		.set_trans(Tween.TRANS_EXPO).set_ease(Tween.EASE_OUT)
	tween.parallel()
	tween.tween_property(label, "position:y", label.position.y - FLOAT_HEIGHT * 1.5, ANIMATION_DURATION)\
		.set_trans(Tween.TRANS_EXPO).set_ease(Tween.EASE_OUT)
	tween.parallel()
	tween.tween_property(label, "scale", Vector2(0.1, 0.1), ANIMATION_DURATION)\
		.set_ease(Tween.EASE_IN)
	
	await tween.finished
	label.queue_free()

## 显示经验值文字（蓝色）
## @param exp: 经验值
## @param position: 世界坐标位置
func show_exp_text(exp: String, position: Vector2) -> void:
	show_damage_text("+" + exp + " EXP", position, Color.CYAN)

## 显示金币文字（金色）
## @param gold: 金币数量
## @param position: 世界坐标位置
func show_gold_text(gold: String, position: Vector2) -> void:
	show_damage_text("+" + gold + " G", position, Color.GOLD)

## 显示状态文字（白色）
## @param text: 状态文本
## @param position: 世界坐标位置
func show_status_text(text: String, position: Vector2) -> void:
	show_damage_text(text, position, Color.WHITE)

## 显示连击数（橙色）
## @param combo: 连击数
## @param position: 世界坐标位置
func show_combo_text(combo: int, position: Vector2) -> void:
	show_damage_text(str(combo) + " Combo!", position, Color.ORANGE)

## 显示等级提升（紫色）
## @param level: 新等级
## @param position: 世界坐标位置
func show_level_up_text(level: int, position: Vector2) -> void:
	show_damage_text("LEVEL UP! " + str(level), position, Color.PURPLE)
	
	
	
	
	
	
	
