class_name VisionCircle
extends Node2D

## 视野范围可视化圆
## 以虚线圆的形式显示玩家的视野半径范围
## 作为 Player 的子节点，自动跟随玩家移动
## 注意：Player 的 scale 为 0.3，需通过 _unhandled_key_input 或补偿缩放

## 虚线圆颜色（各玩家可在 PlayScene 中单独设置）
@export var circle_color: Color = Color(1, 1, 1, 0.5)
## 虚线宽度
@export var dash_width: float = 2.0
## 虚线段长度
@export var dash_length: float = 10.0
## 虚线间隔长度
@export var gap_length: float = 8.0
## 视野半径（世界坐标系，与 VisionSensor.vision_radius 同步）
@export var vision_radius: float = 250.0

## 内部引用
var _player: CharacterBody2D = null
## 实际绘制半径（补偿父节点缩放）
var _draw_radius: float = 250.0

func _ready() -> void:
	z_index = 10  # 在玩家精灵上方绘制，便于观察

func _draw() -> void:
	if _player == null or not is_instance_valid(_player):
		return
	_draw_dashed_circle(Vector2.ZERO, _draw_radius, circle_color, dash_width, dash_length, gap_length)

## 绘制虚线圆
func _draw_dashed_circle(center: Vector2, radius: float, color: Color, width: float, dash_len: float, gap_len: float) -> void:
	var circumference: float = 2.0 * PI * radius
	var segment_len: float = dash_len + gap_len
	var segment_count: int = max(1, int(circumference / segment_len))
	var angle_per_segment: float = 2.0 * PI / segment_count
	var dash_angle: float = angle_per_segment * (dash_len / segment_len)

	for i in range(segment_count):
		var start_angle: float = i * angle_per_segment
		var end_angle: float = start_angle + dash_angle
		_draw_arc(center, radius, start_angle, end_angle, color, width)

## 绘制弧线（近似用多段直线）
func _draw_arc(center: Vector2, radius: float, start_angle: float, end_angle: float, color: Color, width: float) -> void:
	var steps: int = max(2, int((end_angle - start_angle) / 0.1))
	var points: PackedVector2Array = PackedVector2Array()
	for j in range(steps + 1):
		var angle: float = start_angle + (end_angle - start_angle) * float(j) / float(steps)
		points.append(center + Vector2(cos(angle), sin(angle)) * radius)
	if points.size() >= 2:
		draw_polyline(points, color, width, true)

## 更新视野半径（外部调用）
func update_radius(new_radius: float) -> void:
	vision_radius = new_radius
	# 补偿父节点缩放：Player scale=0.3，圆半径需要除以 0.3 才能在世界坐标中显示正确大小
	if _player != null and is_instance_valid(_player):
		var parent_scale := _player.scale.x if _player.scale.x != 0 else 1.0
		_draw_radius = new_radius / parent_scale
	else:
		_draw_radius = new_radius
	queue_redraw()

## 绑定玩家引用
func setup(player: CharacterBody2D, radius: float, color: Color) -> void:
	_player = player
	vision_radius = radius
	circle_color = color
	# 补偿缩放
	var parent_scale := player.scale.x if player.scale.x != 0 else 1.0
	_draw_radius = radius / parent_scale
	queue_redraw()
