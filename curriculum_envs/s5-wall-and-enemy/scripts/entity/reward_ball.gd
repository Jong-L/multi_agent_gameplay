class_name RewardBall
extends Area2D

## 奖励球实体
## 被玩家拾取后通过 EventBus 发出信号，由 RewardBallManager 处理奖励和重生逻辑
## 碰撞层：层4（奖励球专属层）
## 碰撞掩码：层1（检测 Player 的 CharacterBody2D）

enum BallType { TYPE_A, TYPE_B }

@export var ball_type: BallType = BallType.TYPE_A
@export var reward_value: float = 1.0

var is_active: bool = true  ## 当前是否可被拾取

@onready var _collision_shape: CollisionShape2D = $CollisionShape2D
@onready var _outer_glow: Sprite2D = $Visual/OuterGlow
@onready var _inner_core: Sprite2D = $Visual/InnerCore

## 球体纹理半径（像素）
const TEXTURE_SIZE := 35
## A类球颜色（青蓝）
const COLOR_A := Color(0.2, 0.85, 1.0, 1.0)
const GLOW_A := Color(0.1, 0.6, 1.0, 1)
## B类球颜色（金黄）
const COLOR_B := Color(0.153, 0.851, 0.149, 1.0)
const GLOW_B := Color(0.0, 0.698, 0.102, 1.0)


func _ready() -> void:
	# 碰撞层设置：奖励球在层4，检测层1（Player）
	collision_layer = 8  # 二进制 1000 = 层4
	collision_mask = 1   # 二进制 0001 = 层1（Player CharacterBody2D）
	
	body_entered.connect(_on_body_entered)
	
	# 生成程序化纹理
	_generate_textures()


func _on_body_entered(body: Node2D) -> void:
	if not is_active:
		return
	if not body is Player:
		return
	if body.is_dead:
		return
	
	# 拾取：发信号，由 RewardBallManager 处理奖励
	is_active = false
	EventBus.reward_ball_collected.emit(body.player_id, ball_type, self)
	
	# 视觉和碰撞隐藏
	_set_visual_active(false)
	
	# A类球永久消失，B类球由 Manager 管理重生
	if ball_type == BallType.TYPE_A:
		_collision_shape.set_deferred("disabled", true)


## 激活球体（由 RewardBallManager 在重生时调用）
func activate() -> void:
	is_active = true
	_set_visual_active(true)


## 游戏重置时调用：恢复初始状态
func reset_ball() -> void:
	is_active = true
	_set_visual_active(true)


func _set_visual_active(active: bool) -> void:
	if _outer_glow:
		_outer_glow.visible = active
	if _inner_core:
		_inner_core.visible = active
	if _collision_shape:
		_collision_shape.set_deferred("disabled", not active)


## 程序化生成圆形渐变纹理
func _generate_textures() -> void:
	var core_color := COLOR_A if ball_type == BallType.TYPE_A else COLOR_B
	var glow_color := GLOW_A if ball_type == BallType.TYPE_A else GLOW_B
	
	# 外层光晕
	_outer_glow.texture = _create_circle_texture(TEXTURE_SIZE * 2, glow_color)
	# 内核
	_inner_core.texture = _create_circle_texture(TEXTURE_SIZE, core_color)


## 创建一个径向渐变圆形纹理
static func _create_circle_texture(size: int, color: Color) -> ImageTexture:
	var img := Image.create(size, size, false, Image.FORMAT_RGBA8)
	var center := Vector2(size / 2.0, size / 2.0)
	var radius := size / 2.0
	
	for x in range(size):
		for y in range(size):
			var dist := Vector2(x, y).distance_to(center) / radius
			if dist <= 1.0:
				var alpha := 1.0 - smoothstep(0.0, 1.0, dist)
				img.set_pixel(x, y, Color(color.r, color.g, color.b, alpha * color.a))
			else:
				img.set_pixel(x, y, Color.TRANSPARENT)
	
	var tex := ImageTexture.create_from_image(img)
	return tex
