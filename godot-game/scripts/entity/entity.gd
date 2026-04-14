class_name Entity
extends CharacterBody2D 

" 实体基类
 所有游戏角色（玩家、敌人）的公共父类
 提供：生命值管理、伤害处理、动画播放、受击反馈等通用功能
 架构位置：Entity → Player/Enemy
 关联系统：
   - SkillController: 技能释放时通过 SkillContext 传递施法者引用
   - SkillGetTarget/SkillTargetPlayer: 技能目标检测的目标类型
   - FloatText: 受伤时显示飘字
   - AnimationWrapper: 动画优先级管理
"

@onready var animated_sprite: AnimatedSprite2D = $AnimatedSprite2D

@export var max_health: float = 100 
@export var damage_text_color = Color.AZURE   

var current_animation_wrapper: AnimationWrapper  
var current_health: float                       
var is_dead: bool = false                        

# 外部推力系统,用于击退、击飞等
# 与 velocity 独立，每帧向零衰减，实现平滑的受击位移
var external_velocity: Vector2 = Vector2.ZERO
var external_velocity_decay: float = 10.0        ## 推力衰减速率
# 衰减公式：external_velocity.move_toward(ZERO, length * decay * delta)

func _ready() -> void:
	# 复制材质实例，避免多个实体共享同一材质导致状态互相影响
	animated_sprite.material = animated_sprite.material.duplicate()
	current_health = max_health

func bear_damage(damage: float) -> void:#承受伤害
	if current_health == 0:
		return
	
	current_health = max(0, current_health - damage)
	_show_damage_taken_effect()
	_show_damage_popup(damage)
	
	if current_health == 0:
		is_dead = true
		play_animation(AnimationWrapper.new("die", true))  # 死亡动画,高优先级

func play_animation(animation_wrapper: AnimationWrapper) -> void:# 用动画包装器播放动画
	if current_animation_wrapper != null \
		and current_animation_wrapper.is_high_priority \
		and not animation_wrapper.is_high_priority:
		return
	
	current_animation_wrapper = animation_wrapper
	animated_sprite.play(animation_wrapper.name)

func _show_damage_taken_effect() -> void:#受击特效
	if animated_sprite.material != null:
		for i in 2:
			animated_sprite.material.set_shader_parameter("is_hurt", true)
			await get_tree().create_timer(0.05).timeout
			animated_sprite.material.set_shader_parameter("is_hurt", false)
			await get_tree().create_timer(0.05).timeout

func _show_damage_popup(damage: float) -> void:#伤害跳字
	var animation = animated_sprite.animation
	var frame_texture = animated_sprite.sprite_frames.get_frame_texture(animation, 0)
	var height = frame_texture.get_height()
	FloatText.show_damage_text(str(int(damage)), self.global_position, damage_text_color)
