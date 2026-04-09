class_name Entity
extends CharacterBody2D 

@onready var animated_sprite:AnimatedSprite2D=$AnimatedSprite2D

@export var max_health:float=100
@export var damage_text_color=Color.AZURE

var current_animation_wrapper:AnimationWrapper
var current_health:float
var is_dead:bool=false

# 外部推力（用于 pushback 等效果）
var external_velocity:Vector2=Vector2.ZERO
var external_velocity_decay:float=10.0  # 衰减速率

func _ready() -> void:
	animated_sprite.material=animated_sprite.material.duplicate()
	current_health=max_health

func bear_damage(damage:float):
	if current_health==0:return
	current_health=max(0,current_health-damage)
	_show_damage_taken_effect()
	_show_damage_popup(damage)
	if current_health==0:
		is_dead=true
		play_animation(AnimationWrapper.new("die",true))

func play_animation(animation_wrapper:AnimationWrapper):
	if(
		current_animation_wrapper!=null and current_animation_wrapper.is_high_priority
		and not animation_wrapper.is_high_priority
	):return
	
	current_animation_wrapper=animation_wrapper
	animated_sprite.play(animation_wrapper.name)

func _show_damage_taken_effect():
	if animated_sprite.material !=null:
		for i in 2:
			animated_sprite.material.set_shader_parameter("is_hurt",true)
			await get_tree().create_timer(0.05).timeout
			animated_sprite.material.set_shader_parameter("is_hurt",false)
			await get_tree().create_timer(0.05).timeout

func _show_damage_popup(damage:float):
	var animation=animated_sprite.animation
	var frame_texture=animated_sprite.sprite_frames.get_frame_texture(animation,0)
	var height=frame_texture.get_height()
	FloatText.show_damage_text(str(int(damage)),self.global_position,damage_text_color)
