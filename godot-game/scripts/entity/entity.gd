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
@export var atk:float=10.0  

var current_animation_wrapper: AnimationWrapper  
var current_health: float                       
var is_dead: bool = false                        

# 最后一次伤害来源（用于追踪击杀者）
var last_damage_source: Entity = null

func _ready() -> void:
	# 复制材质实例，避免多个实体共享同一材质导致状态互相影响
	animated_sprite.material = animated_sprite.material.duplicate()
	current_health = max_health

func bear_damage(source: Entity,skill:Skill) -> void:#承受伤害
	if current_health == 0:
		return
	
	# 记录伤害来源（用于击杀判定）
	if source != null:
		last_damage_source = source
	var damage=source.atk*skill.dmg_multiplier
	
	current_health = max(0, current_health - damage)
	_show_damage_taken_effect()
	_show_damage_popup(damage)
	
	# 发射全局受伤信号（供 RewardManager 监听）
	EventBus.entity_damaged.emit(self, source)
	
	if current_health == 0:
		is_dead = true
		_on_death()  # 死亡时立即触发（子类覆写此方法发射特定信号）
		play_animation(AnimationWrapper.new("die", true))  # 死亡动画,高优先级

## 死亡回调（虚方法，子类覆写以发射特定信号）
## 在 is_dead = true 之后、死亡动画播放之前调用
## 用于即时发放奖励，避免动画延迟导致信用分配问题
func _on_death() -> void:
	pass

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
	FloatText.show_damage_text(str(int(damage)), self.global_position, damage_text_color)
