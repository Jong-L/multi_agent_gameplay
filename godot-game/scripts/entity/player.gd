class_name Player
extends Entity

## 玩家类
## 继承自 Entity，代表 4 名玩家之一
##
## 核心职责：
##   - 接收外部动作指令（来自强化学习端）
##   - 执行离散动作（移动/攻击/待机）
##   - 支持皮肤颜色切换（Blue/Red/Yellow/Purple/Black）
##   - 提供状态观测接口（位置、血量、朝向等）
##
## 架构位置：
##   - 父类：Entity（生命值、动画、受击反馈）
##   - 组件：SkillController（技能管理）
##   - UI 绑定：SpellBar（技能栏）、PlayerHealthBar（血条）

## 动作枚举（与 Python 端约定的离散动作空间）
## 强化学习环境中的动作空间大小为 6
enum Action {
	MOVE_UP,     ## 0: 向上移动
	MOVE_DOWN,   ## 1: 向下移动
	MOVE_LEFT,   ## 2: 向左移动
	MOVE_RIGHT,  ## 3: 向右移动
	ATTACK,      ## 4: 攻击（触发 SkillController 的第 0 个技能）
	IDLE,        ## 5: 待机（不移动）
}

@export var walk_speed: float = 65          ## 行走速度（基础移动速度）
@export var run_speed: float = 100          ## 奔跑速度（当前实际使用的速度）
@export var player_spell_bar: SpellBar = null  ## 专属技能栏（PlayScene 动态绑定）
@export var skin_color: String = "Blue"        ## 皮肤颜色（决定贴图目录）
@export var player_id: int = 0                 ## 玩家唯一标识（0-3，多智能体区分）

@onready var skill_controller: Node = $SkillController

var is_moving: bool = false                  
var horizontal: float = 0.0                  
var spawn_position: Vector2 = Vector2.ZERO
var pending_action: Action = Action.IDLE # 当前待执行动作

signal player_died(player: Player)# 死亡信号（PlayScene 监听处理）

func _ready() -> void:
	super._ready()
	spawn_position = position
	add_to_group("player")
	add_to_group("player_%d" % player_id)  # 带 ID 的分组，方便快速查找
	
	_apply_skin_color()
	
	ai_controller.init(self)#初始化绑定该玩家到控制器
	EventBus.player_cast_skill.connect(_handle_skill)  # 点击按钮也可以释放技能

func _apply_skin_color() -> void:
	if skin_color == "Blue":
		return
	
	animated_sprite.sprite_frames = animated_sprite.sprite_frames.duplicate()
	var frames: SpriteFrames = animated_sprite.sprite_frames
	
	for anim_name in frames.get_animation_names():
		var frame_count = frames.get_frame_count(anim_name)
		for i in frame_count:
			var tex = frames.get_frame_texture(anim_name, i)
			if tex is AtlasTexture and tex.atlas != null:
				var atlas_path: String = tex.atlas.resource_path
				if "Blue Units" in atlas_path:
					var new_path = atlas_path.replace("Blue Units", skin_color + " Units")
					var new_texture = load(new_path)
					if new_texture != null:
						var new_atlas_tex = tex.duplicate()
						new_atlas_tex.atlas = new_texture
						frames.set_frame(anim_name, i, new_atlas_tex)

## 设置待执行动作
## @param action: 动作枚举值（0-5，对应 Action 枚举）
func set_action(action: int) -> void:
	if action >= 0 and action <= 5:
		pending_action = action as Action

func _process(delta: float) -> void:
	if is_dead:
		return
	
	_handle_movement(delta)
	_handle_animation()
	_execute_action()

#根据 pending_action执行动作
func _handle_movement(delta: float) -> void:
	is_moving = false
	var movement = Vector2.ZERO
	
	match pending_action:
		Action.MOVE_UP:
			movement = Vector2.UP
		Action.MOVE_DOWN:
			movement = Vector2.DOWN
		Action.MOVE_LEFT:
			movement = Vector2.LEFT
			animated_sprite.flip_h = true
		Action.MOVE_RIGHT:
			movement = Vector2.RIGHT
			animated_sprite.flip_h = false
	
	horizontal = movement.x
	
	## 外部推力衰减（击退效果逐渐归零）
	if external_velocity != Vector2.ZERO:
		external_velocity = external_velocity.move_toward(
			Vector2.ZERO,
			external_velocity.length() * external_velocity_decay * delta
		)
		if external_velocity.length() < 1.0:
			external_velocity = Vector2.ZERO
	
	if movement.length() > 0:
		## 上下移动时保持当前水平朝向
		if movement.x > 0:
			animated_sprite.flip_h = false
		elif movement.x < 0:
			animated_sprite.flip_h = true
		
		is_moving = true
		velocity = movement.normalized() * run_speed + external_velocity
	else:
		velocity = external_velocity
	
	move_and_slide()

# 执行非移动动作（攻击/待机）.执行后自动重置为 IDLE
func _execute_action() -> void:
	if pending_action == Action.ATTACK:
		skill_controller.trigger_skill_by_idx(0)  # 触发第 0 个技能
	
	pending_action = Action.IDLE

# 动画状态更新
func _handle_animation() -> void:
	if is_moving:
		play_animation(AnimationWrapper.new("run", false))
	else:
		play_animation(AnimationWrapper.new("idle", false))

#点击技能按钮触发
func _handle_skill(skill: Skill) -> void:
	skill_controller.trigger_skill(skill)

# 获取观测
func get_obs() -> Dictionary:
	return {
		"id": player_id,
		"x": global_position.x,
		"y": global_position.y,
		"hp": current_health/max_health,
		"max_hp": max_health,
		"alive": not is_dead,
		"flip_h": animated_sprite.flip_h,
	}

#override
func bear_damage(damage: float) -> void:
	if current_health == 0:
		return
	
	current_health = max(0, current_health - damage)
	_show_damage_taken_effect()
	_show_damage_popup(damage)
	
	ai_controller.reward-=1
	
	if current_health == 0:
		is_dead = true
		play_animation(AnimationWrapper.new("die", true))  # 死亡动画,高优先级

func _on_animated_sprite_2d_animation_finished() -> void:
	if current_animation_wrapper != null and current_animation_wrapper.name == "die":
		player_died.emit(self)
