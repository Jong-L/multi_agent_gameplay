class_name Player
extends Entity

# 玩家类
enum Action {
	MOVE_UP,#0
	MOVE_DOWN,
	MOVE_LEFT,   
	MOVE_RIGHT,  
	IDLE,#4
	ATTACK,#5:SkillController 的第 0 个技能
}

@export var run_speed: float = 100#当前实际使用
@export var player_spell_bar: SpellBar = null  #PlayScene 动态绑定技能栏
@export var skin_color: String
@export var player_id: int = 0#玩家唯一标识（0-3）

@onready var skill_controller: Node = $SkillController
@onready var ai_controller:AIController2D=$AIController2D
@onready var sync_node:Sync=$"../Sync"
@onready var reward_label:Label=$RewardLabel

var is_moving: bool = false                #期望速度大于零就为true，而非实际速度
var movement:Vector2=Vector2.ZERO     #键盘输入或智能体动作的期望移动方向                  
var spawn_position: Vector2
var pending_action: Action = Action.IDLE # 当前待执行动作
var _anim_idle: AnimationWrapper
var _anim_run: AnimationWrapper
var _last_displayed_reward: float = 0.0  # 缓存上次显示的 reward 值

signal player_died(player: Player)# 由PlayScene监听

func _ready() -> void:
	super._ready()
	# 缓存常用 AnimationWrapper，避免每帧创建
	_anim_idle = AnimationWrapper.new("idle", false)
	_anim_run = AnimationWrapper.new("run", false)
	spawn_position = position
	add_to_group("player")
	add_to_group("player_%d" % player_id)  # 带 ID 的分组，方便快速查找
	
	_apply_skin_color()
	
	ai_controller.init(self)#初始化绑定该玩家到控制器
	EventBus.player_cast_skill.connect(_handle_skill)  # 点击按钮也可以释放技能

func _physics_process(_delta: float) -> void:
	if is_dead:
		return
	
	_handle_movement()

func _process(_delta: float) -> void:
	if ai_controller.needs_reset:
		ai_controller.reset()
		return
	if is_dead:
		return
	
	_handle_animation()
	_execute_action()
	
	
	#奖励标签（仅在 reward 值变化时更新）
	if CameraManager.current_camera_id!=-1:
		var current_reward: float = ai_controller.reward
		if abs(current_reward - _last_displayed_reward) > 0.000001:
			reward_label.text = String.num(current_reward, 6)
			_last_displayed_reward = current_reward

func _apply_skin_color() -> void:#根据skin_color设置使用的材质
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

func set_action(action: int) -> void:#设置待执行动作
	if action >= 0 and action <= 5:
		pending_action = action as Action

#根据 pending_action执行动作
func _handle_movement() -> void:
	is_moving = false#不移动时为false
	
	#human模式下操控当前看到的角色
	if sync_node.control_mode==sync_node.ControlModes.HUMAN\
	and player_id==CameraManager.current_camera_id:
		movement=Input.get_vector("move_left", "move_right", "move_up", "move_down")
		
	match pending_action:
		Action.MOVE_UP:
			movement = Vector2.UP
		Action.MOVE_DOWN:
			movement = Vector2.DOWN
		Action.MOVE_LEFT:
			movement = Vector2.LEFT
		Action.MOVE_RIGHT:
			movement = Vector2.RIGHT
	
	if movement.length() > 0:
		velocity = movement.normalized() * run_speed
	else:
		velocity = Vector2.ZERO
	
	if velocity.length()>0:
		is_moving=true
	move_and_slide()

func _execute_action() -> void:# 执行非移动动作（攻击/待机）
	if pending_action == Action.ATTACK:
		skill_controller.trigger_skill_by_idx(0)  # 触发第 0 个技能

func _handle_animation() -> void:# 动画状态更新
	#if player_id==0:
		#print(self.get_real_velocity())
	if movement.length() > 0:
		if movement.x > 0:
			animated_sprite.flip_h = false
		elif movement.x < 0:
			animated_sprite.flip_h = true
	if is_moving:
		play_animation(_anim_run)
	else:
		play_animation(_anim_idle)

func _handle_skill(skill: Skill) -> void:#点击技能按钮触发
	skill_controller.trigger_skill(skill)

## 死亡回调：在 bear_damage 检测到 current_health==0 时立即调用
## 即时发射全局死亡信号，确保奖励无延迟
func _on_death() -> void:
	EventBus.player_died.emit(self)

func _get_die_anim() -> AnimationWrapper:
	return AnimationWrapper.new("die", true)

func _on_animated_sprite_2d_animation_finished() -> void:
	if current_animation_wrapper != null and current_animation_wrapper.name == "die":
		player_died.emit(self)

func _on_reward_perform_button_pressed() -> void:
	reward_label.visible=!reward_label.visible
