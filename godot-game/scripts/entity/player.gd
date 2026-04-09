class_name  Player
extends Entity

## 动作枚举：与 Python 端约定的 6 个离散动作
enum Action {
	MOVE_UP,     ## 0: 上移
	MOVE_DOWN,   ## 1: 下移
	MOVE_LEFT,   ## 2: 左移
	MOVE_RIGHT,  ## 3: 右移
	ATTACK,      ## 4: 攻击
	IDLE,        ## 5: 待机
}

@export var walk_speed=65
@export var run_speed=100
@export var player_spell_bar:SpellBar=null
@export var skin_color:String="Blue"  ## 皮肤颜色: Blue/Red/Yellow/Purple/Black
@export var player_id:int = 0  ## 唯一标识，用于区分不同智能体

@onready var skill_controller:Node=$SkillController

var is_moving=false
var horizontal:float = 0.0
var spawn_position=Vector2.ZERO

## 当前待执行的动作，由外部(NetworkManager)通过 set_action() 设置
var pending_action:Action = Action.IDLE

signal player_died(player:Player)

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	super._ready()
	spawn_position=position
	add_to_group("player")
	# 同时加入带 id 的分组，方便按 id 查找
	add_to_group("player_%d" % player_id)
	
	_apply_skin_color()
	
	# SpellBar 绑定和技能注册由 PlayScene._setup_player_uis() 统一管理
	# 不再在 Player._ready() 中注册，避免多玩家覆盖共享 SpellBar 的问题
		
	EventBus.player_cast_skill.connect(_handle_skill)#直接点击技能按钮时释放技能

## 根据skin_color替换SpriteFrames中所有AtlasTexture的atlas贴图
func _apply_skin_color() -> void:
	if skin_color == "Blue":
		return  # 蓝色是默认贴图，无需替换
	
	# 让每个实例拥有独立的SpriteFrames，避免影响其他实例
	animated_sprite.sprite_frames = animated_sprite.sprite_frames.duplicate()
	
	var frames: SpriteFrames = animated_sprite.sprite_frames
	
	for anim_name in frames.get_animation_names():
		var frame_count = frames.get_frame_count(anim_name)
		for i in frame_count:
			var tex = frames.get_frame_texture(anim_name, i)
			if tex is AtlasTexture and tex.atlas != null:
				var atlas_path: String = tex.atlas.resource_path
				# 只替换 Blue Units 目录下的贴图
				if "Blue Units" in atlas_path:
					var new_path = atlas_path.replace("Blue Units", skin_color + " Units")
					var new_texture = load(new_path)
					if new_texture != null:
						# 需要duplicate AtlasTexture本身，避免修改共享资源
						var new_atlas_tex = tex.duplicate()
						new_atlas_tex.atlas = new_texture
						frames.set_frame(anim_name, i, new_atlas_tex)

## 由外部调用，设置本帧要执行的动作
func set_action(action: int) -> void:
	if action >= 0 and action <= 5:
		pending_action = action as Action

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	if is_dead:
		return
	
	_handle_movement(delta)
	_handle_animation()
	_execute_action()

## 根据 pending_action 执行移动（替代原来的 Input 键盘输入）
func _handle_movement(delta):
	is_moving = false
	var movement = Vector2.ZERO
	
	# 根据 pending_action 计算移动方向
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
	
	# 衰减外部推力
	if external_velocity!=Vector2.ZERO:
		external_velocity=external_velocity.move_toward(Vector2.ZERO,external_velocity.length()*external_velocity_decay*delta)
		if external_velocity.length()<1.0:
			external_velocity=Vector2.ZERO
	
	# 移动时
	if movement.length() > 0:
		# 处理翻转（上/下移动时保持当前朝向，左/右已处理）
		if movement.x > 0:
			animated_sprite.flip_h = false
		elif movement.x < 0:
			animated_sprite.flip_h = true
		is_moving = true
		velocity = movement.normalized() * run_speed + external_velocity
	else:
		velocity = external_velocity
	
	move_and_slide()

## 执行非移动类动作（攻击/待机），每帧末尾重置 pending_action
func _execute_action() -> void:
	if pending_action == Action.ATTACK:
		skill_controller.trigger_skill_by_idx(0)
	# 动作执行完毕，重置为 IDLE
	pending_action = Action.IDLE

func _handle_animation():
	if is_moving:
		play_animation(AnimationWrapper.new("run",false))
	else:
		play_animation(AnimationWrapper.new("idle",false))

func _handle_skill(skill:Skill):
	skill_controller.trigger_skill(skill)

## 获取玩家状态数据，用于发送给 Python 端
func get_state() -> Dictionary:
	return {
		"id": player_id,
		"x": position.x,
		"y": position.y,
		"hp": current_health,
		"max_hp": max_health,
		"alive": not is_dead,
		"flip_h": animated_sprite.flip_h,
	}

func _on_animated_sprite_2d_animation_finished() -> void:
	if current_animation_wrapper.name=="die":
		player_died.emit(self)
