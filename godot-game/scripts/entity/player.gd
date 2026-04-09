class_name  Player
extends Entity

@export var walk_speed=65
@export var run_speed=100
@export var player_spell_bar:SpellBar=null
@export var skin_color:String="Blue"  ## 皮肤颜色: Blue/Red/Yellow/Purple/Black

@onready var skill_controller:Node=$SkillController

var is_moving=false
var horizontal
var spawn_position=Vector2.ZERO

signal player_died(player:Player)

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	super._ready()
	spawn_position=position
	add_to_group("player")
	
	_apply_skin_color()
	
	for skill_idx in range(skill_controller.skills.size()):
		var skill=skill_controller.skills[skill_idx]
		player_spell_bar.register_skill(skill,skill_idx)
		
	EventBus.player_cast_skill.connect(_handle_skill)

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
	
# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	if is_dead:
		return
	
	_handle_movement(delta)
	_handle_animation()

func _handle_movement(delta):
	is_moving=false
	horizontal=Input.get_axis("move_left","move_right")
	var vertical=Input.get_axis("move_up","move_down")
	var movement=Vector2(horizontal,vertical)
	
	# 衰减外部推力
	if external_velocity!=Vector2.ZERO:
		external_velocity=external_velocity.move_toward(Vector2.ZERO,external_velocity.length()*external_velocity_decay*delta)
		if external_velocity.length()<1.0:
			external_velocity=Vector2.ZERO
	
	#移动时
	if movement.normalized().length()>0:
		#处理翻转
		if horizontal>0:
			animated_sprite.flip_h=false
		elif horizontal<0:
			animated_sprite.flip_h=true
		#动作切换
		is_moving=true
		velocity=movement.normalized()*run_speed+external_velocity
	else:
		velocity=external_velocity
	
	move_and_slide()

func _handle_animation():
	if is_moving:
		play_animation(AnimationWrapper.new("run",false))
	else:
		play_animation(AnimationWrapper.new("idle",false))

func _handle_skill(skill:Skill):
	skill_controller.trigger_skill(skill)

func _on_animated_sprite_2d_animation_finished() -> void:
	if current_animation_wrapper.name=="die":
		player_died.emit(self)
