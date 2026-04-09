class_name  SlashManifest
extends SkillManifest


@onready var animated_sprite:AnimatedSprite2D=$AnimatedSprite2D

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	if animated_sprite.frame_progress>=1.0:
		queue_free()
	pass

func activate(context:SkillContext):
	_activate(context)
	pass
	
# SlashManifest.gd

func _activate(context: SkillContext):
	var caster = context.caster as Entity
	caster.set_process(false)
	caster.get_node("AnimatedSprite2D").hide()
	animated_sprite.flip_h = caster.animated_sprite.flip_h
	
	# 如果施法者是 Player，根据 skin_color 替换攻击贴图
	var player_caster = caster as Player
	if player_caster != null:
		_apply_skin_to_manifest(player_caster)
	
	animated_sprite.play("slash")
	
	# 绑定额外参数
	var callable = Callable(self, "_on_animated_sprite2D_animation_finished").bind(caster)
	animated_sprite.animation_finished.connect(callable)

## 根据caster的skin_color替换攻击动画贴图
func _apply_skin_to_manifest(caster: Player) -> void:
	if caster.skin_color == "Blue":
		return  # 蓝色是默认贴图，无需替换
	
	# 让这个manifest实例拥有独立的SpriteFrames
	animated_sprite.sprite_frames = animated_sprite.sprite_frames.duplicate()
	var frames: SpriteFrames = animated_sprite.sprite_frames
	
	for anim_name in frames.get_animation_names():
		var frame_count = frames.get_frame_count(anim_name)
		for i in frame_count:
			var tex = frames.get_frame_texture(anim_name, i)
			if tex is AtlasTexture and tex.atlas != null:
				var atlas_path: String = tex.atlas.resource_path
				if "Blue Units" in atlas_path:
					var new_path = atlas_path.replace("Blue Units", caster.skin_color + " Units")
					var new_texture = load(new_path)
					if new_texture != null:
						var new_atlas_tex = tex.duplicate()
						new_atlas_tex.atlas = new_texture
						frames.set_frame(anim_name, i, new_atlas_tex)

func _on_animated_sprite2D_animation_finished(caster:Entity):
	caster.set_process(true)
	caster.get_node("AnimatedSprite2D").show()
