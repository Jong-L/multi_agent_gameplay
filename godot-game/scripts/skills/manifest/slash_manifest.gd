class_name SlashManifest
extends SkillManifest

## 斩击 Manifest（近战攻击特效）
## 表现：暂停施法者，播放全屏斩击动画，结束后恢复
##
## 视觉效果：
##   - 施法者隐身，由斩击动画代替
##   - 支持皮肤颜色切换（Blue/Red/Yellow/Purple/Black）
##   - 动画结束后自动销毁

@onready var animated_sprite: AnimatedSprite2D = $AnimatedSprite2D

func _process(delta: float) -> void:
	if animated_sprite.frame_progress >= 1.0:
		queue_free()

func activate(context: SkillContext) -> void:
	_activate(context)

func _activate(context: SkillContext) -> void:
	var caster = context.caster as Entity
	
	## 暂停施法者，显示斩击动画
	caster.set_process(false)
	caster.get_node("AnimatedSprite2D").hide()
	animated_sprite.flip_h = caster.animated_sprite.flip_h
	
	## 皮肤颜色适配
	var player_caster = caster as Player
	if player_caster != null:
		_apply_skin_to_manifest(player_caster)
	
	animated_sprite.play("slash")
	
	## 绑定动画完成回调
	var callable = Callable(self, "_on_animated_sprite2D_animation_finished").bind(caster)
	animated_sprite.animation_finished.connect(callable)

## 皮肤颜色替换
## 原理：遍历所有帧，替换 "Blue Units" 为对应颜色目录
func _apply_skin_to_manifest(caster: Player) -> void:
	if caster.skin_color == "Blue":
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
					var new_path = atlas_path.replace("Blue Units", caster.skin_color + " Units")
					var new_texture = load(new_path)
					if new_texture != null:
						var new_atlas_tex = tex.duplicate()
						new_atlas_tex.atlas = new_texture
						frames.set_frame(anim_name, i, new_atlas_tex)

## 动画完成回调：恢复施法者
func _on_animated_sprite2D_animation_finished(caster: Entity) -> void:
	caster.set_process(true)
	caster.get_node("AnimatedSprite2D").show()
