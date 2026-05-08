class_name SkillSpawnManifest
extends SkillComponent

@export var manifest_scene: PackedScene    # Manifest 场景资源

func _activate(context: SkillContext) -> void:
	#super._activate(context)
	if manifest_scene == null:
		return
	
	var skill_manifest: Node = manifest_scene.instantiate()
	var caster = context.caster
	
	caster.add_child(skill_manifest)
	
	skill_manifest.activate(context)
