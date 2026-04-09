class_name SkillSpawnManifest
extends SkillComponent

@export var manifest_scene:PackedScene
@export var set_as_child=false


# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	pass

func _activate(context:SkillContext):
	super._activate(context)
	if manifest_scene==null:return
	var skill_manifest:Node=manifest_scene.instantiate() 
	var caster=context.caster
	if set_as_child==true:
		caster.add_child(skill_manifest)
	else:
		var root=get_tree().get_root()
		root.add_child(skill_manifest)
	
	skill_manifest.activate(context)
	
	
	
	
	
	
	
	
	
	
	
	
