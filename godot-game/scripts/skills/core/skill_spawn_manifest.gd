class_name SkillSpawnManifest
extends SkillComponent

" 技能 Manifest 生成组件
 激活时实例化并启动一个 SkillManifest 节点
 Manifest 负责具体的技能效果呈现（如攻击判定、特效、持续效果）

 与 SkillComponent 的区别：
   - SkillComponent：同步执行，立即完成
   - SkillManifest：独立节点，可包含动画、持续判定、延迟效果

 典型使用：SlashManifest（斩击特效）、AreaEffectManifest（范围持续效果）"

@export var manifest_scene: PackedScene    # Manifest 场景资源
@export var set_as_child: bool = false     #true=作为施法者子节点，false=添加到场景根节点

func _activate(context: SkillContext) -> void:
	super._activate(context)
	if manifest_scene == null:
		return
	
	var skill_manifest: Node = manifest_scene.instantiate()
	var caster = context.caster
	
	if set_as_child == true:
		caster.add_child(skill_manifest)
	else:
		var root = get_tree().get_root()
		root.add_child(skill_manifest)
	
	skill_manifest.activate(context)
