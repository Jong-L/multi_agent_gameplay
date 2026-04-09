class_name SkillAnimationRunner
extends SkillComponent

## 技能动画播放组件
## 激活时播放施法者的 "slash" 动画（高优先级）
##
## 高优先级特性：
##   - 可打断移动、待机等低优先级动画
##   - 不会被低优先级动画打断
##   - 适用于攻击、技能释放等关键动画

func _activate(context: SkillContext) -> void:
	context.caster.play_animation(AnimationWrapper.new("slash", true))
