class_name SkillController
extends Node

## 技能控制器
## 挂载在 Entity 下，管理该实体的所有技能
##
## 职责：
##   - 收集 Skill 子节点
##   - 管理技能冷却（每帧递减）
##   - 提供触发接口（检查冷却 → 激活）
##
## 使用方式：
##   - Player：通过 Action.ATTACK 触发 idx=0 的技能
##   - Enemy：AI 状态机中调用 trigger_skill_by_idx()
##   - UI：SpellButton 点击触发特定技能

## 技能成功激活信号（冷却检查通过后发射）
## @param entity: 激活技能的实体
## @param skill: 被激活的技能
signal skill_activated(entity: Entity, skill: Skill)

var skills: Array[Skill] = []           ## 技能数组（自动收集）
var cooldowns: Dictionary = {}          ## 冷却字典：Skill → 剩余时间
var entity: Entity                      ## 所属实体

func _ready() -> void:
	entity = get_parent()
	for child in get_children():
		if child is Skill:
			skills.push_back(child)

func _process(delta: float) -> void:
	## 更新所有技能冷却
	for skill in cooldowns.keys():
		if cooldowns[skill] > 0.0:
			cooldowns[skill] = max(0.0, cooldowns[skill] - delta)
			skill.current_cooldown = cooldowns[skill]

## 根据索引触发技能
## @param idx: 技能索引（从 0 开始）
func trigger_skill_by_idx(idx: int) -> void:
	if skills.size() == 0:
		return
	var skill = skills.get(idx)
	trigger_skill(skill)

## 触发指定技能
## 流程：检查冷却 → 激活 → 设置冷却
## @param skill: 目标技能
func trigger_skill(skill: Skill) -> void:
	if skill == null:
		return
	
	if cooldowns.get(skill, 0.0) > 0.0:
		return  ## 冷却中
	
	skill.activate(entity)
	cooldowns[skill] = skill.cooldown  ## 进入冷却
	skill_activated.emit(entity, skill)  ## 通知技能成功激活

## 重置所有技能冷却（用于游戏重置）
func reset_all_cooldowns() -> void:
	for skill in cooldowns.keys():
		cooldowns[skill] = 0.0
		skill.current_cooldown = 0.0

## 获取指定索引的技能
## @param idx: 技能索引
## @return: Skill 或 null
func get_skill(idx: int) -> Skill:
	if idx >= 0 and idx < skills.size():
		return skills[idx]
	return null

## 检查技能是否可用（未在冷却中）
## 只查询不触发，供状态机等逻辑做决策前检查
## @param idx: 技能索引
## @return: true 如果技能存在且冷却为 0
func can_use_skill(idx: int) -> bool:
	var skill = get_skill(idx)
	if skill == null:
		return false
	return cooldowns.get(skill, 0.0) <= 0.0

## 获取技能数量
func get_skill_count() -> int:
	return skills.size()  ## 返回技能数组的长度，用于遍历或检查技能数量
