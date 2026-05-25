class_name AnimationWrapper
extends RefCounted

## 动画包装器
## 用于传递动画名称和优先级信息，实现动画打断机制
##
## 使用场景：
##   - 普通动画（移动、待机）：低优先级，可被攻击/死亡打断
##   - 关键动画（攻击、死亡）：高优先级，不可被打断
##
## 优先级规则：
##   - 高优先级可打断低优先级
##   - 同优先级遵循"先来先服务"
##   - 死亡动画通常设为最高优先级

var name: String = ""                    ## 动画名称（AnimatedSprite2D 中的动画名）
var is_high_priority: bool = false       ## 是否高优先级

func _init(_name: String, _is_high_priority: bool = false) -> void:
	self.name = _name
	self.is_high_priority = _is_high_priority

## 创建低优先级动画
static func create_low_priority(anim_name: String) -> AnimationWrapper:
	return AnimationWrapper.new(anim_name, false)

## 创建高优先级动画
static func create_high_priority(anim_name: String) -> AnimationWrapper:
	return AnimationWrapper.new(anim_name, true)

## 检查是否可以打断目标动画
## @param other: 目标动画包装器
## @return: true 表示可以打断
func can_interrupt(other: AnimationWrapper) -> bool:
	if other == null:
		return true
	## 高优先级可以打断低优先级
	if self.is_high_priority and not other.is_high_priority:
		return true
	## 同优先级不能打断
	return false

## 获取动画描述
func get_description() -> String:
	var priority = "High" if is_high_priority else "Low"
	return "%s (%s)" % [name, priority]

## 字符串表示
func _to_string() -> String:
	return "AnimationWrapper[name=%s, priority=%s]" % [name, "high" if is_high_priority else "low"]  ## 返回动画包装器的字符串表示，包含动画名称和优先级信息，便于调试和日志输出时查看详细信息

## 检查是否是移动动画
func is_movement_animation() -> bool:
	return name in ["run", "walk", "move"]  ## 检查当前动画名称是否属于移动类动画，用于判断实体是否处于移动状态

## 检查是否是攻击动画
func is_attack_animation() -> bool:
	return name in ["slash", "attack", "shoot", "cast"]  ## 检查当前动画名称是否属于攻击类动画，用于战斗状态判断和动画优先级管理

## 检查是否是死亡动画
func is_death_animation() -> bool:
	return name == "die" or name == "death"  ## 检查当前动画名称是否属于死亡类动画，死亡动画通常具有最高优先级且不可被打断

## 检查是否是待机动画
func is_idle_animation() -> bool:
	return name == "idle" or name == "stand"  ## 检查当前动画名称是否属于待机类动画，待机动画通常具有低优先级，容易被其他动画打断

## 获取建议的优先级
## 根据动画名称自动判断优先级
static func get_suggested_priority(anim_name: String) -> bool:
	if anim_name in ["die", "death"]:
		return true  ## 死亡动画高优先级
	if anim_name in ["slash", "attack", "shoot", "cast", "skill"]:
		return true  ## 攻击/技能动画高优先级
	return false  ## 其他动画低优先级  ## 根据动画名称返回建议的优先级设置，死亡和攻击类动画建议设为高优先级，其他动画建议设为低优先级

## 创建带有建议优先级的动画
## @param anim_name: 动画名称
## @return: 自动设置优先级的动画包装器
static func create_with_suggested_priority(anim_name: String) -> AnimationWrapper:
	var priority = get_suggested_priority(anim_name)
	return AnimationWrapper.new(anim_name, priority)  ## 根据动画名称自动判断并设置优先级，创建对应的动画包装器实例

## 复制当前动画包装器
## @return: 新的动画包装器实例
func duplicate() -> AnimationWrapper:
	return AnimationWrapper.new(self.name, self.is_high_priority)  ## 创建当前动画包装器的副本，用于需要保留原始设置的场景，如动画队列管理

## 比较两个动画包装器是否相等
## @param other: 另一个动画包装器
## @return: true 表示相等
func equals(other: AnimationWrapper) -> bool:
	if other == null:
		return false
	return self.name == other.name and self.is_high_priority == other.is_high_priority  ## 比较两个动画包装器的名称和优先级是否完全相同，用于动画去重或状态比较

## 获取动画持续时间估算（帧数 / 帧率）
## @param fps: 动画帧率（默认 8fps）
## @return: 估算持续时间（秒）
func get_estimated_duration(fps: float = 8.0) -> float:
	## 这里可以根据动画名称返回预定义时长
	## 实际项目中可以从 SpriteFrames 获取
	var frame_counts = {
		"idle": 4,
		"run": 6,
		"slash": 6,
		"die": 8
	}
	var frames = frame_counts.get(name, 4)
	return frames / fps  ## 根据动画名称估算动画持续时间，基于预定义的帧数和指定的帧率计算，用于动画调度和状态机计时

## 检查动画是否需要循环播放
## @return: true 表示应该循环
func should_loop() -> bool:
	return name in ["idle", "run", "walk"]  ## 检查当前动画是否应该循环播放，移动和待机类动画通常需要循环，而攻击和死亡动画通常只播放一次

## 获取动画类别
## @return: 动画类别字符串
func get_category() -> String:
	if is_death_animation():
		return "death"
	if is_attack_animation():
		return "attack"
	if is_movement_animation():
		return "movement"
	if is_idle_animation():
		return "idle"
	return "other"  ## 返回动画所属的类别，包括死亡、攻击、移动、待机等，用于动画分类管理和状态机逻辑

## 序列化为字典
## @return: 包含动画信息的字典
func to_dictionary() -> Dictionary:
	return {
		"name": name,
		"is_high_priority": is_high_priority,
		"category": get_category(),
		"should_loop": should_loop()
	}  ## 将动画包装器的信息序列化为字典格式，便于网络传输、存档保存或调试信息输出

## 从字典反序列化
## @param data: 包含动画信息的字典
## @return: 动画包装器实例
static func from_dictionary(data: Dictionary) -> AnimationWrapper:
	var anim_name = data.get("name", "")
	var priority = data.get("is_high_priority", false)
	return AnimationWrapper.new(anim_name, priority)  ## 从字典数据反序列化创建动画包装器，用于从存档或网络数据恢复动画状态

## 获取动画优先级数值
## @return: 优先级数值（越高越优先）
func get_priority_value() -> int:
	if is_high_priority:
		return 10
	return 5  ## 返回动画的优先级数值表示，高优先级返回 10，低优先级返回 5，用于数值比较和排序

## 比较优先级
## @param other: 另一个动画包装器
## @return: 正数表示当前优先级更高，负数表示更低，0 表示相等
func compare_priority(other: AnimationWrapper) -> int:
	return self.get_priority_value() - other.get_priority_value()  ## 比较两个动画包装器的优先级，返回差值用于排序操作，正值表示当前动画优先级更高

## 检查是否是过渡动画
## @return: true 表示是过渡动画
func is_transition_animation() -> bool:
	return name.begins_with("transition_") or name.begins_with("to_")  ## 检查动画名称是否以过渡相关前缀开头，用于识别状态之间的过渡动画

## 获取动画资源路径
## @return: 可能的资源路径
func get_resource_path() -> String:
	return "res://assets/animations/%s.tres" % name  ## 根据动画名称生成可能的资源路径，用于动态加载动画资源

## 验证动画名称有效性
## @return: true 表示名称有效
func is_valid() -> bool:
	return not name.is_empty()  ## 检查动画名称是否非空，用于验证动画包装器是否已被正确初始化
