extends Node

## 全局事件总线（Autoload 单例）
## 用于跨场景/跨节点的解耦事件通信
##
## 使用方式：
##   发送：EventBus.信号名.emit(参数)
##   接收：EventBus.信号名.connect(回调函数)
##
## 设计原则：
##   - 避免直接节点引用，降低耦合
##   - 适合全局状态变化（暂停、重置、技能释放等）
##   - 不适合高频事件（如每帧更新）

## 玩家释放技能
## @param skill: 被触发的技能实例
signal player_cast_skill(skill: Skill)

## 游戏暂停状态切换
## @param paused: true=暂停, false=恢复
signal game_paused(paused: bool)

## 游戏重置请求

## 奖励球被拾取
## @param player_id: 拾取者的玩家 ID
## @param ball_type: RewardBall.BallType 枚举值
## @param reward_value: 奖励数值
## @param ball: 被拾取的奖励球实例
signal reward_ball_collected(player_id: int, ball_type: int, reward_value: float, ball: RewardBall)

## 玩家死亡
## @param player: 死亡的玩家实例
signal player_died(player: Player)

## 敌人死亡
## @param enemy: 死亡的敌人实例
signal enemy_died(enemy: Enemy)

## 游戏状态变化
## @param state: 游戏状态字典
signal game_state_changed(state: Dictionary)

## 相机切换
## @param camera_id: -1=主相机, 0-3=玩家相机
signal camera_switched(camera_id: int)

## 技能冷却完成
## @param skill: 冷却完成的技能
## @param entity: 所属实体
signal skill_cooldown_finished(skill: Skill, entity: Entity)

## 实体受到伤害
## @param entity: 受击实体
## @param damage: 伤害值
## @param source: 伤害来源
signal entity_damaged(entity: Entity, damage: float, source: Entity)

## 游戏开始
signal game_started

## 游戏结束
## @param result: 游戏结果字典
signal game_ended(result: Dictionary)

## 回合开始（用于回合制扩展）
## @param round_number: 回合数
signal round_started(round_number: int)

## 回合结束
## @param round_number: 回合数
signal round_ended(round_number: int)

## 实体生成
## @param entity: 新实体
signal entity_spawned(entity: Entity)

## 实体销毁
## @param entity: 被销毁的实体
signal entity_despawned(entity: Entity)

## 技能效果触发
## @param skill: 技能
## @param context: 技能上下文
signal skill_effect_triggered(skill: Skill, context: SkillContext)

## 网络消息接收（保留供未来扩展）
## @param message: 消息字典
signal network_message_received(message: Dictionary)

## 调试信息更新
## @param info: 调试信息字典
signal debug_info_updated(info: Dictionary)

## 性能统计更新
## @param stats: 性能统计字典
signal performance_stats_updated(stats: Dictionary)

## UI 更新请求
## @param ui_type: UI 类型
## @param data: 更新数据
signal ui_update_requested(ui_type: String, data: Dictionary)

## 存档加载完成
## @param save_data: 存档数据
signal save_loaded(save_data: Dictionary)

## 存档保存完成
## @param save_path: 存档路径
signal save_saved(save_path: String)

## 设置变更
## @param setting_key: 设置项
## @param value: 新值
signal setting_changed(setting_key: String, value: Variant)

## 语言切换
## @param language: 语言代码
signal language_changed(language: String)

## 音量变化
## @param bus_name: 音频总线名
## @param volume: 音量值（0-1）
signal volume_changed(bus_name: String, volume: float)

## 全屏模式切换
## @param fullscreen: true=全屏
signal fullscreen_toggled(fullscreen: bool)

## 输入设备变化
## @param device_type: 设备类型
signal input_device_changed(device_type: String)

## 成就解锁
## @param achievement_id: 成就 ID
signal achievement_unlocked(achievement_id: String)

## 统计更新
## @param stat_name: 统计项名
## @param value: 新值
signal statistic_updated(stat_name: String, value: int)

## 教程步骤触发
## @param step_id: 步骤 ID
signal tutorial_step_triggered(step_id: String)

## 教程完成
## @param tutorial_id: 教程 ID
signal tutorial_completed(tutorial_id: String)

## 错误发生
## @param error_code: 错误码
## @param error_message: 错误信息
signal error_occurred(error_code: int, error_message: String)

## 警告发生
## @param warning_message: 警告信息
signal warning_occurred(warning_message: String)

## 信息提示
## @param message: 提示信息
## @param type: 提示类型
signal notification_showed(message: String, type: int)

## 加载场景开始
## @param scene_path: 场景路径
signal scene_load_started(scene_path: String)

## 加载场景完成
## @param scene_path: 场景路径
signal scene_load_finished(scene_path: String)

## 加载进度更新
## @param progress: 进度（0-1）
signal load_progress_updated(progress: float)

## 实体选择变化
## @param selected_entity: 选中的实体
## @param previous_entity: 之前选中的实体
signal entity_selection_changed(selected_entity: Entity, previous_entity: Entity)

## 命令执行
## @param command: 命令名
## @param params: 参数
signal command_executed(command: String, params: Dictionary)

## 回放开始
signal replay_started

## 回放结束
signal replay_ended

## 回放暂停
## @param paused: true=暂停
signal replay_paused(paused: bool)

## 回放进度
## @param progress: 进度（0-1）
signal replay_progress_updated(progress: float)
