extends Node

## 全局事件总线
## 用于跨场景/跨节点的解耦事件通信
## 使用方式：
##   发送：EventBus.信号名.emit(参数)
##   接收：EventBus.信号名.connect(回调函数)

@warning_ignore("unused_signal")
## 玩家释放技能
## @param skill: 被触发的技能实例
## 连接：Player._handle_skill() (player.gd:39)
signal player_cast_skill(skill: Skill)

@warning_ignore("unused_signal")
## 游戏暂停状态切换
## @param paused: true=暂停, false=恢复
## 连接：PlayScene._handle_pause() (play_scene.gd:48)
signal game_paused(paused: bool)

@warning_ignore("unused_signal")
## 奖励球被拾取
## @param player_id: 拾取者的玩家 ID
## @param ball_type: RewardBall.BallType 枚举值
## @param ball: 被拾取的奖励球实例
## 连接：RewardManager._on_reward_ball_collected() (reward_manager.gd:118)
## 连接：RewardBallManager._on_reward_ball_collected() (reward_ball_manager.gd:43)
signal reward_ball_collected(player_id: int, ball_type: int, ball: RewardBall)

@warning_ignore("unused_signal")
## 玩家死亡
## @param player: 死亡的玩家实例
## 连接：RewardManager._on_player_died() 
## 连接：PlayScene._on_player_player_died()
signal player_died(player: Player)

@warning_ignore("unused_signal")
## 敌人死亡
## @param enemy: 死亡的敌人实例
## 连接：RewardManager._on_enemy_died() (reward_manager.gd:116)
signal enemy_died(enemy: Enemy)

@warning_ignore("unused_signal")
## 实体受到伤害
## @param entity: 受伤的实体
## @param source: 伤害来源
## 连接：RewardManager._on_entity_damaged() (reward_manager.gd:115)
signal entity_damaged(entity: Entity, source: Entity)

@warning_ignore("unused_signal")
## 相机切换
## @param camera_id: -1=主相机, 0-3=玩家相机
## 连接：PlayScene._on_camera_switched() (play_scene.gd:156)
signal camera_switched(camera_id: int)

@warning_ignore("unused_signal")
## 纯奖励值变更（不含塑形奖励）
## @param player_id: 玩家 ID
## @param total_pure_reward: 累计纯奖励值
## 连接：ScoreboardDrawer._on_pure_reward_changed() (scoreboard_drawer.gd)
signal pure_reward_changed(player_id: int, total_pure_reward: float)
