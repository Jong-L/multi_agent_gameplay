# 项目记忆 - Roguelike Game (Godot 4.6)

## 项目概况
- 2D俯视角动作Roguelike游戏，像素风格
- 使用免费素材包（Cute_Fantasy, Tiny Swords, Tiny RPG Character）
- 核心架构：Entity基类 → Player/Enemy，组件化技能系统
- 技能系统：Skill → SkillComponent组合（GetTarget/Animation/DealDamage/Pushback/Manifest）

## 已识别的设计问题
- 敌人攻击"必中"问题：SkillTargetPlayer不检查距离，敌人无前摇后摇（2026-04-08诊断）
- 改进方案：引入攻击状态机(追击→前摇→攻击中→后摇) + SkillTargetPlayer距离校验 + 前摇视觉反馈
- 状态：**已实施**（2026-04-08）
- 角色穿墙问题：Entity用position+=绕过物理引擎，无TileSet碰撞层
- 改进方案：CharacterBody2D + move_and_slide() + TileSet physics_layer
- 状态：**已实施**（2026-04-08）

## 碰撞系统架构（2026-04-08实施）
- 物理层：layer1=player, layer2=enemy, layer3=wall
- Player: collision_layer=1, collision_mask=4（只检测墙壁）
- Enemy: collision_layer=2, collision_mask=4（只检测墙壁）
- Water TileSet: physics_layer=4 (wall)，所有Water tile有矩形碰撞多边形
- Entity新增external_velocity机制，替代pushback的tween position方式
- pathfinding查询设collide_with_bodies=false，只检测Area2D

## 多目标感知系统（2026-04-09实施）
- enemy.gd：`var player:Player` → `var target:Player`，语义从"唯一玩家"变为"当前锁定目标"
- enemy.gd：新增 `_get_alive_players()` → 返回所有活着的Player数组
- enemy.gd：新增 `_find_nearest_player()` → 按距离找最近活玩家（纯距离候选查询）
- enemy.gd：新增 `_find_nearest_visible_player()` → 视野内最近活玩家
- enemy.gd：新增 `_can_see_any_player()` → 视野内是否有任何活玩家
- enemy.gd：新增 `_update_target()` → CHASE阶段黏性切换逻辑，新候选距离<当前×0.7才换
- enemy.gd：CHASE阶段每帧调用 `_update_target()` 允许换目标（黏性条件满足时）
- enemy.gd：ATTACK_WINDUP/ATTACKING阶段锁定目标不切换
- enemy.gd：ATTACK_RECOVERY阶段重新评估目标（`_find_nearest_visible_player()`）
- enemy.gd：目标死亡即时处理——追击中找新目标或回巡逻，攻击中中断到后摇
- enemy.gd：PATROL阶段用 `_find_nearest_visible_player()` 替代原 `_can_see_player()`
- enemy.gd：RETURN阶段用 `_find_nearest_visible_player()` 替代原 `_can_see_player()`
- enemy.gd：`_pick_respawn_position()` 改为远离所有玩家（遍历 `_get_alive_players()`）
- enemy.gd：删除 `_can_see_player()` 和 `var player` 的初始化
- skill_target_player.gd：从 `get_first_node_in_group("player")` 单目标改为遍历整个player group多目标扫描
- skill_target_player.gd：targets从单个Player变为Array[Player]，扇形范围内所有玩家都可被命中
- skill_get_target.gd：Entity过滤增加 `not parent.is_dead`，跳过死亡实体
- 不引入目标互斥：多敌人可追同一玩家

## 代码改动记录（历史）
- enemy.gd：6状态机(PATROL/CHASE/ATTACK_WINDUP/ATTACKING/ATTACK_RECOVERY/RETURN)
- enemy.gd：前摇闪烁提示，攻击中锁定不移动，ATTACKING计时器驱动(0.75s)
- enemy.gd：双偏好点寻路，侧面站位(attack_position_offset=20)
- enemy.gd：重生系统，8秒倒计时随机位置复活
- PlayScene.tscn：Road节点groups=["road"]，动态获取巡逻范围
- skeleton_slash.tscn：execution_delay_time=0.3；check_range=35, attack_fov=100

## 关键文件路径
- 敌人逻辑: scripts/entity/enemy.gd
- 技能目标获取: scripts/skills/core/components/skill_target_player.gd
- 技能伤害: scripts/skills/core/components/skill_deal_damage.gd
- 敌人技能场景: assets/scenes/skill_scene/skeleton_slash.tscn
- 着色器: shaders/entity.gdshader（受击闪烁）
