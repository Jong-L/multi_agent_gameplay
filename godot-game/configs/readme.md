设计几种性格：

先观察是否有行为差异，没有再引入能力差异。

note：

- 最后一刀会同时有攻击奖励和击杀奖励，本来设计的是击杀奖励是攻击的3倍，于是这里就设计为击杀奖励大概是攻击奖励的2倍。
- 饥饿机制暂不实装
- 撞墙惩罚统一为-0.5



p0(Blue)，p2(Red)：好战型  （p2未来会在回放模型时用随机策略代替，与其他训练过的智能体对抗作为对比，因此主要设计3种性格）

```
collect_ball_A = 2
collect_ball_B = 4
cause_damage_to_enemy =8.0
cause_damage_to_player = 10.0
kill_enemy = 16.0
kill_player = 20.0
bear_damage = -5.0
attack = -0.01
died = -10.0
ball_potential_scale = 0.5  #shaping奖励与奖励球本身奖励有关，比例不用再差异
distance_reward_scale = 0.002

```

p1(Black):避战吃球型

```
# 避战吃球型	
collect_ball_A = 10.0
collect_ball_B = 12.0
cause_damage_to_enemy = 3.0
cause_damage_to_player = 4.0
kill_enemy = 6.0
kill_player = 8.0
bear_damage = -10.0
attack = -0.06
died = -20.0
ball_potential_scale = 0.5
distance_reward_scale = 0.02

```

p3(Yellow):吃球和战斗奖励持平，但是完美主义

```

collect_ball_A = 4
collect_ball_B = 6
cause_damage_to_enemy = 4-attack #抵消攻击惩罚
cause_damage_to_player = 6-attack
kill_enemy = 8
kill_player = 12
bear_damage = -8
attack = -0.05    
died = -15             
ball_potential_scale = 0.5
distance_reward_scale = 0.002

```
