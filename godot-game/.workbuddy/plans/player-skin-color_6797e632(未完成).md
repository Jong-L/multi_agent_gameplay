---
name: player-skin-color
overview: 利用素材包自带的5种颜色Warrior贴图（Blue/Red/Yellow/Purple/Black），为每个Player实例添加可配置的颜色皮肤。通过在Player中新增@export参数指定颜色，运行时动态替换SpriteFrames中的贴图资源。
todos:
  - id: add-skin-color
    content: 在 player.gd 中新增 skin_color 导出属性和 _apply_skin_color 运行时换肤逻辑
    status: pending
---

## 用户需求

为 PlayScene 中的4个 Player 实例分配不同颜色的外观，每个 Player 使用素材包中不同颜色目录下的 Warrior 贴图（Blue/Red/Yellow/Purple/Black），动画结构完全相同，只是贴图不同。

## 产品概述

4个 Player 在 PlayScene 中是同一个 Player.tscn 的实例，当前共享同一套蓝色贴图。需要在编辑器中为每个实例指定皮肤颜色，运行时自动加载对应颜色的贴图。

## 核心功能

- Player 新增 `skin_color` 导出属性，在编辑器 Inspector 中可选 Blue/Red/Yellow/Purple/Black
- 运行时根据 skin_color 自动替换 AnimatedSprite2D 的 SpriteFrames 中所有 AtlasTexture 的 atlas 贴图
- 不同 Player 实例之间互不影响

## Tech Stack

- Godot 4.x GDScript（沿用现有项目技术栈）

## Implementation Approach

**核心思路**：利用素材包自带5种颜色 Warrior 贴图，在运行时动态替换 SpriteFrames 中的 atlas 引用。

**具体方案**：

1. 在 `player.gd` 中新增 `@export var skin_color: String = "Blue"`
2. 在 `_ready()` 中，先将 `sprite_frames` 做 `duplicate()` 使其独立（避免改一个全变）
3. 遍历 SpriteFrames 所有动画的所有帧，将每帧的 AtlasTexture 的 atlas 替换为对应颜色目录下的贴图
4. 建立贴图映射：原始贴图路径中的 `Blue Units` 替换为 `{skin_color} Units`，用 `load()` 加载新贴图

**贴图映射关系**（从 Player.tscn 分析）：

- `die` 动画 → `Warrior_Run.png`（8帧，64x64裁切）
- `idle` 动画 → `Warrior_Idle.png`（8帧，192x192裁切）
- `run` 动画 → `Warrior_Run.png`（6帧，192x192裁切）

**关键设计决策**：

- 不在 shader 中做换色，直接用官方美术贴图，颜色协调性更好
- 不修改 Player.tscn 场景文件本身，只在运行时替换，保持场景文件简洁
- `sprite_frames.duplicate()` 确保每个实例的帧数据独立
- 贴图用 `load()` 加载（非 `preload`），因为颜色是运行时变量决定的

**性能**：`load()` 在 _ready 中只执行一次，3张贴图加载开销可忽略不计。

## Architecture Design

- 修改范围仅限 `player.gd`，不涉及 shader、entity.gd 或场景文件
- 在 `player.gd._ready()` 的 `super._ready()` 之后执行换肤逻辑

## Directory Structure

```
scripts/entity/
└── player.gd  # [MODIFY] 新增 skin_color 导出属性和 _apply_skin_color() 换肤方法
```