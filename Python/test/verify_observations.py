"""
观测数据验证脚本
检查 Godot 发送的观测数据中各区段是否包含非零信息。
用法: 需先启动 Godot 游戏环境 (training 模式)
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "training"))

import numpy as np
from godot_env_wrapper import GodotDiscreteEnvWrapper, ObsSegmentDims

def verify():
    print("=" * 60)
    print("观测数据管线验证")
    print("=" * 60)
    
    # 读取维度配置
    seg = ObsSegmentDims.from_config(
        str(pathlib.Path(__file__).resolve().parent.parent.parent 
            / "godot-game/configs/game_config.tres")
    )
    print(f"\n[维度配置] use_observation_valid_mask=true, ray_count=36")
    print(f"  self_dim   = {seg.self_dim}")
    print(f"  player_dim = {seg.player_dim}")
    print(f"  ball_dim   = {seg.ball_dim}")
    print(f"  enemy_dim  = {seg.enemy_dim}")
    print(f"  map_dim    = {seg.map_dim}")
    print(f"  TOTAL      = {seg.total}")
    
    # 连接环境
    env_path = str(pathlib.Path(__file__).resolve().parent.parent.parent 
                   / "godot-game/build-multiagent/game.exe")
    
    try:
        envs = GodotDiscreteEnvWrapper(
            env_path=env_path,
            show_window=True,
            speedup=1,
            seed=42,
            n_parallel=1,
            port=11008,
        )
    except Exception as e:
        print(f"\n[ERROR] 无法连接 Godot: {e}")
        print("请确保 Godot 游戏已在 training 模式下运行")
        return
    
    obs_space = envs.single_observation_space
    print(f"\n[环境确认] obs_space.shape = {obs_space.shape}")
    assert obs_space.shape[0] == seg.total, (
        f"维度不匹配! Godot报告={obs_space.shape[0]}, Python计算={seg.total}"
    )
    
    # 采样多个时间步，检查各区段
    n_samples = 200
    seg_stats = {
        "self": {"nonzero_count": 0, "max_abs": 0.0},
        "player": {"nonzero_count": 0, "max_abs": 0.0},
        "ball": {"nonzero_count": 0, "max_abs": 0.0},
        "enemy": {"nonzero_count": 0, "max_abs": 0.0},
        "map": {"nonzero_count": 0, "max_abs": 0.0},
    }
    
    obs, _ = envs.reset()
    obs = np.array(obs[0], dtype=np.float32)
    assert len(obs) == seg.total, f"reset obs维度={len(obs)}, 期望={seg.total}"
    
    offsets = {
        "self":  (0, seg.self_dim),
        "player": (seg.self_dim, seg.self_dim + seg.player_dim),
        "ball":   (seg.self_dim + seg.player_dim, 
                   seg.self_dim + seg.player_dim + seg.ball_dim),
        "enemy":  (seg.self_dim + seg.player_dim + seg.ball_dim,
                   seg.self_dim + seg.player_dim + seg.ball_dim + seg.enemy_dim),
        "map":    (seg.self_dim + seg.player_dim + seg.ball_dim + seg.enemy_dim,
                   seg.total),
    }
    
    print(f"\n[观测采样] 采集 {n_samples} 步观测...")
    all_steps_with_nonzero = {k: 0 for k in seg_stats}
    
    for step in range(n_samples):
        action = np.random.randint(0, 6)
        obs_raw, reward, term, trunc, info = envs.step(np.array([action]))
        obs = np.array(obs_raw[0], dtype=np.float32)
        
        for name, (start, end) in offsets.items():
            segment = obs[start:end]
            max_abs = float(np.max(np.abs(segment)))
            seg_stats[name]["max_abs"] = max(seg_stats[name]["max_abs"], max_abs)
            if max_abs > 1e-6:
                seg_stats[name]["nonzero_count"] += 1
                all_steps_with_nonzero[name] += 1
    
    # 打印结果
    print(f"\n{'区段':<10} {'含非零步数':>10} {'占比':>8} {'最大绝对值':>12} {'状态'}")
    print("-" * 55)
    for name in ["self", "player", "ball", "enemy", "map"]:
        nz = seg_stats[name]["nonzero_count"]
        pct = nz / n_samples * 100
        mv = seg_stats[name]["max_abs"]
        status = "✅" if pct > 10 else "⚠️" if pct > 0 else "🔴"
        
        if name in ["player", "enemy"]:
            if pct < 5:
                detail = f"(视野内实体出现频率过低)"
            else:
                detail = ""
        else:
            detail = ""
        
        print(f"  {name:<8} {nz:>8} / {n_samples}  {pct:>6.1f}%  {mv:>10.4f}    {status} {detail}")
    
    # 详细检查: 打印倒数第二步各段前20个值
    print(f"\n[详细数值] 最后一步观测各区段前 20 维:")
    obs_raw, _, _, _, _ = envs.step(np.array([action]))
    obs = np.array(obs_raw[0], dtype=np.float32)
    
    for name, (start, end) in offsets.items():
        segment = obs[start:min(start+20, end)]
        nonzero_idx = np.where(np.abs(segment) > 1e-6)[0]
        print(f"  {name}[{start}:{end}]: {segment}")
        if len(nonzero_idx) == 0:
            print(f"    → 全部为零!")
        else:
            print(f"    → 非零索引: {list(nonzero_idx)}")
    
    envs.close()
    
    # 诊断总结
    print(f"\n{'='*60}")
    print("诊断总结:")
    print(f"{'='*60}")
    
    issues = []
    if seg_stats["player"]["nonzero_count"] / n_samples < 0.1:
        issues.append(
            "🔴 PLAYER 区段几乎全零 → vision_radius(200) 可能太小，"
            "智能体大部分时间看不到对手玩家"
        )
    if seg_stats["enemy"]["nonzero_count"] / n_samples < 0.1:
        issues.append(
            "🔴 ENEMY 区段几乎全零 → 智能体看不到敌人; "
            "敌人 sight_range=105 远小于 vision_radius=200, "
            "但敌人集中在中心区域，角落智能体距离 >500px"
        )
    
    if issues:
        for issue in issues:
            print(f"  {issue}")
        print(f"\n  建议: 增大 vision_radius (当前200 → 建议600+), "
              "或将出生点移至 arena 内部")
    else:
        print("  观测数据正常，问题可能在网络学习层面。")

if __name__ == "__main__":
    verify()
