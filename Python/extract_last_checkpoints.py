"""
提取智能体1、2、3最近20个对手池检查点
按回合数和训练步数排序，取最后20个
"""
import os
import re
import shutil
from pathlib import Path

SRC_DIR = Path(r"D:\schoolTour\softwares\multi-agent-gameplay\saved_models\ippo_pool_checkpoints")
DST_DIR = Path(r"D:\schoolTour\softwares\multi-agent-gameplay\saved_models\ippo_pool_last20")

def parse_filename(fname: str) -> dict:
    """解析文件名: round{R}_agent{id}_step{step}_agent{id}.pt 或 round{R}_agent{id}_end_step{step}_agent{id}.pt"""
    m = re.match(r'round(\d+)_agent(\d+)_?(end_)?step(\d+)_agent(\d+)\.pt', fname)
    if m:
        return {
            'round': int(m.group(1)),
            'agent': int(m.group(2)),
            'is_end': bool(m.group(3)),
            'step': int(m.group(4)),
            'filename': fname
        }
    return None

def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)

    # 收集所有 agent 1/2/3 的检查点
    all_files = []
    for fname in os.listdir(SRC_DIR):
        info = parse_filename(fname)
        if info and info['agent'] in (1, 2, 3):
            all_files.append(info)

    # 按 agent 分组
    for agent_id in (1, 2, 3):
        agent_files = [f for f in all_files if f['agent'] == agent_id]
        # 按回合降序，回合内按步数降序
        agent_files.sort(key=lambda x: (x['round'], x['step']), reverse=True)
        last20 = agent_files[:20]
        
        # 再次按回合升序排列输出
        last20.sort(key=lambda x: (x['round'], x['step']))

        subdir = DST_DIR / f"agent_{agent_id}"
        subdir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Agent {agent_id}: 提取最近 {len(last20)} 个检查点")
        print(f"{'='*60}")
        
        for info in last20:
            src = SRC_DIR / info['filename']
            dst = subdir / info['filename']
            shutil.copy2(src, dst)
            tag = "[END]" if info['is_end'] else "[CHK]"
            print(f"  {tag} Round {info['round']:>2d} | Step {info['step']:>10d} | → agent_{agent_id}/{info['filename']}")

        # 汇总
        rounds = sorted(set(f['round'] for f in last20))
        end_count = sum(1 for f in last20 if f['is_end'])
        print(f"\n  回合范围: round{min(rounds)} ~ round{max(rounds)}")
        print(f"  包含 {end_count} 个 end_checkpoint + {len(last20) - end_count} 个中间检查点")

    print(f"\n全部提取完成 → {DST_DIR}")

if __name__ == '__main__':
    main()
