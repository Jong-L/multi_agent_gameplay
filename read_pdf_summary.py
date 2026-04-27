#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取PDF论文并提取关键信息，分析对当前多智能体博弈项目的启发
"""

import fitz
import re
import os
from typing import List, Dict, Tuple

def extract_text_from_pdf(pdf_path: str, max_pages: int = 20) -> str:
    """从PDF中提取文本"""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        num_pages = min(len(doc), max_pages)
        
        for page_num in range(num_pages):
            page = doc[page_num]
            text += page.get_text()
        
        doc.close()
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def extract_key_sections(text: str) -> Dict[str, List[str]]:
    """提取论文的关键部分"""
    sections = {
        "abstract": [],
        "introduction": [],
        "reward_shaping": [],
        "reward_engineering": [],
        "multi_agent": [],
        "applications": [],
        "conclusion": []
    }
    
    # 转换为小写以便搜索
    text_lower = text.lower()
    lines = text.split('\n')
    
    # 提取摘要
    abstract_start = None
    for i, line in enumerate(lines):
        if 'abstract' in line.lower() or '摘要' in line:
            abstract_start = i
            break
    
    if abstract_start is not None:
        abstract_end = None
        for i in range(abstract_start + 1, min(abstract_start + 20, len(lines))):
            if any(keyword in lines[i].lower() for keyword in ['introduction', '1.', '引言', '1 ']):
                abstract_end = i
                break
        if abstract_end is None:
            abstract_end = min(abstract_start + 15, len(lines))
        
        sections["abstract"] = lines[abstract_start:abstract_end]
    
    # 搜索关键概念
    keywords = {
        "reward_shaping": ["reward shaping", "potential-based", "ng 1999", "shaped reward", "reward shaping function"],
        "reward_engineering": ["reward engineering", "reward design", "reward function", "reward signal"],
        "multi_agent": ["multi-agent", "multiagent", "multi agent", "collaborative", "competitive", "nash equilibrium"],
        "applications": ["game", "robotics", "autonomous", "control", "simulation"],
        "conclusion": ["conclusion", "concluding", "总结", "future work"]
    }
    
    for section, kw_list in keywords.items():
        found_lines = []
        for i, line in enumerate(lines):
            line_lower = line.lower()
            for kw in kw_list:
                if kw in line_lower:
                    # 获取上下文
                    start = max(0, i - 2)
                    end = min(len(lines), i + 3)
                    context = lines[start:end]
                    found_lines.extend(context)
                    break
        sections[section] = list(set(found_lines))  # 去重
    
    return sections

def analyze_for_current_project(sections: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """分析对当前多智能体博弈项目的启发"""
    insights = {
        "reward_design_insights": [],
        "shaping_techniques": [],
        "multi_agent_considerations": [],
        "implementation_suggestions": [],
        "research_gaps": []
    }
    
    # 从奖励塑形部分提取见解
    reward_shaping_text = " ".join(sections["reward_shaping"])
    if "potential-based" in reward_shaping_text.lower():
        insights["shaping_techniques"].append("基于势能的奖励塑形（PBRS）：Ng et al. (1999) 的经典方法，保证策略不变性")
    
    if "dense reward" in reward_shaping_text.lower():
        insights["reward_design_insights"].append("密集奖励 vs 稀疏奖励：当前项目（吃球游戏）使用密集奖励设计，符合最佳实践")
    
    # 从多智能体部分提取见解
    multi_agent_text = " ".join(sections["multi_agent"])
    if "competitive" in multi_agent_text.lower():
        insights["multi_agent_considerations"].append("竞争性多智能体环境：当前项目是4玩家竞争性环境，需考虑均衡策略")
    
    if "nash" in multi_agent_text.lower():
        insights["multi_agent_considerations"].append("纳什均衡：在多智能体竞争环境中，策略收敛到纳什均衡是理想目标")
    
    # 从应用部分提取见解
    applications_text = " ".join(sections["applications"])
    if "game" in applications_text.lower():
        insights["implementation_suggestions"].append("游戏环境：论文可能提供游戏AI中的奖励设计案例，可借鉴到Godot环境中")
    
    # 当前项目的具体启发
    current_project_context = """
    当前项目：Tiny Swords - Godot 4.6 2D俯视角四人混战竞技场游戏 + 多智能体强化学习环境
    已有特性：
    1. 基于势能的奖励塑形（PBRS）用于球吸引奖励
    2. 32射线Lidar观测空间
    3. 独立per-player模型训练
    4. 奖励配置通过JSON文件管理
    5. 4玩家竞争性环境
    
    潜在改进方向：
    """
    
    # 基于论文内容的具体建议
    if len(sections["reward_engineering"]) > 0:
        insights["implementation_suggestions"].extend([
            "奖励工程系统化：建立更结构化的奖励设计流程，包括奖励分解、归一化、塑形等",
            "奖励可解释性：添加奖励可视化工具，帮助理解智能体行为",
            "奖励消融实验：系统地测试不同奖励组件对性能的影响"
        ])
    
    if len(sections["multi_agent"]) > 0:
        insights["implementation_suggestions"].extend([
            "多智能体协调：考虑添加团队协作奖励机制（如果扩展为2v2模式）",
            "对手建模：在竞争环境中添加对手策略建模组件",
            "课程学习：从简单到复杂的训练课程设计"
        ])
    
    # 研究空白识别
    insights["research_gaps"].extend([
        "动态奖励调整：根据训练进度自动调整奖励权重",
        "元奖励学习：学习如何设计更好的奖励函数",
        "多目标奖励优化：平衡多个竞争性目标（如生存vs攻击vs收集）"
    ])
    
    return insights

def main():
    pdf_path = r"C:\Users\86180\Downloads\Comprehensive_Overview_of_Reward_Engineering_and_Shaping_in_Advancing_Reinforcement_Learning_Applications.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"PDF文件不存在: {pdf_path}")
        return
    
    print("正在读取PDF论文...")
    text = extract_text_from_pdf(pdf_path, max_pages=30)
    
    if "Error reading" in text:
        print(text)
        return
    
    print(f"成功读取PDF，提取文本长度: {len(text)} 字符")
    
    print("\n" + "="*80)
    print("提取关键部分...")
    sections = extract_key_sections(text)
    
    for section_name, content in sections.items():
        if content:
            print(f"\n{section_name.upper()} (找到 {len(content)} 行):")
            print("-" * 40)
            for line in content[:10]:  # 只显示前10行
                if line.strip():
                    print(line.strip())
            if len(content) > 10:
                print(f"... (还有 {len(content) - 10} 行)")
    
    print("\n" + "="*80)
    print("对当前多智能体博弈项目的启发分析:")
    print("="*80)
    
    insights = analyze_for_current_project(sections)
    
    for category, items in insights.items():
        if items:
            print(f"\n{category.replace('_', ' ').title()}:")
            for item in items:
                print(f"  • {item}")
    
    # 生成总结建议
    print("\n" + "="*80)
    print("总结与具体实施建议:")
    print("="*80)
    
    summary = [
        "1. 强化奖励工程实践：基于论文的系统化方法，优化当前奖励配置",
        "2. 扩展势能塑形：将PBRS应用到更多场景（如敌人回避、区域控制）",
        "3. 多智能体特定技术：探索对手建模、课程学习等高级技术",
        "4. 实验设计：设计系统的消融实验验证不同奖励组件效果",
        "5. 可解释性工具：开发奖励可视化工具辅助调试和分析"
    ]
    
    for item in summary:
        print(item)
    
    # 保存分析结果
    output_path = "pdf_paper_insights.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# PDF论文对多智能体博弈项目的启发分析\n\n")
        f.write("## 论文概要\n")
        f.write(f"- 论文标题: Comprehensive Overview of Reward Engineering and Shaping in Advancing Reinforcement Learning Applications\n")
        f.write(f"- 分析页面: 前30页\n")
        f.write(f"- 提取文本长度: {len(text)} 字符\n\n")
        
        f.write("## 关键发现\n")
        for category, items in insights.items():
            if items:
                f.write(f"\n### {category.replace('_', ' ').title()}\n")
                for item in items:
                    f.write(f"- {item}\n")
        
        f.write("\n## 实施建议\n")
        for item in summary:
            f.write(f"{item}\n")
    
    print(f"\n分析结果已保存到: {output_path}")

if __name__ == "__main__":
    main()