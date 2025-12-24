"""
测试VLM Agent的各个组件
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'vlm_agents'))

import pooltool as pt
import numpy as np


def test_drawer():
    """测试图片绘制功能"""
    print("\n=== 测试 1: Drawer (图片绘制) ===")
    
    from drawer import BilliardsDrawer
    
    # 创建简单测试场景
    table = pt.Table.default()
    balls = {
        'cue': pt.Ball.create("cue", xy=[0.5, 0.5]),
        '1': pt.Ball.create("1", xy=[1.0, 0.56]),
        '2': pt.Ball.create("2", xy=[1.1, 0.6]),
        '8': pt.Ball.create("8", xy=[1.5, 0.56]),
        '9': pt.Ball.create("9", xy=[1.8, 0.7]),
        '10': pt.Ball.create("10", xy=[1.6, 0.4]),
    }
    
    drawer = BilliardsDrawer()
    
    # 测试基础绘图
    print("  绘制基础对局图...")
    img = drawer.draw_table_state(
        balls, 
        my_targets=['1', '2'], 
        enemy_targets=['9', '10'],
        title="Test Game State"
    )
    
    output_path = "/tmp/test_billiards_basic.png"
    img.save(output_path)
    print(f"  ✓ 基础图片已保存: {output_path}")
    print(f"  图片大小: {img.size}")
    
    # 测试带建议的绘图
    print("  绘制带建议shot的图...")
    img_with_suggestion = drawer.draw_with_suggested_shot(
        balls,
        my_targets=['1', '2'],
        suggested_target='1',
        suggested_direction=45.0,
        enemy_targets=['9', '10']
    )
    
    output_path2 = "/tmp/test_billiards_suggested.png"
    img_with_suggestion.save(output_path2)
    print(f"  ✓ 建议图片已保存: {output_path2}")
    
    return True


def test_chat_fallback():
    """测试Chat模块的降级功能（不调用API）"""
    print("\n=== 测试 2: Chat (降级模式，无需API) ===")
    
    from chat import VLMChat
    
    # 测试降级策略
    vlm = VLMChat(provider='openai')  # 即使没有API key，也应该能降级
    
    strategy = vlm._fallback_strategy(
        my_remaining=3,
        enemy_remaining=5,
        my_targets=['1', '2', '3']
    )
    
    print(f"  降级策略结果:")
    print(f"    Strategy: {strategy['strategy']}")
    print(f"    Risk tolerance: {strategy['risk_tolerance']}")
    print(f"    Target priority: {strategy['target_priority']}")
    print(f"  ✓ 降级功能正常")
    
    return True


def test_vlm_agent_without_api():
    """测试VLM Agent（不使用API，纯启发式模式）"""
    print("\n=== 测试 3: VLM Agent (启发式模式，无需API) ===")
    
    from VlmAssistedAgent import VLMAssistedAgent
    
    # 创建agent（禁用VLM）
    print("  初始化VLM Agent (use_vlm=False)...")
    agent = VLMAssistedAgent(
        vlm_provider='openai',
        use_vlm=False,  # 禁用VLM，使用启发式
        n_cores=4
    )
    
    # 创建测试场景
    table = pt.Table.default()
    balls = {
        'cue': pt.Ball.create("cue", xy=[0.5, 0.5]),
        '1': pt.Ball.create("1", xy=[1.0, 0.56]),
        '2': pt.Ball.create("2", xy=[1.1, 0.6]),
        '8': pt.Ball.create("8", xy=[1.5, 0.56]),
    }
    
    my_targets = ['1', '2']
    
    # 初始化时间管理
    agent.time_manager.initialize(n_games=1, time_per_game=60.0)
    
    # 执行决策
    print("  执行决策...")
    action = agent.decision(balls=balls, my_targets=my_targets, table=table)
    
    print(f"  决策结果:")
    print(f"    V0 (速度): {action['V0']:.2f}")
    print(f"    phi (角度): {action['phi']:.2f}°")
    print(f"    theta: {action['theta']:.2f}°")
    print(f"  ✓ Agent决策功能正常")
    
    return True


def test_integration():
    """测试与主agent系统的集成"""
    print("\n=== 测试 4: 主Agent系统集成 ===")
    
    # 切换回agents目录
    sys.path.insert(0, os.path.dirname(__file__))
    
    from agent import NewAgent
    
    print("  从agent.py加载NewAgent (AGENT_TYPE='vlm')...")
    
    # 注意：这会尝试加载VLM agent
    # 如果AGENT_TYPE不是'vlm'，可能会加载其他agent
    try:
        agent = NewAgent()
        print(f"  ✓ Agent已加载")
        
        # 简单测试
        table = pt.Table.default()
        balls = {
            'cue': pt.Ball.create("cue", xy=[0.5, 0.5]),
            '1': pt.Ball.create("1", xy=[1.0, 0.56]),
        }
        
        action = agent.decision(balls=balls, my_targets=['1'], table=table)
        print(f"  ✓ 决策成功: V0={action['V0']:.2f}, phi={action['phi']:.1f}°")
        
        return True
        
    except Exception as e:
        print(f"  ⚠ 集成测试失败（可能是AGENT_TYPE不是'vlm'）: {e}")
        return False


def main():
    """运行所有测试"""
    print("=" * 60)
    print("VLM Agent 组件测试")
    print("=" * 60)
    
    tests = [
        ("Drawer", test_drawer),
        ("Chat Fallback", test_chat_fallback),
        ("VLM Agent (Heuristic Mode)", test_vlm_agent_without_api),
        ("Integration", test_integration),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n  ✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"  {status}: {name}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！VLM Agent已就绪。")
        print("\n使用说明:")
        print("1. 不使用VLM (纯启发式): 设置 use_vlm=False")
        print("2. 使用VLM (需要API key): 设置环境变量 OPENAI_API_KEY 或 ANTHROPIC_API_KEY")
        print("3. 在agent.py中设置 AGENT_TYPE='vlm'")
        print("4. 运行: conda activate poolenv && python evaluate.py")
    else:
        print("\n⚠ 部分测试失败，请检查错误信息")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

