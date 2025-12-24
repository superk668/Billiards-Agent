"""
测试drawer在真实evaluate环境中的表现
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'vlm_agents'))

from drawer import BilliardsDrawer
from poolenv import PoolEnv


def test_real_environment():
    """使用真实的PoolEnv测试drawer"""
    print("=" * 70)
    print("测试Drawer在真实PoolEnv环境中的表现")
    print("=" * 70)
    
    # 创建真实环境
    print("\n步骤 1: 初始化PoolEnv...")
    env = PoolEnv()
    env.reset(target_ball='solid')
    
    # 获取观察
    print("\n步骤 2: 获取游戏观察...")
    balls, my_targets, table = env.get_observation('A')
    
    print(f"  获取到 {len(balls)} 个球")
    print(f"  我方目标: {my_targets}")
    
    # 检查球的结构
    print("\n步骤 3: 检查球对象结构...")
    if 'cue' in balls:
        cue_ball = balls['cue']
        print(f"  Cue ball类型: {type(cue_ball)}")
        print(f"  Cue ball属性: {dir(cue_ball)[:10]}...")  # 只打印前10个属性
        
        # 检查状态
        if hasattr(cue_ball, 'state'):
            print(f"  有state属性")
            if hasattr(cue_ball.state, 'rvw'):
                pos = cue_ball.state.rvw[0][:2]
                print(f"  Cue ball位置: ({pos[0]:.3f}, {pos[1]:.3f})")
            if hasattr(cue_ball.state, 's'):
                print(f"  Cue ball状态码: {cue_ball.state.s}")
    
    # 测试drawer
    print("\n步骤 4: 使用drawer绘制...")
    drawer = BilliardsDrawer()
    
    # 获取所有目标球
    all_balls = set(str(i) for i in range(1, 16))
    enemy_targets = list(all_balls - set(my_targets) - {'8'})
    
    try:
        image = drawer.draw_table_state(
            balls,
            my_targets=my_targets,
            enemy_targets=enemy_targets,
            title="Real Environment Test",
            table=table  # 传递table对象以获取正确尺寸
        )
        
        output_path = "/home/yuhc/data/AI_project/AI3603-Billiards/vlm_agents/billiards_real_env_test.png"
        image.save(output_path)
        
        print(f"\n✅ 图片生成成功!")
        print(f"  保存路径: {output_path}")
        print(f"  图片大小: {image.size}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 图片生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_real_environment()
    
    if success:
        print("\n" + "=" * 70)
        print("✅ 测试成功！Drawer可以处理真实环境的球对象")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("❌ 测试失败，需要进一步调试")
        print("=" * 70)
    
    sys.exit(0 if success else 1)

