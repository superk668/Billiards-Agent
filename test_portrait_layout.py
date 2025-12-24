"""
测试竖屏布局的drawer
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'vlm_agents'))

from drawer import BilliardsDrawer
from poolenv import PoolEnv


def main():
    print("=" * 70)
    print("测试竖屏布局（长边在y轴，短边在x轴）")
    print("=" * 70)
    
    # 创建真实环境
    env = PoolEnv()
    env.reset(target_ball='solid')
    
    # 获取观察
    balls, my_targets, table = env.get_observation('A')
    
    # 检查table尺寸
    print("\n1. 台球桌尺寸:")
    print(f"  宽度(w): {table.w:.3f} (对应x轴)")
    print(f"  长度(l): {table.l:.3f} (对应y轴)")
    print(f"  比例: {table.l/table.w:.2f}:1")
    
    # 检查球的分布
    print("\n2. 球的分布:")
    active_balls = []
    for ball_id, ball in balls.items():
        if hasattr(ball, 'state') and hasattr(ball.state, 's') and ball.state.s != 4:
            pos = ball.state.rvw[0][:2]
            active_balls.append((ball_id, pos[0], pos[1]))
    
    if active_balls:
        x_coords = [x for _, x, y in active_balls]
        y_coords = [y for _, x, y in active_balls]
        print(f"  X范围: {min(x_coords):.3f} ~ {max(x_coords):.3f}")
        print(f"  Y范围: {min(y_coords):.3f} ~ {max(y_coords):.3f}")
    
    # 绘制图片
    print("\n3. 生成竖屏布局图片...")
    drawer = BilliardsDrawer()
    
    all_balls = set(str(i) for i in range(1, 16))
    enemy_targets = list(all_balls - set(my_targets) - {'8'})
    
    image = drawer.draw_table_state(
        balls,
        my_targets=my_targets,
        enemy_targets=enemy_targets,
        title=f"Portrait Layout Test - My: {len(my_targets)} vs Enemy: {len(enemy_targets)}",
        table=table
    )
    
    output_path = "/home/yuhc/data/AI_project/AI3603-Billiards/vlm_agents/portrait_layout_test.png"
    image.save(output_path)
    
    print(f"  ✓ 保存至: {output_path}")
    print(f"  图片大小: {image.size[0]}x{image.size[1]} (宽x高)")
    
    # 验证要点
    print("\n4. 竖屏布局验证:")
    print("  ✓ 图片应该是竖向的（高度>宽度）")
    print(f"    实际: {image.size[0]}x{image.size[1]}")
    print("  ✓ x轴（水平，短边）：2个袋口（左下角+右下角，左上角+右上角）")
    print("  ✓ y轴（垂直，长边）：额外2个中袋（左侧中点+右侧中点）")
    print("  ✓ 总共6个袋口分布：")
    print("    - 底部2个：(0, 0), (w, 0)")
    print("    - 中部2个：(0, h/2), (w, h/2)")
    print("    - 顶部2个：(0, h), (w, h)")
    
    print("\n" + "=" * 70)
    print("✅ 竖屏布局测试完成！")
    print("=" * 70)
    
    print(f"\n📸 请查看图片验证袋口布局:")
    print(f"  {output_path}")


if __name__ == "__main__":
    main()

