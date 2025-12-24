"""
诊断坐标系统问题
"""

import sys
import os
from poolenv import PoolEnv


def diagnose():
    print("=" * 70)
    print("诊断球的坐标系统")
    print("=" * 70)
    
    env = PoolEnv()
    env.reset(target_ball='solid')
    
    balls, my_targets, table = env.get_observation('A')
    
    # 检查table对象
    print("\n1. 检查Table对象:")
    print(f"  Table类型: {type(table)}")
    
    if hasattr(table, 'w'):
        print(f"  Table宽度(w): {table.w}")
    if hasattr(table, 'l'):
        print(f"  Table长度(l): {table.l}")
    if hasattr(table, 'width'):
        print(f"  Table width: {table.width}")
    if hasattr(table, 'length'):
        print(f"  Table length: {table.length}")
    
    # 检查所有球的坐标
    print("\n2. 所有球的坐标:")
    print(f"  {'球ID':<6} {'X坐标':>10} {'Y坐标':>10} {'状态':>6}")
    print("  " + "-" * 36)
    
    for ball_id in sorted(balls.keys(), key=lambda x: (x != 'cue', x)):
        ball = balls[ball_id]
        if hasattr(ball, 'state'):
            pos = ball.state.rvw[0][:2]
            status = ball.state.s if hasattr(ball.state, 's') else '?'
            print(f"  {ball_id:<6} {pos[0]:>10.3f} {pos[1]:>10.3f} {status:>6}")
    
    # 分析坐标范围
    print("\n3. 坐标范围分析:")
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
        
        # 检查是否有球超出范围
        if max(y_coords) > 1.12:
            print(f"\n  ⚠️  警告: 有球的Y坐标 > 1.12 (标准球桌高度)")
            print(f"       最大Y坐标: {max(y_coords):.3f}")
            
            # 找出所有Y > 1.12的球
            out_of_bounds = [(bid, y) for bid, x, y in active_balls if y > 1.12]
            print(f"       超出范围的球: {len(out_of_bounds)}/{len(active_balls)}")
            for bid, y in out_of_bounds[:5]:
                print(f"         Ball {bid}: Y={y:.3f}")
    
    # 检查球桌规格
    print("\n4. 可能的坐标系统:")
    print("  方案A: 标准台球桌 2.24m(长) x 1.12m(宽)")
    print("  方案B: pooltool可能使用不同的单位或坐标系")
    
    # 猜测实际的台球桌尺寸
    if active_balls:
        x_coords = [x for _, x, y in active_balls]
        y_coords = [y for _, x, y in active_balls]
        
        actual_width = max(x_coords) - min(x_coords) + 0.2  # 加一些边距
        actual_height = max(y_coords) - min(y_coords) + 0.2
        
        print(f"\n5. 根据球的分布推测实际台球桌尺寸:")
        print(f"  推测宽度: {actual_width:.3f} (X方向)")
        print(f"  推测高度: {actual_height:.3f} (Y方向)")
        
        # 检查是否是2倍关系
        ratio = max(y_coords) / 1.12
        if ratio > 1.3:
            print(f"\n  💡 发现: Y坐标比例约为 {ratio:.2f}")
            print(f"     可能的球桌实际高度: {max(y_coords) + 0.2:.3f}")


if __name__ == "__main__":
    diagnose()

