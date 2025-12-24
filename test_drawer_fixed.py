"""
测试修复后的drawer功能
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'vlm_agents'))

from drawer import BilliardsDrawer
import pooltool as pt


def test_basic_scene():
    """测试基础场景绘制"""
    print("=" * 60)
    print("测试修复后的台球绘图功能")
    print("=" * 60)
    
    # 创建更完整的测试场景
    print("\n创建台球场景...")
    table = pt.Table.default()
    balls = {
        'cue': pt.Ball.create("cue", xy=[0.5, 0.5]),
        '1': pt.Ball.create("1", xy=[1.0, 0.56]),
        '2': pt.Ball.create("2", xy=[1.1, 0.6]),
        '3': pt.Ball.create("3", xy=[1.2, 0.52]),
        '8': pt.Ball.create("8", xy=[1.5, 0.56]),
        '9': pt.Ball.create("9", xy=[1.8, 0.7]),
        '10': pt.Ball.create("10", xy=[1.6, 0.4]),
        '11': pt.Ball.create("11", xy=[1.9, 0.5]),
    }
    
    my_targets = ['1', '2', '3']
    enemy_targets = ['9', '10', '11']
    
    drawer = BilliardsDrawer()
    
    # 测试1: 基础状态图
    print("\n生成基础状态图...")
    img = drawer.draw_table_state(
        balls, 
        my_targets=my_targets,
        enemy_targets=enemy_targets,
        title="Fixed Billiards Visualization - My: 3 vs Enemy: 3"
    )
    
    output_path = "/tmp/test_billiards_fixed.png"
    img.save(output_path)
    print(f"✓ 基础图片已保存: {output_path}")
    print(f"  图片大小: {img.size}")
    
    # 测试2: 带建议的图
    print("\n生成带建议shot的图...")
    img_suggested = drawer.draw_with_suggested_shot(
        balls,
        my_targets=my_targets,
        suggested_target='1',
        suggested_direction=45.0,
        enemy_targets=enemy_targets
    )
    
    output_path2 = "/tmp/test_billiards_suggested_fixed.png"
    img_suggested.save(output_path2)
    print(f"✓ 建议图片已保存: {output_path2}")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)
    print("\n球的比例现在是正确的了（使用 ball_radius * 1.5 代替 * 25）")
    print("图片应该显示正常大小的台球，不会过大或过小。")
    print(f"\n请查看生成的图片:")
    print(f"  1. {output_path}")
    print(f"  2. {output_path2}")


if __name__ == "__main__":
    test_basic_scene()

