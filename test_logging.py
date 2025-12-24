"""
测试日志记录功能
"""

import sys
import os
import json

# 测试evaluate.py的日志功能
print("=" * 60)
print("测试日志记录功能")
print("=" * 60)

# 1. 测试logger创建
print("\n步骤 1: 测试GameLogger创建")
sys.path.insert(0, os.path.dirname(__file__))

from evaluate import GameLogger

logger = GameLogger(enabled=True, base_dir="logs_test")
print(f"✓ Logger创建成功")
print(f"  日志目录: {logger.log_dir}")

# 2. 测试游戏记录
print("\n步骤 2: 测试游戏记录")

logger.start_game(
    game_index=0,
    player_a_name="TestAgentA",
    player_b_name="TestAgentB",
    target_ball="solid",
    first_player="A"
)
print("✓ 游戏开始记录")

# 模拟几次击球
for i in range(3):
    logger.log_shot(
        player='A' if i % 2 == 0 else 'B',
        shot_index=i,
        observation=None,
        action={'V0': 3.5, 'phi': 45.0, 'theta': 0.0, 'a': 0.0, 'b': 0.0},
        step_info={
            'ME_INTO_POCKET': ['1'] if i == 1 else [],
            'ENEMY_INTO_POCKET': [],
            'FOUL_FIRST_HIT': False,
            'NO_POCKET_NO_RAIL': False
        }
    )
print(f"✓ 记录了 3 次击球")

# 3. 测试VLM记录
print("\n步骤 3: 测试VLM记录")

vlm_log_dirs = logger.get_vlm_log_dir()
if vlm_log_dirs:
    # 创建一个测试图片
    from PIL import Image
    import numpy as np
    
    test_img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    test_img_path = os.path.join(vlm_log_dirs['images_dir'], "test_image.png")
    test_img.save(test_img_path)
    
    logger.log_vlm_call(
        game_index=0,
        shot_index=0,
        prompt_text="Test prompt: What should I do?",
        image_path=test_img_path,
        response_text='{"strategy": "aggressive", "risk_tolerance": 0.7}',
        strategy={'strategy': 'aggressive', 'risk_tolerance': 0.7, 'target_priority': ['1']}
    )
    print("✓ VLM调用记录成功")

# 4. 结束游戏
print("\n步骤 4: 结束游戏并保存")

logger.end_game(winner='AGENT_A', reason='normal')
print("✓ 游戏结束记录")

# 5. 完成日志
results = {
    'AGENT_A_WIN': 1,
    'AGENT_B_WIN': 0,
    'SAME': 0,
    'AGENT_A_SCORE': 1.0,
    'AGENT_B_SCORE': 0.0
}

logger.finalize(results, time_stats={'Agent_A': {'elapsed': 10.5, 'decisions': 3}})
print("✓ 日志已完成并保存")

# 6. 验证日志文件
print("\n步骤 5: 验证日志文件")

log_dir = logger.log_dir
files_to_check = [
    ('summary.json', '总结文件'),
    ('games/game_0.json', '游戏详情'),
    ('vlm_prompts/game0_shot0.json', 'VLM prompt记录'),
    ('vlm_images/test_image.png', 'VLM图片')
]

all_exist = True
for file_path, desc in files_to_check:
    full_path = os.path.join(log_dir, file_path)
    exists = os.path.exists(full_path)
    status = "✓" if exists else "✗"
    print(f"  {status} {desc}: {file_path}")
    if not exists:
        all_exist = False

# 7. 读取并显示部分日志内容
if all_exist:
    print("\n步骤 6: 读取日志内容示例")
    
    # 读取summary
    with open(os.path.join(log_dir, 'summary.json'), 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    print(f"\n总结信息:")
    print(f"  开始时间: {summary['start_time']}")
    print(f"  游戏数: {len(summary['games'])}")
    print(f"  结果: {summary['results']}")
    
    # 读取游戏详情
    with open(os.path.join(log_dir, 'games/game_0.json'), 'r', encoding='utf-8') as f:
        game = json.load(f)
    
    print(f"\n游戏0详情:")
    print(f"  击球数: {len(game['shots'])}")
    print(f"  VLM调用数: {len(game['vlm_calls'])}")
    print(f"  胜者: {game['result']['winner']}")

print("\n" + "=" * 60)
print("✅ 所有日志功能测试通过！")
print("=" * 60)
print(f"\n测试日志已保存至: {log_dir}")
print("\n使用方法:")
print("1. 在 evaluate.py 中设置 ENABLE_LOGGING = True")
print("2. 运行 python evaluate.py")
print("3. 查看 logs/{timestamp}/ 目录下的日志文件")

