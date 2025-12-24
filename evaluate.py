"""
evaluate.py - Agent 评估脚本（带日志记录）

功能：
- 让两个 Agent 进行多局对战
- 统计胜负和得分
- 支持切换先后手和球型分配
- 记录详细的对局日志（可选）

使用方式：
1. 修改 agent_b 为你设计的待测试的 Agent， 与课程提供的BasicAgent对打
2. 调整 n_games 设置对战局数（评分时设置为120局来计算胜率）
3. 设置 ENABLE_LOGGING = True 启用日志记录
4. 运行脚本查看结果
"""

# ============ 日志配置 ============
ENABLE_LOGGING = True  # 设为 False 关闭日志记录
LOG_BASE_DIR = "logs"  # 日志基础目录
# ===================================

# 导入必要的模块
from utils import set_random_seed
from poolenv import PoolEnv
from agent import BasicAgent, NewAgent
import os
import json
from datetime import datetime
import shutil


# ============ 日志记录器 ============
class GameLogger:
    """游戏日志记录器"""
    
    def __init__(self, enabled=True, base_dir="logs"):
        self.enabled = enabled
        self.base_dir = base_dir
        self.log_dir = None
        self.current_game_log = None
        self.game_index = 0
        
        if self.enabled:
            # 创建时间戳目录
            timestamp = datetime.now().strftime("%m%d_%H%M%S")
            self.log_dir = os.path.join(base_dir, timestamp)
            os.makedirs(self.log_dir, exist_ok=True)
            
            # 创建子目录
            os.makedirs(os.path.join(self.log_dir, "games"), exist_ok=True)
            os.makedirs(os.path.join(self.log_dir, "vlm_prompts"), exist_ok=True)
            os.makedirs(os.path.join(self.log_dir, "vlm_images"), exist_ok=True)
            
            print(f"[Logger] 日志记录已启用，保存至: {self.log_dir}")
            
            # 创建总结文件
            self.summary_file = os.path.join(self.log_dir, "summary.json")
            self.summary = {
                'start_time': datetime.now().isoformat(),
                'games': [],
                'results': {},
                'config': {}
            }
    
    def start_game(self, game_index, player_a_name, player_b_name, 
                   target_ball, first_player):
        """开始记录一局游戏"""
        if not self.enabled:
            return
        
        self.game_index = game_index
        self.current_game_log = {
            'game_index': game_index,
            'player_a': player_a_name,
            'player_b': player_b_name,
            'target_ball': target_ball,
            'first_player': first_player,
            'start_time': datetime.now().isoformat(),
            'shots': [],
            'result': None,
            'vlm_calls': []
        }
    
    def log_shot(self, player, shot_index, observation, action, step_info):
        """记录一次击球"""
        if not self.enabled or self.current_game_log is None:
            return
        
        shot_log = {
            'shot_index': shot_index,
            'player': player,
            'action': {k: float(v) if isinstance(v, (int, float)) else v 
                      for k, v in action.items()},
            'step_info': {
                'ME_INTO_POCKET': step_info.get('ME_INTO_POCKET', []),
                'ENEMY_INTO_POCKET': step_info.get('ENEMY_INTO_POCKET', []),
                'FOUL_FIRST_HIT': step_info.get('FOUL_FIRST_HIT', False),
                'NO_POCKET_NO_RAIL': step_info.get('NO_POCKET_NO_RAIL', False),
                'NO_HIT': step_info.get('NO_HIT', False),
                'CUE_INTO_POCKET': step_info.get('CUE_INTO_POCKET', False)
            }
        }
        
        self.current_game_log['shots'].append(shot_log)
    
    def log_vlm_call(self, game_index, shot_index, prompt_text, 
                     image_path, response_text, strategy):
        """记录VLM调用"""
        if not self.enabled:
            return
        
        vlm_log = {
            'game_index': game_index,
            'shot_index': shot_index,
            'timestamp': datetime.now().isoformat(),
            'prompt_text': prompt_text,
            'image_path': image_path,
            'response_text': response_text,
            'parsed_strategy': strategy
        }
        
        if self.current_game_log is not None:
            self.current_game_log['vlm_calls'].append(vlm_log)
        
        # 同时保存到独立的VLM日志文件
        vlm_log_file = os.path.join(
            self.log_dir, "vlm_prompts", 
            f"game{game_index}_shot{shot_index}.json"
        )
        with open(vlm_log_file, 'w', encoding='utf-8') as f:
            json.dump(vlm_log, f, indent=2, ensure_ascii=False)
    
    def end_game(self, winner, reason=None):
        """结束一局游戏"""
        if not self.enabled or self.current_game_log is None:
            return
        
        self.current_game_log['end_time'] = datetime.now().isoformat()
        self.current_game_log['result'] = {
            'winner': winner,
            'reason': reason,
            'total_shots': len(self.current_game_log['shots'])
        }
        
        # 保存当前游戏日志
        game_log_file = os.path.join(
            self.log_dir, "games", f"game_{self.game_index}.json"
        )
        with open(game_log_file, 'w', encoding='utf-8') as f:
            json.dump(self.current_game_log, f, indent=2, ensure_ascii=False)
        
        # 添加到总结
        self.summary['games'].append({
            'game_index': self.game_index,
            'winner': winner,
            'shots': len(self.current_game_log['shots']),
            'vlm_calls': len(self.current_game_log['vlm_calls'])
        })
        
        self.current_game_log = None
    
    def finalize(self, results, time_stats=None):
        """完成所有记录，保存总结"""
        if not self.enabled:
            return
        
        self.summary['end_time'] = datetime.now().isoformat()
        self.summary['results'] = results
        if time_stats:
            self.summary['time_stats'] = time_stats
        
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            json.dump(self.summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n[Logger] 日志已保存至: {self.log_dir}")
        print(f"  - 总结: summary.json")
        print(f"  - 对局详情: games/game_*.json")
        print(f"  - VLM记录: vlm_prompts/ 和 vlm_images/")
    
    def get_vlm_log_dir(self):
        """获取VLM日志目录（供VLM agent使用）"""
        if not self.enabled:
            return None
        return {
            'prompts_dir': os.path.join(self.log_dir, "vlm_prompts"),
            'images_dir': os.path.join(self.log_dir, "vlm_images")
        }


# ============ 主评估逻辑 ============

# 设置随机种子
set_random_seed(enable=False, seed=42)

# 初始化日志记录器
logger = GameLogger(enabled=ENABLE_LOGGING, base_dir=LOG_BASE_DIR)

# 将日志目录传递给agents（如果是VLM agent）
vlm_log_dirs = logger.get_vlm_log_dir()

env = PoolEnv()
results = {'AGENT_A_WIN': 0, 'AGENT_B_WIN': 0, 'SAME': 0}
n_games = 20  # 对战局数

agent_a, agent_b = BasicAgent(), NewAgent()

# 配置VLM agent的日志记录
for agent in [agent_a, agent_b]:
    if hasattr(agent, 'agent') and agent.agent is not None:
        if hasattr(agent.agent, 'set_logger'):
            agent.agent.set_logger(logger)
            print(f"[Evaluate] Logger configured for {agent.agent.__class__.__name__}")

players = [agent_a, agent_b]
target_ball_choice = ['solid', 'solid', 'stripe', 'stripe']

# 记录配置信息
if ENABLE_LOGGING:
    logger.summary['config'] = {
        'n_games': n_games,
        'agent_a': agent_a.__class__.__name__,
        'agent_b': agent_b.__class__.__name__,
        'agent_a_type': getattr(agent_a, 'agent_type', 'unknown'),
        'agent_b_type': getattr(agent_b, 'agent_type', 'unknown'),
    }

# 初始化时间管理器
for player in players:
    if hasattr(player, 'agent') and player.agent is not None:
        if hasattr(player.agent, 'time_manager'):
            player.agent.time_manager.initialize(n_games, time_per_game=180.0)
            print(f"[Evaluate] Initialized global time manager for {player.agent.__class__.__name__}")

# 主循环
for i in range(n_games): 
    print()
    print(f"------- 第 {i} 局比赛开始 -------")
    env.reset(target_ball=target_ball_choice[i % 4])
    
    # 通知时间管理器和agents
    for player in players:
        if hasattr(player, 'agent') and player.agent is not None:
            # 优先调用agent自己的start_new_game方法（用于VLM等需要跟踪game索引的agent）
            if hasattr(player.agent, 'start_new_game'):
                player.agent.start_new_game()
            # 否则调用time_manager的start_new_game
            elif hasattr(player.agent, 'time_manager') and hasattr(player.agent.time_manager, 'start_new_game'):
                player.agent.time_manager.start_new_game()
            # 或者调用reset_game_timer
            elif hasattr(player.agent, 'reset_game_timer'):
                player.agent.reset_game_timer()
    
    # 开始游戏日志
    player_a_class = players[i % 2].__class__.__name__
    player_b_class = players[(i + 1) % 2].__class__.__name__
    ball_type = target_ball_choice[i % 4]
    first_player = 'A' if i % 2 == 0 else 'B'
    
    logger.start_game(i, player_a_class, player_b_class, ball_type, first_player)
    
    print(f"本局 Player A: {player_a_class}, 目标球型: {ball_type}")
    
    shot_count = 0
    while True:
        player = env.get_curr_player()
        print(f"[第{env.hit_count}次击球] player: {player}")
        obs = env.get_observation(player)
        
        # 获取动作
        if player == 'A':
            action = players[i % 2].decision(*obs)
        else:
            action = players[(i + 1) % 2].decision(*obs)
        
        # 执行动作
        step_info = env.take_shot(action)
        
        # 记录击球
        logger.log_shot(player, shot_count, obs, action, step_info)
        shot_count += 1
        
        # 检查游戏是否结束
        done, info = env.get_done()
        if not done:
            if step_info.get('ENEMY_INTO_POCKET'):
                print(f"对方球入袋：{step_info['ENEMY_INTO_POCKET']}")
        
        if done:
            # 统计结果
            if info['winner'] == 'SAME':
                results['SAME'] += 1
                winner = 'SAME'
            elif info['winner'] == 'A':
                winner_key = ['AGENT_A_WIN', 'AGENT_B_WIN'][i % 2]
                results[winner_key] += 1
                winner = 'AGENT_A' if i % 2 == 0 else 'AGENT_B'
            else:
                winner_key = ['AGENT_A_WIN', 'AGENT_B_WIN'][(i+1) % 2]
                results[winner_key] += 1
                winner = 'AGENT_B' if i % 2 == 0 else 'AGENT_A'
            
            # 结束游戏日志
            logger.end_game(winner, reason=info.get('reason', 'normal'))
            break

# 计算分数
results['AGENT_A_SCORE'] = results['AGENT_A_WIN'] * 1 + results['SAME'] * 0.5
results['AGENT_B_SCORE'] = results['AGENT_B_WIN'] * 1 + results['SAME'] * 0.5

print("\n最终结果：", results)

# 收集时间统计
time_stats = {}
for idx, player in enumerate(players):
    if hasattr(player, 'agent') and player.agent is not None:
        if hasattr(player.agent, 'time_manager') and hasattr(player.agent.time_manager, 'get_stats'):
            stats = player.agent.time_manager.get_stats()
            agent_name = f"Agent_{'A' if idx == 0 else 'B'}"
            time_stats[agent_name] = stats
            
            print(f"\n[{player.agent.__class__.__name__}] Time Statistics:")
            print(f"  Total Budget: {stats['budget']:.0f}s")
            print(f"  Time Used: {stats['elapsed']:.0f}s ({stats['utilization']*100:.1f}%)")
            print(f"  Time Remaining: {stats['remaining']:.0f}s")
            print(f"  Total Decisions: {stats['decisions']}")
            print(f"  Avg Time/Decision: {stats['avg_time_per_decision']:.2f}s")
            print(f"  Games Completed: {stats['games_completed']}/{n_games}")

# 完成日志记录
logger.finalize(results, time_stats)
