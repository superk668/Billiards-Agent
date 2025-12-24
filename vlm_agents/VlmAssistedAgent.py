"""
VlmAssistedAgent.py - VLM-Guided MCTS Billiards Agent

完整流程：
1. 读取环境 → drawer.py生成图片
2. 图片+prompt → chat.py调用VLM → 获取战略指导
3. 基于VLM指导 → VLM-Guided MCTS → 计算最优参数
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pooltool as pt
import copy
import time
import signal
import math
import random
from typing import Dict, List, Tuple, Optional
from collections import deque
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError

# 导入VLM模块
from drawer import BilliardsDrawer
from chat import VLMChat

# 导入Bayesian Optimization（用于参数精细优化）
try:
    from bayes_opt import BayesianOptimization
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    HAS_BAYES_OPT = True
except ImportError:
    print("[VLMAgent] Warning: bayesian-optimization not available")
    HAS_BAYES_OPT = False


# ============ 时间管理器 ============
class VLMTimeManager:
    """时间管理（考虑VLM调用开销）"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VLMTimeManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        self.total_time_budget = 0.0
        self.start_time = None
        self.total_games = 0
        self.current_game = 0
        self.decisions_made = 0
        
        self.decision_time_history = deque(maxlen=100)
        self.vlm_call_time_history = deque(maxlen=50)
        
        self.time_safety_margin = 0.94  # 更保守（VLM调用不可预测）
        self.estimated_avg_decisions_per_game = 25.0
        self.predicted_decision_time = 8.0
        self.predicted_vlm_time = 3.0  # VLM调用预计时间
        
        self.games_completed = 0
        self.total_decisions = 0
        self.total_vlm_calls = 0
    
    def initialize(self, n_games, time_per_game=180.0):
        self.total_time_budget = n_games * time_per_game
        self.start_time = time.time()
        self.total_games = n_games
        self.current_game = 0
        self.decisions_made = 0
        self.games_completed = 0
        self.total_vlm_calls = 0
        
        self.decision_time_history.clear()
        self.vlm_call_time_history.clear()
        
        print(f"[VLMTimeManager] Initialized: {self.total_time_budget:.0f}s for {n_games} games")
        print(f"[VLMTimeManager] VLM-aware time management enabled")
    
    def record_vlm_call(self, call_time: float):
        """记录VLM调用时间"""
        self.vlm_call_time_history.append(call_time)
        self.total_vlm_calls += 1
        
        if len(self.vlm_call_time_history) >= 3:
            self.predicted_vlm_time = np.mean(list(self.vlm_call_time_history)[-10:])
    
    def learn_from_decision(self, decision_time: float):
        self.decision_time_history.append(decision_time)
        self.decisions_made += 1
        self.total_decisions += 1
        
        if len(self.decision_time_history) >= 5:
            recent_avg = np.mean(list(self.decision_time_history)[-10:])
            self.predicted_decision_time = 0.7 * self.predicted_decision_time + 0.3 * recent_avg
    
    def get_time_budget(self, game_state=None, will_call_vlm=False):
        """获取时间预算（考虑VLM调用）"""
        if self.start_time is None:
            return 10.0
        
        elapsed = time.time() - self.start_time
        remaining = self.total_time_budget - elapsed
        safe_remaining = remaining * self.time_safety_margin
        
        if safe_remaining <= 0:
            return 0.5
        
        games_remaining = max(1, self.total_games - self.current_game)
        
        if len(self.decision_time_history) >= 3:
            decisions_in_current_game = self.decisions_made
            estimated_remaining_in_game = max(1, self.estimated_avg_decisions_per_game - decisions_in_current_game)
            estimated_decisions_remaining = estimated_remaining_in_game + (games_remaining - 1) * self.estimated_avg_decisions_per_game
        else:
            estimated_decisions_remaining = games_remaining * 25
        
        base_budget = safe_remaining / max(1, estimated_decisions_remaining)
        
        # 如果要调用VLM，预留时间
        if will_call_vlm:
            base_budget = max(0.5, base_budget - self.predicted_vlm_time)
        
        return max(1.0, min(20.0, base_budget))
    
    def should_call_vlm(self) -> bool:
        """判断是否应该调用VLM"""
        if self.start_time is None:
            return True
        
        # 规则：每3-5个决策调用一次，或关键时刻
        if self.decisions_made == 0:  # 每局第一个决策
            return True
        elif self.decisions_made % 4 == 0:  # 每4个决策
            return True
        else:
            return False
    
    def get_time_pressure(self):
        if self.start_time is None:
            return 0.0
        elapsed = time.time() - self.start_time
        return min(1.0, elapsed / self.total_time_budget)
    
    def start_new_game(self):
        if self.decisions_made > 0:
            self.games_completed += 1
            self.current_game += 1
            self.decisions_made = 0
        
        # 通知所有使用此time_manager的agents
        # (通过外部回调更新game_index)
        return self.current_game


vlm_time_manager = VLMTimeManager()


# ============ Timeout保护 ============
class SimulationTimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise SimulationTimeoutError()

def simulate_with_timeout(shot, timeout=2):
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)
    try:
        pt.simulate(shot, inplace=True)
        signal.alarm(0)
        return True
    except SimulationTimeoutError:
        return False
    except Exception:
        signal.alarm(0)
        return False
    finally:
        signal.signal(signal.SIGALRM, old_handler)


# ============ 评分函数 ============
def evaluate_shot_score(shot, last_state, player_targets):
    """评估shot的得分"""
    new_pocketed = [bid for bid, b in shot.balls.items() 
                   if b.state.s == 4 and last_state[bid].state.s != 4]
    
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed 
                     if bid not in player_targets and bid not in ["cue", "8"]]
    
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed
    
    # 基础得分
    score = 0
    
    if cue_pocketed:
        score -= 100
    if eight_pocketed:
        score += 100 if player_targets == ['8'] else -150
    
    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 25
    
    # 位置奖励
    if not cue_pocketed and 'cue' in shot.balls:
        cue_pos = shot.balls['cue'].state.rvw[0][:2]
        center_dist = np.linalg.norm(cue_pos - np.array([1.12, 0.56]))
        if center_dist < 0.7:
            score += 8
    
    return score


def evaluate_action_worker(args):
    """并行评估worker"""
    action, balls_state, table_state, last_state, player_targets = args
    
    try:
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls_state.items()}
        sim_table = copy.deepcopy(table_state)
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
        
        shot.cue.set_state(**action)
        
        if not simulate_with_timeout(shot, timeout=2):
            return (action, -50)
        
        score = evaluate_shot_score(shot, last_state, player_targets)
        return (action, score)
        
    except Exception:
        return (action, -500)


# ============ VLM引导的候选生成 ============
def generate_vlm_guided_candidates(balls, my_targets, table, vlm_guidance: Dict, 
                                   n_candidates=40) -> List[Dict]:
    """根据VLM指导生成候选动作"""
    
    strategy = vlm_guidance.get('strategy', 'conservative')
    target_priority = vlm_guidance.get('target_priority', my_targets)
    risk_tolerance = vlm_guidance.get('risk_tolerance', 0.5)
    
    candidates = []
    
    # 根据策略调整参数
    if strategy == 'aggressive':
        speed_range = (4.0, 7.0)  # 高速
        angle_variation = 10  # 大角度变化
    elif strategy == 'conservative':
        speed_range = (2.0, 5.0)  # 中低速
        angle_variation = 5  # 小角度变化
    elif strategy == 'defensive':
        speed_range = (1.5, 3.5)  # 低速
        angle_variation = 15  # 多样化
    else:  # positional
        speed_range = (2.5, 5.5)
        angle_variation = 7
    
    # 使用VLM推荐的目标球优先级
    cue_pos = balls['cue'].state.rvw[0][:2]
    
    for target_id in target_priority[:3]:  # 只考虑前3个优先目标
        if target_id not in balls or balls[target_id].state.s == 4:
            continue
        
        target_pos = balls[target_id].state.rvw[0][:2]
        
        # 计算基础方向
        dx = target_pos[0] - cue_pos[0]
        dy = target_pos[1] - cue_pos[1]
        base_phi = math.degrees(math.atan2(dy, dx)) % 360
        
        # 生成候选
        for v_offset in np.linspace(-1.0, 1.0, 5):
            V0 = np.clip(np.mean(speed_range) + v_offset, speed_range[0], speed_range[1])
            
            for phi_offset in range(-angle_variation, angle_variation + 1, angle_variation // 2):
                phi = (base_phi + phi_offset) % 360
                
                for theta in [0.0, 5.0]:
                    candidates.append({
                        'V0': V0,
                        'phi': phi,
                        'theta': theta,
                        'a': 0.0,
                        'b': 0.0
                    })
                    
                    if len(candidates) >= n_candidates:
                        return candidates
    
    # 如果候选不够，添加随机候选
    while len(candidates) < n_candidates:
        candidates.append({
            'V0': random.uniform(speed_range[0], speed_range[1]),
            'phi': random.uniform(0, 360),
            'theta': random.uniform(0, 15),
            'a': 0.0,
            'b': 0.0
        })
    
    return candidates[:n_candidates]


# ============ VLM辅助Agent ============
class VLMAssistedAgent:
    """VLM引导的MCTS台球agent"""
    
    def __init__(self, vlm_provider='openai', vlm_model='gpt-4-vision-preview',
                 vlm_base_url=None, use_vlm=True, n_cores=None):
        """
        初始化VLM辅助agent
        
        Args:
            vlm_provider: VLM提供商 ('openai', 'claude', 'qwen')
            vlm_model: 模型名称
            vlm_base_url: API基础URL（用于Qwen等兼容OpenAI的服务）
            use_vlm: 是否启用VLM（False则降级到纯启发式）
            n_cores: CPU核心数
        """
        # CPU核心
        if n_cores is None:
            try:
                n_cores = len(os.sched_getaffinity(0))
            except:
                n_cores = os.cpu_count() or 8
        self.n_cores = min(n_cores, 16)
        
        # VLM组件
        self.use_vlm = use_vlm
        if use_vlm:
            self.drawer = BilliardsDrawer()
            self.vlm_chat = VLMChat(
                provider=vlm_provider, 
                model=vlm_model,
                base_url=vlm_base_url
            )
            print(f"[VLMAgent] VLM enabled: {vlm_provider}/{vlm_model}")
            if vlm_base_url:
                print(f"[VLMAgent] Base URL: {vlm_base_url}")
        else:
            self.drawer = None
            self.vlm_chat = None
            print("[VLMAgent] VLM disabled, using heuristic mode")
        
        # 搜索参数
        self.MIN_CANDIDATES = 15
        self.MAX_CANDIDATES = 60
        
        # BO参数
        self.pbounds = {
            'V0': (0.5, 8.0),
            'phi': (0, 360),
            'theta': (0, 90),
            'a': (-0.5, 0.5),
            'b': (-0.5, 0.5)
        }
        
        self.time_manager = vlm_time_manager
        self.executor = None
        
        # 缓存最近的VLM指导
        self.last_vlm_guidance = None
        self.decisions_since_last_vlm = 0
        
        # 日志记录器
        self.logger = None
        self.current_game_index = 0
        self.current_shot_index = 0
        
        print(f"[VLMAgent] Initialized with {self.n_cores} cores")
    
    def __del__(self):
        if self.executor is not None:
            self.executor.shutdown(wait=False)
    
    def set_logger(self, logger):
        """设置日志记录器"""
        self.logger = logger
        print(f"[VLMAgent] Logger configured")
    
    def start_new_game(self):
        """通知agent开始新游戏（用于更新日志索引）"""
        self.current_game_index += 1
        self.current_shot_index = 0
        if self.time_manager and hasattr(self.time_manager, 'start_new_game'):
            self.time_manager.start_new_game()
    
    def _random_action(self):
        return {
            'V0': random.uniform(2.0, 6.0),
            'phi': random.uniform(0, 360),
            'theta': random.uniform(0, 15),
            'a': 0.0,
            'b': 0.0
        }
    
    def _get_vlm_guidance(self, balls, my_targets, enemy_targets, 
                         my_remaining, enemy_remaining, table=None) -> Dict:
        """获取VLM战略指导"""
        
        if not self.use_vlm or self.vlm_chat.client is None:
            # 降级到启发式
            return self._heuristic_guidance(my_remaining, enemy_remaining, my_targets)
        
        try:
            vlm_start = time.time()
            
            # 生成图片（传递table对象以获取正确尺寸）
            image = self.drawer.draw_table_state(
                balls, my_targets, enemy_targets,
                title=f"Game State - My: {my_remaining} vs Enemy: {enemy_remaining}",
                table=table
            )
            
            # 调用VLM（获取原始响应用于日志）
            guidance, prompt_text, raw_response = self.vlm_chat.get_strategy_from_image(
                image, my_remaining, enemy_remaining, my_targets,
                game_phase='end' if my_remaining <= 2 else 'mid',
                return_raw_response=True
            )
            
            vlm_time = time.time() - vlm_start
            self.time_manager.record_vlm_call(vlm_time)
            
            print(f"[VLMAgent] VLM guidance: {guidance['strategy']}, "
                  f"risk={guidance['risk_tolerance']:.2f}, time={vlm_time:.2f}s")
            
            # 记录VLM调用到日志
            if self.logger is not None:
                # 保存图片
                log_dirs = self.logger.get_vlm_log_dir()
                if log_dirs:
                    image_filename = f"game{self.current_game_index}_shot{self.current_shot_index}.png"
                    image_path = os.path.join(log_dirs['images_dir'], image_filename)
                    image.save(image_path)
                    
                    # 记录到logger
                    self.logger.log_vlm_call(
                        game_index=self.current_game_index,
                        shot_index=self.current_shot_index,
                        prompt_text=prompt_text,
                        image_path=image_path,
                        response_text=raw_response,
                        strategy=guidance
                    )
            
            return guidance
            
        except Exception as e:
            print(f"[VLMAgent] VLM error: {e}, falling back to heuristics")
            import traceback
            traceback.print_exc()
            return self._heuristic_guidance(my_remaining, enemy_remaining, my_targets)
    
    def _heuristic_guidance(self, my_remaining, enemy_remaining, my_targets) -> Dict:
        """启发式指导（VLM降级）"""
        if my_remaining < enemy_remaining:
            strategy = 'conservative'
            risk = 0.3
        elif my_remaining == enemy_remaining:
            strategy = 'balanced'
            risk = 0.5
        else:
            strategy = 'aggressive'
            risk = 0.7
        
        return {
            'strategy': strategy,
            'target_priority': my_targets,
            'risk_tolerance': risk,
            'reasoning': 'Heuristic fallback',
            'key_considerations': []
        }
    
    def _evaluate_candidates_parallel(self, candidates, balls, table, 
                                     last_state, my_targets, timeout):
        """并行评估候选"""
        if self.executor is None:
            self.executor = ProcessPoolExecutor(max_workers=self.n_cores)
        
        eval_args = [(c, balls, table, last_state, my_targets) for c in candidates]
        
        try:
            futures = [self.executor.submit(evaluate_action_worker, arg) 
                      for arg in eval_args]
            
            results = []
            for future in futures:
                try:
                    result = future.result(timeout=timeout/len(candidates) + 1.0)
                    results.append(result)
                except:
                    results.append((None, -100))
            
            return results
        except Exception as e:
            print(f"[VLMAgent] Parallel eval error: {e}")
            return [(c, -500) for c in candidates]
    
    def decision(self, balls=None, my_targets=None, table=None):
        """
        主决策函数
        流程：环境读取 → 图片生成 → VLM推理 → VLM-Guided搜索 → 参数优化
        """
        decision_start = time.time()
        
        # 更新shot索引（用于日志）
        self.current_shot_index += 1
        
        if balls is None:
            return self._random_action()
        
        try:
            # 计算剩余球数
            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                my_targets = ["8"]
            
            my_remaining = len(remaining_own)
            all_balls = set(str(i) for i in range(1, 16))
            enemy_targets = list(all_balls - set(my_targets) - {'8'})
            enemy_remaining = len([bid for bid in enemy_targets 
                                  if bid in balls and balls[bid].state.s != 4])
            
            # 判断是否调用VLM
            should_call_vlm = self.time_manager.should_call_vlm()
            time_budget = self.time_manager.get_time_budget(
                {'n_remaining_balls': my_remaining},
                will_call_vlm=should_call_vlm
            )
            
            print(f"\n[VLMAgent] Decision #{self.time_manager.decisions_made + 1}")
            print(f"  Situation: My {my_remaining} vs Enemy {enemy_remaining}")
            print(f"  Time budget: {time_budget:.1f}s, Will call VLM: {should_call_vlm}")
            
            # Step 1: 获取VLM指导
            if should_call_vlm or self.last_vlm_guidance is None:
                vlm_guidance = self._get_vlm_guidance(
                    balls, my_targets, enemy_targets, 
                    my_remaining, enemy_remaining,
                    table=table
                )
                self.last_vlm_guidance = vlm_guidance
                self.decisions_since_last_vlm = 0
            else:
                vlm_guidance = self.last_vlm_guidance
                self.decisions_since_last_vlm += 1
                print(f"[VLMAgent] Reusing VLM guidance (age: {self.decisions_since_last_vlm})")
            
            # Step 2: VLM引导的候选生成
            n_candidates = min(self.MAX_CANDIDATES, 
                             int(self.MAX_CANDIDATES * (time_budget / 10.0)))
            n_candidates = max(self.MIN_CANDIDATES, n_candidates)
            
            candidates = generate_vlm_guided_candidates(
                balls, my_targets, table, vlm_guidance, n_candidates
            )
            
            print(f"[VLMAgent] Generated {len(candidates)} VLM-guided candidates")
            
            # Step 3: 并行评估
            last_state = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            eval_timeout = time_budget * 0.5
            
            results = self._evaluate_candidates_parallel(
                candidates, balls, table, last_state, my_targets, eval_timeout
            )
            
            # 找到最佳候选
            best_candidate = None
            best_score = -float('inf')
            for action, score in results:
                if action is not None and score > best_score:
                    best_score = score
                    best_candidate = action
            
            print(f"[VLMAgent] Best candidate score: {best_score:.1f}")
            
            # Step 4: Bayesian Optimization精细优化（如果时间允许）
            remaining_time = time_budget - (time.time() - decision_start)
            
            if remaining_time > 2.0 and HAS_BAYES_OPT and best_score < 80:
                print(f"[VLMAgent] Running BO refinement ({remaining_time:.1f}s available)")
                
                def reward_fn(V0, phi, theta, a, b):
                    action = {'V0': V0, 'phi': phi, 'theta': theta, 'a': a, 'b': b}
                    results = self._evaluate_candidates_parallel(
                        [action], balls, table, last_state, my_targets, 2.0
                    )
                    return results[0][1] if results else -500
                
                optimizer = BayesianOptimization(
                    f=reward_fn,
                    pbounds=self.pbounds,
                    random_state=np.random.randint(1e6),
                    verbose=0
                )
                
                # 用最佳候选初始化
                if best_candidate:
                    try:
                        optimizer.probe(params=best_candidate, lazy=True)
                    except:
                        pass
                
                n_init = min(8, int(remaining_time * 1.5))
                n_iter = min(10, int(remaining_time * 1.2))
                
                optimizer.maximize(init_points=n_init, n_iter=n_iter)
                
                bo_best = optimizer.max
                if bo_best['target'] > best_score:
                    action = {k: float(v) for k, v in bo_best['params'].items()}
                    print(f"[VLMAgent] BO improved score: {best_score:.1f} → {bo_best['target']:.1f}")
                else:
                    action = best_candidate
            else:
                action = best_candidate if best_score >= 10 else self._random_action()
            
            # 记录时间
            decision_time = time.time() - decision_start
            self.time_manager.learn_from_decision(decision_time)
            
            print(f"[VLMAgent] Decision time: {decision_time:.2f}s")
            print(f"  Action: V0={action['V0']:.2f}, phi={action['phi']:.1f}°\n")
            
            return action
        
        except Exception as e:
            print(f"[VLMAgent] Error: {e}")
            import traceback
            traceback.print_exc()
            
            decision_time = time.time() - decision_start
            self.time_manager.learn_from_decision(decision_time)
            
            return self._random_action()

