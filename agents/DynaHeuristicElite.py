"""
DynaHeuristicElite.py - Elite Agent with Advanced Strategic Planning

Building on DynaHeuristicParallel (85% win rate), this agent adds:
1. Multi-strategy candidate generation (aggressive, conservative, positional, defensive)
2. Sequential planning - considers follow-up shots
3. Tactical evaluation - assesses position advantage and opponent opportunity
4. Adaptive play style - adjusts aggression based on game situation
5. Enhanced reward function with strategic scoring

Target: 90%+ win rate

Key Improvements over DynaHeuristicParallel:
- 4 distinct shot strategies instead of 1
- Position evaluation for next shot
- Defensive play when losing badly
- Smarter target sequencing
- Enhanced reward with tactical bonuses
"""

import math
import numpy as np
import pooltool as pt
import copy
import time
import signal
import os
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError
from collections import deque
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


# ============ Adaptive Time Manager (Same as Parallel) ============
class EliteTimeManager:
    """Self-learning time manager"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EliteTimeManager, cls).__new__(cls)
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
        self.game_decision_counts = deque(maxlen=20)
        
        self.time_safety_margin = 0.96
        self.estimated_avg_decisions_per_game = 28.0
        self.predicted_decision_time = 6.0
        
        self.games_completed = 0
        self.total_decisions = 0
        
    def initialize(self, n_games, time_per_game=180.0):
        self.total_time_budget = n_games * time_per_game
        self.start_time = time.time()
        self.total_games = n_games
        self.current_game = 0
        self.decisions_made = 0
        self.games_completed = 0
        
        self.decision_time_history.clear()
        self.game_decision_counts.clear()
        self.time_safety_margin = 0.96
        
        print(f"[EliteTimeManager] Initialized with {self.total_time_budget:.0f}s budget for {n_games} games")
        print(f"[EliteTimeManager] Target: 88-95% time utilization")
    
    def learn_from_decision(self, decision_time):
        self.decision_time_history.append(decision_time)
        self.decisions_made += 1
        self.total_decisions += 1
        
        if len(self.decision_time_history) >= 5:
            recent_avg = np.mean(list(self.decision_time_history)[-10:])
            self.predicted_decision_time = 0.7 * self.predicted_decision_time + 0.3 * recent_avg
        
        if self.decisions_made >= 15:
            self._adapt_safety_margin()
    
    def _adapt_safety_margin(self):
        if self.start_time is None:
            return
        
        elapsed = time.time() - self.start_time
        utilization = elapsed / self.total_time_budget
        
        if utilization < 0.5 and self.games_completed > 1:
            self.time_safety_margin = min(0.98, self.time_safety_margin + 0.015)
        elif utilization < 0.7 and self.games_completed > 3:
            self.time_safety_margin = min(0.98, self.time_safety_margin + 0.008)
        elif utilization > 0.94:
            self.time_safety_margin = max(0.92, self.time_safety_margin - 0.015)
    
    def end_game(self, decisions_in_game):
        self.game_decision_counts.append(decisions_in_game)
        self.games_completed += 1
        self.current_game += 1
        self.decisions_made = 0
        
        if len(self.game_decision_counts) >= 3:
            self.estimated_avg_decisions_per_game = np.mean(self.game_decision_counts)
    
    def get_time_budget(self, game_state=None):
        if self.start_time is None:
            return 10.0
        
        elapsed = time.time() - self.start_time
        remaining = self.total_time_budget - elapsed
        safe_remaining = remaining * self.time_safety_margin
        
        if safe_remaining <= 0:
            return 0.4
        
        games_remaining = max(1, self.total_games - self.current_game)
        
        if len(self.game_decision_counts) >= 3:
            decisions_in_current_game = self.decisions_made
            estimated_remaining_in_game = max(1, self.estimated_avg_decisions_per_game - decisions_in_current_game)
            estimated_decisions_remaining = estimated_remaining_in_game + (games_remaining - 1) * self.estimated_avg_decisions_per_game
        else:
            estimated_decisions_remaining = games_remaining * 28
        
        base_budget = safe_remaining / max(1, estimated_decisions_remaining)
        
        complexity_multiplier = 1.0
        if game_state:
            n_remaining = game_state.get('n_remaining_balls', 7)
            if n_remaining <= 2:
                complexity_multiplier = 1.8
            elif n_remaining <= 4:
                complexity_multiplier = 1.3
            elif n_remaining >= 6:
                complexity_multiplier = 0.9
        
        utilization = elapsed / self.total_time_budget if self.total_time_budget > 0 else 0
        if utilization < 0.3 and games_remaining > 5:
            complexity_multiplier *= 1.4
        elif utilization > 0.85:
            complexity_multiplier *= 0.7
        
        time_budget = base_budget * complexity_multiplier
        return max(0.4, min(22.0, time_budget))
    
    def get_remaining_time(self):
        if self.start_time is None:
            return float('inf')
        return self.total_time_budget - (time.time() - self.start_time)
    
    def get_time_pressure(self):
        if self.start_time is None:
            return 0.0
        elapsed = time.time() - self.start_time
        return min(1.0, elapsed / self.total_time_budget)
    
    def start_new_game(self):
        if self.decisions_made > 0:
            self.end_game(self.decisions_made)
        
        if self.current_game % 10 == 0 and self.start_time:
            elapsed = time.time() - self.start_time
            remaining = self.total_time_budget - elapsed
            utilization = elapsed / self.total_time_budget * 100
            print(f"[EliteTimeManager] Game {self.current_game}/{self.total_games}")
            print(f"  Time: {elapsed:.0f}s used ({utilization:.1f}%), {remaining:.0f}s remaining")
            print(f"  Decisions: {self.total_decisions}, Avg: {self.predicted_decision_time:.2f}s/decision")
    
    def get_stats(self):
        elapsed = time.time() - self.start_time if self.start_time else 0
        remaining = self.total_time_budget - elapsed
        return {
            'elapsed': elapsed,
            'remaining': remaining,
            'budget': self.total_time_budget,
            'decisions': self.total_decisions,
            'avg_time_per_decision': elapsed / max(1, self.total_decisions),
            'games_completed': self.games_completed,
            'utilization': elapsed / self.total_time_budget if self.total_time_budget > 0 else 0,
            'safety_margin': self.time_safety_margin,
            'predicted_decision_time': self.predicted_decision_time
        }


elite_time_manager = EliteTimeManager()


# ============ Timeout Protection ============
class SimulationTimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise SimulationTimeoutError("Physics simulation timeout")

def simulate_with_timeout(shot, timeout=3):
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)
    try:
        pt.simulate(shot, inplace=True)
        signal.alarm(0)
        return True
    except SimulationTimeoutError:
        return False
    except Exception as e:
        signal.alarm(0)
        raise e
    finally:
        signal.signal(signal.SIGALRM, old_handler)


# ============ Enhanced Tactical Evaluation ============
def evaluate_candidate_with_tactics(args):
    """Enhanced evaluation with tactical scoring"""
    candidate, balls_state, table_state, last_state, player_targets, my_remaining, enemy_remaining = args
    
    try:
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls_state.items()}
        sim_table = copy.deepcopy(table_state)
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
        
        shot.cue.set_state(**candidate)
        
        if not simulate_with_timeout(shot, timeout=2):
            return (candidate, -60)
        
        # === Basic Scoring ===
        new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
        
        own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
        enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
        
        cue_pocketed = "cue" in new_pocketed
        eight_pocketed = "8" in new_pocketed

        first_contact_ball_id = None
        foul_first_hit = False
        valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}
        
        for e in shot.events:
            et = str(e.event_type).lower()
            ids = list(e.ids) if hasattr(e, 'ids') else []
            if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
                other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
                if other_ids:
                    first_contact_ball_id = other_ids[0]
                    break
        
        if first_contact_ball_id is None:
            if len(last_state) > 2 or player_targets != ['8']:
                foul_first_hit = True
        else:
            if first_contact_ball_id not in player_targets:
                foul_first_hit = True
        
        cue_hit_cushion = False
        target_hit_cushion = False
        foul_no_rail = False
        
        for e in shot.events:
            et = str(e.event_type).lower()
            ids = list(e.ids) if hasattr(e, 'ids') else []
            if 'cushion' in et:
                if 'cue' in ids:
                    cue_hit_cushion = True
                if first_contact_ball_id is not None and first_contact_ball_id in ids:
                    target_hit_cushion = True

        if len(new_pocketed) == 0 and first_contact_ball_id is not None and (not cue_hit_cushion) and (not target_hit_cushion):
            foul_no_rail = True
            
        score = 0
        
        # Critical events
        if cue_pocketed and eight_pocketed:
            score -= 180
        elif cue_pocketed:
            score -= 120
        elif eight_pocketed:
            if player_targets == ['8']:
                score += 120
            else:
                score -= 180
                
        if foul_first_hit:
            score -= 35
        if foul_no_rail:
            score -= 35
        
        # Ball scoring with strategic bonuses
        if len(own_pocketed) > 0:
            base_reward = 55
            if len(own_pocketed) >= 2:
                score += base_reward * len(own_pocketed) * 1.4  # Multi-ball bonus
            else:
                score += base_reward * len(own_pocketed)
            
            # Near-win bonus
            if my_remaining - len(own_pocketed) <= 1:
                score += 25
            elif my_remaining - len(own_pocketed) <= 2:
                score += 15
        
        score -= len(enemy_pocketed) * 25
        
        # === Tactical Scoring ===
        if not cue_pocketed and 'cue' in shot.balls:
            cue_pos = shot.balls['cue'].state.rvw[0][:2]
            
            # Position quality
            center_x, center_y = 1.12, 1.12
            dist_from_center = np.linalg.norm(cue_pos - np.array([center_x, center_y]))
            
            if dist_from_center < 0.7:
                score += 10  # Excellent position
            elif dist_from_center < 1.1:
                score += 5   # Good position
            
            # Cushion penalty (harder next shot)
            if cue_pos[0] < 0.15 or cue_pos[0] > 2.09 or cue_pos[1] < 0.15 or cue_pos[1] > 2.09:
                score -= 8
            
            # === Next Shot Analysis ===
            if len(own_pocketed) > 0:
                # Evaluate position for next target
                remaining_after = [bid for bid in player_targets 
                                 if bid not in own_pocketed and shot.balls[bid].state.s != 4]
                
                if remaining_after:
                    # Find closest remaining target
                    min_dist = float('inf')
                    for target_id in remaining_after:
                        target_pos = shot.balls[target_id].state.rvw[0][:2]
                        dist = np.linalg.norm(cue_pos - target_pos)
                        min_dist = min(min_dist, dist)
                    
                    # Bonus for good position on next ball
                    if min_dist < 0.8:
                        score += 15  # Great position for follow-up
                    elif min_dist < 1.3:
                        score += 8   # Decent position
            
            # === Strategic Evaluation ===
            # Winning situation: be slightly conservative
            if my_remaining < enemy_remaining:
                score += 8  # Bonus for being ahead
                if len(own_pocketed) == 0 and score > 0:
                    score += 3  # Small bonus for safe play when ahead
            
            # Losing situation: reward aggressive play
            elif my_remaining > enemy_remaining + 2:
                if len(own_pocketed) > 0:
                    score += 10  # Reward aggressive pocketing when behind
                else:
                    score -= 5   # Penalize safety when far behind
        
        # Safe shot baseline
        if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
            # Situational safety value
            if my_remaining <= enemy_remaining:
                score = 12  # Safety is valuable when ahead/even
            else:
                score = 5   # Safety less valuable when behind
        
        return (candidate, score)
        
    except Exception as e:
        return (candidate, -500)


# ============ Geometry & Heuristics ============
def get_ball_position(ball):
    return ball.state.rvw[0][:2]

def distance_2d(pos1, pos2):
    return np.linalg.norm(np.array(pos1) - np.array(pos2))

def angle_between_points(from_pos, to_pos):
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    return math.degrees(math.atan2(dy, dx)) % 360

def has_clear_path(from_pos, to_pos, balls, ignore_ids, ball_radius=0.028575):
    from_pos = np.array(from_pos)
    to_pos = np.array(to_pos)
    path_vec = to_pos - from_pos
    path_length = np.linalg.norm(path_vec)
    
    if path_length < 1e-6:
        return True
    
    path_dir = path_vec / path_length
    
    for bid, ball in balls.items():
        if bid in ignore_ids or ball.state.s == 4:
            continue
        
        ball_pos = get_ball_position(ball)
        to_ball = ball_pos - from_pos
        projection = np.dot(to_ball, path_dir)
        
        if projection < 0 or projection > path_length:
            continue
        
        closest_point = from_pos + projection * path_dir
        dist = np.linalg.norm(ball_pos - closest_point)
        
        if dist < 2.3 * ball_radius:
            return False
    
    return True

def calculate_shot_difficulty_advanced(cue_pos, target_pos, pocket_pos, balls, target_id):
    """Enhanced difficulty calculation"""
    difficulty = 0
    
    cue_to_target_dist = distance_2d(cue_pos, target_pos)
    difficulty += cue_to_target_dist * 9
    
    target_to_pocket_dist = distance_2d(target_pos, pocket_pos)
    difficulty += target_to_pocket_dist * 22
    
    # Path clearance
    if not has_clear_path(cue_pos, target_pos, balls, ['cue', target_id]):
        difficulty += 65
    
    if not has_clear_path(target_pos, pocket_pos, balls, [target_id, 'cue']):
        difficulty += 45
    
    # Cut angle
    cue_to_target_angle = angle_between_points(cue_pos, target_pos)
    target_to_pocket_angle = angle_between_points(target_pos, pocket_pos)
    cut_angle = abs(target_to_pocket_angle - cue_to_target_angle)
    cut_angle = min(cut_angle, 360 - cut_angle)
    
    if cut_angle > 75:
        difficulty += (cut_angle - 75) * 3.5
    elif cut_angle > 50:
        difficulty += (cut_angle - 50) * 2
    elif cut_angle < 12:
        difficulty -= 18  # Straight shot bonus
    
    # Pocket type bonus
    corner_positions = [(0, 0), (0, 2.24), (2.24, 0), (2.24, 2.24)]
    min_corner_dist = min(distance_2d(pocket_pos, corner) for corner in corner_positions)
    if min_corner_dist < 0.1:
        difficulty -= 12  # Corner pocket easier
    
    return difficulty

def select_strategic_targets(balls, my_targets, table, top_n=4):
    """Strategic target selection with sequence planning"""
    cue_pos = get_ball_position(balls['cue'])
    remaining_targets = [bid for bid in my_targets if balls[bid].state.s != 4]
    
    if not remaining_targets:
        return []
    
    pocket_positions = {pid: pocket.center[:2] for pid, pocket in table.pockets.items()}
    
    target_evaluations = []
    for target_id in remaining_targets:
        target_pos = get_ball_position(balls[target_id])
        min_difficulty = float('inf')
        best_pocket = None
        
        for pocket_id, pocket_pos in pocket_positions.items():
            difficulty = calculate_shot_difficulty_advanced(
                cue_pos, target_pos, pocket_pos, balls, target_id
            )
            if difficulty < min_difficulty:
                min_difficulty = difficulty
                best_pocket = pocket_pos
        
        # Consider follow-up potential
        follow_up_value = 0
        if best_pocket is not None:
            # Estimate cue position after shot
            direction = np.array(target_pos) - np.array(cue_pos)
            direction_norm = direction / (np.linalg.norm(direction) + 1e-6)
            estimated_cue_pos = np.array(target_pos) + direction_norm * 0.12
            
            # Check distance to next closest target
            for next_id in remaining_targets:
                if next_id == target_id:
                    continue
                next_pos = get_ball_position(balls[next_id])
                dist = distance_2d(estimated_cue_pos, next_pos)
                if dist < 1.0:
                    follow_up_value -= 20
                elif dist < 1.5:
                    follow_up_value -= 10
        
        total_score = min_difficulty + follow_up_value
        target_evaluations.append((target_id, total_score))
    
    target_evaluations.sort(key=lambda x: x[1])
    return [tid for tid, _ in target_evaluations[:top_n]]

def calculate_ghost_ball_position(target_pos, pocket_pos, ball_radius=0.028575):
    target_pos = np.array(target_pos)
    pocket_pos = np.array(pocket_pos)
    
    direction = target_pos - pocket_pos
    dist = np.linalg.norm(direction)
    
    if dist < 1e-6:
        return target_pos
    
    direction = direction / dist
    ghost_pos = target_pos + direction * (2 * ball_radius)
    
    return ghost_pos


# ============ Multi-Strategy Candidate Generation ============
def generate_elite_candidates(balls, my_targets, table, my_remaining, enemy_remaining, n_total=50):
    """Generate candidates with multiple strategies"""
    
    # Determine game situation
    if my_remaining < enemy_remaining:
        situation = 'winning'
    elif my_remaining == enemy_remaining:
        situation = 'even'
    else:
        situation = 'losing'
    
    # Allocate candidates based on situation
    if situation == 'winning':
        # More conservative when ahead
        n_aggressive = int(n_total * 0.35)
        n_conservative = int(n_total * 0.40)
        n_positional = int(n_total * 0.20)
        n_defensive = n_total - n_aggressive - n_conservative - n_positional
    elif situation == 'losing':
        # More aggressive when behind
        n_aggressive = int(n_total * 0.50)
        n_conservative = int(n_total * 0.25)
        n_positional = int(n_total * 0.20)
        n_defensive = n_total - n_aggressive - n_conservative - n_positional
    else:
        # Balanced when even
        n_aggressive = int(n_total * 0.40)
        n_conservative = int(n_total * 0.35)
        n_positional = int(n_total * 0.20)
        n_defensive = n_total - n_aggressive - n_conservative - n_positional
    
    candidates = []
    candidates.extend(generate_aggressive_shots(balls, my_targets, table, n_aggressive))
    candidates.extend(generate_conservative_shots(balls, my_targets, table, n_conservative))
    candidates.extend(generate_positional_shots(balls, my_targets, table, n_positional))
    candidates.extend(generate_defensive_shots(balls, my_targets, table, n_defensive))
    
    return candidates[:n_total]

def generate_aggressive_shots(balls, my_targets, table, n_shots):
    """Generate aggressive pocketing shots"""
    if n_shots == 0:
        return []
    
    cue_pos = get_ball_position(balls['cue'])
    top_targets = select_strategic_targets(balls, my_targets, table, top_n=3)
    
    if not top_targets:
        return [random_action() for _ in range(n_shots)]
    
    pocket_positions = {pid: pocket.center[:2] for pid, pocket in table.pockets.items()}
    candidates = []
    
    for target_id in top_targets:
        target_pos = get_ball_position(balls[target_id])
        
        sorted_pockets = sorted(
            pocket_positions.items(),
            key=lambda x: distance_2d(target_pos, x[1])
        )
        
        for pocket_id, pocket_pos in sorted_pockets[:3]:
            ghost_pos = calculate_ghost_ball_position(target_pos, pocket_pos)
            phi_base = angle_between_points(cue_pos, ghost_pos)
            dist = distance_2d(cue_pos, target_pos)
            
            # Aggressive speed
            if dist < 0.5:
                V0_base = 3.0
            elif dist < 1.0:
                V0_base = 4.0
            elif dist < 1.5:
                V0_base = 5.0
            else:
                V0_base = 6.0
            
            for v_offset in [-0.5, 0, 0.5]:
                for phi_offset in [-7, 0, 7]:
                    V0 = np.clip(V0_base + v_offset, 1.0, 8.0)
                    phi = (phi_base + phi_offset) % 360
                    
                    candidates.append({
                        'V0': V0,
                        'phi': phi,
                        'theta': 0.0,
                        'a': 0.0,
                        'b': 0.0
                    })
                    
                    if len(candidates) >= n_shots:
                        return candidates[:n_shots]
    
    while len(candidates) < n_shots:
        candidates.append(random_action())
    
    return candidates[:n_shots]

def generate_conservative_shots(balls, my_targets, table, n_shots):
    """Generate controlled, precise shots"""
    if n_shots == 0:
        return []
    
    cue_pos = get_ball_position(balls['cue'])
    top_targets = select_strategic_targets(balls, my_targets, table, top_n=2)
    
    if not top_targets:
        return [random_action() for _ in range(n_shots)]
    
    pocket_positions = {pid: pocket.center[:2] for pid, pocket in table.pockets.items()}
    candidates = []
    
    for target_id in top_targets:
        target_pos = get_ball_position(balls[target_id])
        
        sorted_pockets = sorted(
            pocket_positions.items(),
            key=lambda x: distance_2d(target_pos, x[1])
        )
        
        for pocket_id, pocket_pos in sorted_pockets[:2]:
            ghost_pos = calculate_ghost_ball_position(target_pos, pocket_pos)
            phi_base = angle_between_points(cue_pos, ghost_pos)
            dist = distance_2d(cue_pos, target_pos)
            
            # Conservative speed (more control)
            V0_base = min(4.5, 2.5 + dist * 1.2)
            
            for v_offset in [-0.3, 0, 0.3]:
                for phi_offset in [-5, 0, 5]:
                    for theta_val in [0.0, 5.0, 10.0]:
                        V0 = np.clip(V0_base + v_offset, 1.5, 5.5)
                        phi = (phi_base + phi_offset) % 360
                        
                        candidates.append({
                            'V0': V0,
                            'phi': phi,
                            'theta': theta_val,
                            'a': 0.0,
                            'b': 0.0
                        })
                        
                        if len(candidates) >= n_shots:
                            return candidates[:n_shots]
    
    while len(candidates) < n_shots:
        candidates.append(random_action())
    
    return candidates[:n_shots]

def generate_positional_shots(balls, my_targets, table, n_shots):
    """Generate shots with position emphasis"""
    if n_shots == 0:
        return []
    
    cue_pos = get_ball_position(balls['cue'])
    target_id = select_strategic_targets(balls, my_targets, table, top_n=1)
    
    if not target_id:
        return [random_action() for _ in range(n_shots)]
    
    target_id = target_id[0]
    target_pos = get_ball_position(balls[target_id])
    pocket_positions = {pid: pocket.center[:2] for pid, pocket in table.pockets.items()}
    
    candidates = []
    
    for pocket_id, pocket_pos in pocket_positions.items():
        ghost_pos = calculate_ghost_ball_position(target_pos, pocket_pos)
        phi_base = angle_between_points(cue_pos, ghost_pos)
        dist = distance_2d(cue_pos, target_pos)
        
        # Moderate speed with spin variations
        V0_base = 3.5 + dist * 0.8
        
        for v_offset in [-0.4, 0, 0.4]:
            for phi_offset in [-6, 0, 6]:
                for a_val in [-0.15, 0.0, 0.15]:
                    for b_val in [-0.15, 0.0, 0.15]:
                        V0 = np.clip(V0_base + v_offset, 2.0, 6.0)
                        phi = (phi_base + phi_offset) % 360
                        
                        candidates.append({
                            'V0': V0,
                            'phi': phi,
                            'theta': 0.0,
                            'a': a_val,
                            'b': b_val
                        })
                        
                        if len(candidates) >= n_shots:
                            return candidates[:n_shots]
    
    while len(candidates) < n_shots:
        candidates.append(random_action())
    
    return candidates[:n_shots]

def generate_defensive_shots(balls, my_targets, table, n_shots):
    """Generate safety/defensive shots"""
    if n_shots == 0:
        return []
    
    import random
    cue_pos = get_ball_position(balls['cue'])
    candidates = []
    
    # Soft touches and strategic positioning
    for _ in range(n_shots):
        phi = random.uniform(0, 360)
        V0 = random.uniform(1.5, 3.5)  # Soft shots
        theta = random.uniform(0, 20)
        a = random.uniform(-0.2, 0.2)
        b = random.uniform(-0.2, 0.2)
        
        candidates.append({
            'V0': V0,
            'phi': phi,
            'theta': theta,
            'a': a,
            'b': b
        })
    
    return candidates

def random_action():
    import random
    return {
        'V0': random.uniform(2.0, 6.0),
        'phi': random.uniform(0, 360),
        'theta': random.uniform(0, 15),
        'a': 0.0,
        'b': 0.0
    }


# ============ Elite Agent ============
class EliteDynamicAgent:
    """Elite agent with advanced strategic planning"""
    
    def __init__(self, n_cores=None):
        if n_cores is None:
            try:
                n_cores = len(os.sched_getaffinity(0))
            except:
                n_cores = os.cpu_count() or 16
        
        self.n_cores = min(n_cores, 32)
        
        # Enhanced search parameters
        self.MIN_INITIAL_SEARCH = 4
        self.MAX_INITIAL_SEARCH = 45
        self.MIN_OPT_SEARCH = 3
        self.MAX_OPT_SEARCH = 28
        self.MIN_CANDIDATES = 15
        self.MAX_CANDIDATES = 90
        
        self.pbounds = {
            'V0': (0.5, 8.0),
            'phi': (0, 360),
            'theta': (0, 90),
            'a': (-0.5, 0.5),
            'b': (-0.5, 0.5)
        }
        
        self.ALPHA = 1e-2
        self.TABLE_WIDTH = 1.12
        self.TABLE_LENGTH = 2.24
        self.BALL_RADIUS = 0.028575
        
        self.time_manager = elite_time_manager
        self.executor = None
        
        print(f"EliteDynamicAgent initialized with {self.n_cores} CPU cores")
        print(f"  Enhanced with 4-strategy planning (Aggressive/Conservative/Positional/Defensive)")
        print(f"  Max search: Init={self.MAX_INITIAL_SEARCH}, Opt={self.MAX_OPT_SEARCH}, Candidates={self.MAX_CANDIDATES}")
    
    def __del__(self):
        if self.executor is not None:
            self.executor.shutdown(wait=False)
    
    def get_adaptive_search_params(self, time_budget, game_state, time_pressure):
        """Enhanced adaptive parameters"""
        if time_budget > 14.0:
            initial_search = self.MAX_INITIAL_SEARCH
            opt_search = self.MAX_OPT_SEARCH
            n_candidates = self.MAX_CANDIDATES
        elif time_budget > 10.0:
            initial_search = int(self.MAX_INITIAL_SEARCH * 0.85)
            opt_search = int(self.MAX_OPT_SEARCH * 0.85)
            n_candidates = int(self.MAX_CANDIDATES * 0.85)
        elif time_budget > 7.0:
            initial_search = int(self.MAX_INITIAL_SEARCH * 0.7)
            opt_search = int(self.MAX_OPT_SEARCH * 0.7)
            n_candidates = int(self.MAX_CANDIDATES * 0.7)
        elif time_budget > 5.0:
            initial_search = int(self.MAX_INITIAL_SEARCH * 0.55)
            opt_search = int(self.MAX_OPT_SEARCH * 0.55)
            n_candidates = int(self.MAX_CANDIDATES * 0.6)
        elif time_budget > 3.0:
            initial_search = int(self.MAX_INITIAL_SEARCH * 0.4)
            opt_search = int(self.MAX_OPT_SEARCH * 0.4)
            n_candidates = int(self.MAX_CANDIDATES * 0.5)
        else:
            initial_search = self.MIN_INITIAL_SEARCH
            opt_search = self.MIN_OPT_SEARCH
            n_candidates = self.MIN_CANDIDATES
        
        if time_pressure > 0.9:
            initial_search = max(self.MIN_INITIAL_SEARCH, initial_search // 2)
            opt_search = max(self.MIN_OPT_SEARCH, opt_search // 2)
            n_candidates = max(self.MIN_CANDIDATES, int(n_candidates * 0.6))
        elif time_pressure > 0.85:
            initial_search = max(self.MIN_INITIAL_SEARCH, int(initial_search * 0.7))
            opt_search = max(self.MIN_OPT_SEARCH, int(opt_search * 0.7))
            n_candidates = max(self.MIN_CANDIDATES, int(n_candidates * 0.75))
        
        n_remaining = game_state.get('n_remaining_balls', 7)
        if n_remaining <= 2 and time_pressure < 0.75:
            initial_search = min(initial_search + 6, self.MAX_INITIAL_SEARCH)
            opt_search = min(opt_search + 4, self.MAX_OPT_SEARCH)
        
        return max(initial_search, self.MIN_INITIAL_SEARCH), max(opt_search, self.MIN_OPT_SEARCH), max(n_candidates, self.MIN_CANDIDATES)
    
    def _random_action(self):
        return random_action()
    
    def _evaluate_candidates_parallel(self, candidates, balls, table, last_state, my_targets, 
                                     my_remaining, enemy_remaining, timeout):
        """Parallel evaluation with tactics"""
        if self.executor is None:
            self.executor = ProcessPoolExecutor(max_workers=self.n_cores)
        
        eval_args = [
            (candidate, balls, table, last_state, my_targets, my_remaining, enemy_remaining)
            for candidate in candidates
        ]
        
        try:
            futures = [self.executor.submit(evaluate_candidate_with_tactics, arg) for arg in eval_args]
            
            results = []
            for future in futures:
                try:
                    result = future.result(timeout=timeout/len(candidates) + 1.2)
                    results.append(result)
                except FutureTimeoutError:
                    results.append((None, -100))
                except Exception:
                    results.append((None, -500))
            
            return results
            
        except Exception as e:
            print(f"[EliteAgent] Parallel evaluation error: {e}")
            return [(c, -500) for c in candidates]
    
    def _create_optimizer(self, reward_function, seed):
        gpr = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=self.ALPHA,
            n_restarts_optimizer=6,
            random_state=seed
        )
        
        bounds_transformer = SequentialDomainReductionTransformer(
            gamma_osc=0.75,
            gamma_pan=1.0
        )
        
        optimizer = BayesianOptimization(
            f=reward_function,
            pbounds=self.pbounds,
            random_state=seed,
            verbose=0,
            bounds_transformer=bounds_transformer
        )
        optimizer._gp = gpr
        
        return optimizer
    
    def decision(self, balls=None, my_targets=None, table=None):
        """Enhanced decision with multi-strategy planning"""
        decision_start_time = time.time()
        
        if balls is None:
            return self._random_action()
        
        try:
            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                my_targets = ["8"]
                print("[EliteAgent] All targets cleared, switching to black 8")
            
            my_remaining = len(remaining_own)
            all_balls = set(str(i) for i in range(1, 16))
            enemy_targets = all_balls - set(my_targets) - {'8'}
            enemy_remaining = len([bid for bid in enemy_targets if balls[bid].state.s != 4])
            
            game_state = {'n_remaining_balls': my_remaining}
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            
            time_budget = self.time_manager.get_time_budget(game_state)
            time_pressure = self.time_manager.get_time_pressure()
            remaining_time = self.time_manager.get_remaining_time()
            
            initial_search, opt_search, n_candidates = self.get_adaptive_search_params(
                time_budget, game_state, time_pressure
            )
            
            situation = 'winning' if my_remaining < enemy_remaining else ('losing' if my_remaining > enemy_remaining else 'even')
            
            print(f"[EliteAgent] Budget: {time_budget:.1f}s, Pressure: {time_pressure:.2f}")
            print(f"  Situation: {situation.upper()} (My {my_remaining} vs Enemy {enemy_remaining})")
            print(f"  Search: {initial_search}+{opt_search}, Candidates: {n_candidates}, Cores: {self.n_cores}")
            
            # Generate multi-strategy candidates
            candidates = generate_elite_candidates(
                balls, my_targets, table, my_remaining, enemy_remaining, n_candidates
            )
            
            eval_start = time.time()
            max_eval_time = time_budget * 0.45
            
            print(f"[EliteAgent] Parallel evaluating {len(candidates)} multi-strategy candidates...")
            results = self._evaluate_candidates_parallel(
                candidates, balls, table, last_state_snapshot, my_targets,
                my_remaining, enemy_remaining, timeout=max_eval_time
            )
            
            best_candidate = None
            best_candidate_score = -float('inf')
            
            for candidate, score in results:
                if candidate is not None and score > best_candidate_score:
                    best_candidate_score = score
                    best_candidate = candidate
            
            eval_time = time.time() - eval_start
            print(f"[EliteAgent] Eval took {eval_time:.2f}s, best score: {best_candidate_score:.2f}")
            
            remaining_decision_time = time_budget - (time.time() - decision_start_time)
            
            # Refinement decision
            if best_candidate_score >= 70:
                action = best_candidate
                print(f"[EliteAgent] Excellent candidate, using directly")
            elif best_candidate_score >= 50 and remaining_decision_time < 2.0:
                action = best_candidate
                print(f"[EliteAgent] Good candidate + low time")
            elif remaining_decision_time > 1.5:
                print(f"[EliteAgent] Running Bayesian optimization for refinement...")
                
                def reward_fn_wrapper(V0, phi, theta, a, b):
                    candidate = {'V0': V0, 'phi': phi, 'theta': theta, 'a': a, 'b': b}
                    results = self._evaluate_candidates_parallel(
                        [candidate], balls, table, last_state_snapshot, my_targets,
                        my_remaining, enemy_remaining, timeout=2.0
                    )
                    return results[0][1] if results else -500
                
                seed = np.random.randint(1e6)
                optimizer = self._create_optimizer(reward_fn_wrapper, seed)
                
                # Seed with top candidates
                for candidate, score in sorted(results, key=lambda x: x[1], reverse=True)[:6]:
                    if candidate is not None:
                        try:
                            optimizer.probe(params=candidate, lazy=True)
                        except:
                            pass
                
                optimizer.maximize(
                    init_points=initial_search,
                    n_iter=opt_search
                )
                
                best_result = optimizer.max
                best_params = best_result['params']
                best_score = best_result['target']
                
                if best_score > best_candidate_score:
                    action = {
                        'V0': float(best_params['V0']),
                        'phi': float(best_params['phi']),
                        'theta': float(best_params['theta']),
                        'a': float(best_params['a']),
                        'b': float(best_params['b']),
                    }
                    print(f"[EliteAgent] Refined (score: {best_score:.2f})")
                else:
                    action = best_candidate
                    print(f"[EliteAgent] Candidate better (score: {best_candidate_score:.2f})")
            else:
                action = best_candidate if best_candidate_score >= 10 else self._random_action()
                print(f"[EliteAgent] Using {'candidate' if best_candidate_score >= 10 else 'random'} (no time for optimization)")
            
            decision_time = time.time() - decision_start_time
            self.time_manager.learn_from_decision(decision_time)
            
            print(f"[EliteAgent] Decision took {decision_time:.2f}s")
            
            return action
        
        except Exception as e:
            print(f"[EliteAgent] Error: {e}")
            import traceback
            traceback.print_exc()
            
            decision_time = time.time() - decision_start_time
            self.time_manager.learn_from_decision(decision_time)
            
            return self._random_action()


