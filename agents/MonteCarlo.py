"""
MonteCarlo.py - Enhanced Monte Carlo Tree Search Agent

Optimized MCTS agent with hybrid approach combining:
1. Improved heuristic action generation
2. Bayesian Optimization for action refinement
3. Better position evaluation
4. Strategic shot selection

Key Improvements:
- Better ghost ball calculation with pocket geometry
- Position-aware reward function
- Hybrid MCTS + Bayesian Optimization
- Adaptive search based on game state
- Improved target ball selection
"""

import math
import numpy as np
import pooltool as pt
import copy
import time
import random
import signal
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


# ============ Global Time Manager ============
class GlobalTimeManager:
    """Singleton class to manage time budget across all games"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalTimeManager, cls).__new__(cls)
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
        self.time_spent = 0.0
        self.decisions_made = 0
        
    def initialize(self, n_games, time_per_game=180.0):
        """Initialize global time budget"""
        self.total_time_budget = n_games * time_per_game
        self.start_time = time.time()
        self.total_games = n_games
        self.current_game = 0
        self.time_spent = 0.0
        self.decisions_made = 0
        print(f"[GlobalTimeManager] Initialized with {self.total_time_budget:.0f}s budget for {n_games} games")
    
    def get_time_budget(self, game_state=None):
        """Calculate time budget for current decision"""
        if self.start_time is None:
            return 5.0
        
        elapsed = time.time() - self.start_time
        remaining = self.total_time_budget - elapsed
        safe_remaining = remaining * 0.95
        
        if safe_remaining <= 0:
            return 0.3
        
        games_remaining = max(1, self.total_games - self.current_game)
        estimated_decisions_remaining = games_remaining * 30
        base_budget = safe_remaining / estimated_decisions_remaining
        
        complexity_multiplier = 1.0
        if game_state:
            n_remaining = game_state.get('n_remaining_balls', 7)
            if n_remaining <= 2:
                complexity_multiplier = 1.5
            elif n_remaining >= 6:
                complexity_multiplier = 0.8
        
        time_budget = base_budget * complexity_multiplier
        return max(0.3, min(15.0, time_budget))
    
    def get_remaining_time(self):
        if self.start_time is None:
            return float('inf')
        return self.total_time_budget - (time.time() - self.start_time)
    
    def get_time_pressure(self):
        if self.start_time is None:
            return 0.0
        elapsed = time.time() - self.start_time
        return min(1.0, elapsed / self.total_time_budget)
    
    def record_decision(self, decision_time):
        self.time_spent += decision_time
        self.decisions_made += 1
    
    def start_new_game(self):
        self.current_game += 1
        if self.current_game % 10 == 0:
            elapsed = time.time() - self.start_time if self.start_time else 0
            remaining = self.total_time_budget - elapsed
            print(f"[GlobalTimeManager] Game {self.current_game}/{self.total_games}, "
                  f"Time: {elapsed:.0f}s used, {remaining:.0f}s remaining, "
                  f"Decisions: {self.decisions_made}")
    
    def get_stats(self):
        elapsed = time.time() - self.start_time if self.start_time else 0
        remaining = self.total_time_budget - elapsed
        return {
            'elapsed': elapsed,
            'remaining': remaining,
            'budget': self.total_time_budget,
            'decisions': self.decisions_made,
            'avg_time_per_decision': elapsed / max(1, self.decisions_made),
            'games_completed': self.current_game,
            'utilization': elapsed / self.total_time_budget if self.total_time_budget > 0 else 0
        }


global_time_manager = GlobalTimeManager()


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


# ============ Enhanced Reward Function ============
def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """Enhanced reward function with position evaluation"""
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
    
    # Major penalties/rewards
    if cue_pocketed and eight_pocketed:
        score -= 200  # Worse than before
    elif cue_pocketed:
        score -= 120  # Increased penalty
    elif eight_pocketed:
        if player_targets == ['8']:
            score += 150  # Increased reward
        else:
            score -= 200  # Severe penalty
            
    if foul_first_hit:
        score -= 40  # Increased penalty
    if foul_no_rail:
        score -= 40
        
    # Ball pocketing rewards
    score += len(own_pocketed) * 60  # Increased from 50
    score -= len(enemy_pocketed) * 30  # Increased penalty
    
    # Legal no-pocket small reward
    if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
        score = 5  # Reduced from 10 - we want to be aggressive
    
    # Position bonus: reward good cue ball position
    if not cue_pocketed and 'cue' in shot.balls:
        cue_pos = shot.balls['cue'].state.rvw[0][:2]
        # Bonus for center table position (easier next shots)
        center_x, center_y = 1.12, 1.12  # Table center
        dist_from_center = np.linalg.norm(cue_pos - np.array([center_x, center_y]))
        # Small bonus for good position
        if dist_from_center < 0.8:
            score += 5
        
    return score


# ============ Geometry Helper Functions ============
def get_ball_position(ball):
    return ball.state.rvw[0][:2]

def distance_2d(pos1, pos2):
    return np.linalg.norm(np.array(pos1) - np.array(pos2))

def angle_between_points(from_pos, to_pos):
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    angle = math.degrees(math.atan2(dy, dx))
    return angle % 360

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
        
        if dist < 2.2 * ball_radius:  # Slightly tighter tolerance
            return False
    
    return True


def calculate_shot_difficulty(cue_pos, target_pos, pocket_pos, balls, target_id):
    """Enhanced difficulty calculation"""
    difficulty = 0
    
    # Distance factors
    cue_to_target_dist = distance_2d(cue_pos, target_pos)
    difficulty += cue_to_target_dist * 8  # Reduced weight - distance matters less
    
    target_to_pocket_dist = distance_2d(target_pos, pocket_pos)
    difficulty += target_to_pocket_dist * 25  # Increased - pocket distance matters more
    
    # Path clearance
    if not has_clear_path(cue_pos, target_pos, balls, ['cue', target_id]):
        difficulty += 60  # Increased penalty for blocked shots
    
    # Check target to pocket clearance
    if not has_clear_path(target_pos, pocket_pos, balls, [target_id, 'cue']):
        difficulty += 40
    
    # Cut angle penalty
    cue_to_target_angle = angle_between_points(cue_pos, target_pos)
    target_to_pocket_angle = angle_between_points(target_pos, pocket_pos)
    cut_angle = abs(target_to_pocket_angle - cue_to_target_angle)
    cut_angle = min(cut_angle, 360 - cut_angle)
    
    # More aggressive cut angle penalty
    if cut_angle > 70:
        difficulty += (cut_angle - 70) * 3
    elif cut_angle > 45:
        difficulty += (cut_angle - 45) * 1.5
    
    # Bonus for straight shots
    if cut_angle < 15:
        difficulty -= 15
    
    return difficulty


def select_best_target(balls, my_targets, table):
    """Enhanced target selection with multiple strategies"""
    cue_pos = get_ball_position(balls['cue'])
    remaining_targets = [bid for bid in my_targets if balls[bid].state.s != 4]
    
    if not remaining_targets:
        return None
    
    pocket_positions = {pid: pocket.center[:2] for pid, pocket in table.pockets.items()}
    
    best_target = None
    best_score = float('inf')
    
    for target_id in remaining_targets:
        target_pos = get_ball_position(balls[target_id])
        min_difficulty = float('inf')
        
        for pocket_id, pocket_pos in pocket_positions.items():
            difficulty = calculate_shot_difficulty(
                cue_pos, target_pos, pocket_pos, balls, target_id
            )
            min_difficulty = min(min_difficulty, difficulty)
        
        if min_difficulty < best_score:
            best_score = min_difficulty
            best_target = target_id
    
    return best_target


def calculate_ghost_ball_position(target_pos, pocket_pos, ball_radius=0.028575):
    """Precise ghost ball calculation"""
    target_pos = np.array(target_pos)
    pocket_pos = np.array(pocket_pos)
    
    direction = target_pos - pocket_pos
    dist = np.linalg.norm(direction)
    
    if dist < 1e-6:
        return target_pos
    
    direction = direction / dist
    ghost_pos = target_pos + direction * (2 * ball_radius)
    
    return ghost_pos


def generate_candidate_actions(balls, my_targets, table, n_actions=15):
    """Enhanced candidate generation with multiple strategies"""
    cue_pos = get_ball_position(balls['cue'])
    target_id = select_best_target(balls, my_targets, table)
    
    if target_id is None:
        # Random fallback
        return [{
            'V0': random.uniform(2.5, 5.5),
            'phi': random.uniform(0, 360),
            'theta': random.uniform(0, 15),
            'a': 0.0,
            'b': 0.0
        } for _ in range(n_actions)]
    
    target_pos = get_ball_position(balls[target_id])
    pocket_positions = {pid: pocket.center[:2] for pid, pocket in table.pockets.items()}
    
    candidates = []
    
    # Sort pockets by distance from target
    sorted_pockets = sorted(
        pocket_positions.items(),
        key=lambda x: distance_2d(target_pos, x[1])
    )
    
    # Generate candidates for best pockets
    for pocket_id, pocket_pos in sorted_pockets[:4]:  # Top 4 closest pockets
        ghost_pos = calculate_ghost_ball_position(target_pos, pocket_pos)
        phi_base = angle_between_points(cue_pos, ghost_pos)
        dist = distance_2d(cue_pos, target_pos)
        
        # Better speed heuristic
        if dist < 0.4:
            V0_base = 2.0
        elif dist < 0.8:
            V0_base = 3.0
        elif dist < 1.2:
            V0_base = 3.8
        elif dist < 1.8:
            V0_base = 4.5
        else:
            V0_base = 5.5
        
        # Generate variations - more focused
        for v_offset in [-0.8, -0.4, 0, 0.4, 0.8]:
            for phi_offset in [-8, -4, 0, 4, 8]:
                for theta_val in [0.0, 5.0]:  # Top spin variations
                    V0 = np.clip(V0_base + v_offset, 0.5, 8.0)
                    phi_adjusted = (phi_base + phi_offset) % 360
                    
                    candidate = {
                        'V0': V0,
                        'phi': phi_adjusted,
                        'theta': theta_val,
                        'a': 0.0,
                        'b': 0.0
                    }
                    candidates.append(candidate)
                    
                    if len(candidates) >= n_actions:
                        return candidates[:n_actions]
    
    # Fill with variations if needed
    while len(candidates) < n_actions:
        candidates.append({
            'V0': random.uniform(2.0, 6.0),
            'phi': random.uniform(0, 360),
            'theta': random.uniform(0, 15),
            'a': 0.0,
            'b': 0.0
        })
    
    return candidates[:n_actions]


# ============ MCTS Agent with Bayesian Optimization ============
class MCTSAgent:
    """Hybrid MCTS + Bayesian Optimization Agent"""
    
    def __init__(self):
        self.time_manager = global_time_manager
        
        # Search parameters
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
        
        print("MCTSAgent (Enhanced Hybrid MCTS + BO) initialized.")
    
    def _create_optimizer(self, reward_function, seed):
        """Create Bayesian optimizer"""
        gpr = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=self.ALPHA,
            n_restarts_optimizer=5,
            random_state=seed
        )
        
        bounds_transformer = SequentialDomainReductionTransformer(
            gamma_osc=0.8,
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
    
    def _random_action(self):
        return {
            'V0': random.uniform(2.0, 6.0),
            'phi': random.uniform(0, 360),
            'theta': random.uniform(0, 20),
            'a': 0.0,
            'b': 0.0
        }
    
    def decision(self, balls=None, my_targets=None, table=None):
        """Hybrid decision making: candidates + Bayesian optimization"""
        decision_start_time = time.time()
        
        if balls is None:
            return self._random_action()
        
        try:
            # Prepare game state
            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                my_targets = ["8"]
                print("[MCTS] All targets cleared, switching to black 8")
            
            game_state = {'n_remaining_balls': len(remaining_own)}
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            
            # Get time budget
            time_budget = self.time_manager.get_time_budget(game_state)
            time_pressure = self.time_manager.get_time_pressure()
            remaining_time = self.time_manager.get_remaining_time()
            
            # Determine search parameters based on time
            if time_budget > 8.0:
                n_candidates = 20
                initial_search = 12
                opt_search = 8
            elif time_budget > 5.0:
                n_candidates = 15
                initial_search = 10
                opt_search = 6
            elif time_budget > 3.0:
                n_candidates = 12
                initial_search = 7
                opt_search = 4
            elif time_budget > 1.5:
                n_candidates = 10
                initial_search = 5
                opt_search = 3
            else:
                n_candidates = 8
                initial_search = 3
                opt_search = 2
            
            # Adjust for time pressure
            if time_pressure > 0.9:
                n_candidates = max(6, n_candidates // 3)
                initial_search = max(2, initial_search // 3)
                opt_search = max(1, opt_search // 3)
            elif time_pressure > 0.8:
                n_candidates = max(8, n_candidates // 2)
                initial_search = max(3, initial_search // 2)
                opt_search = max(2, opt_search // 2)
            
            print(f"[MCTS] Budget: {time_budget:.1f}s, Pressure: {time_pressure:.2f}, "
                  f"Search: {initial_search}+{opt_search}, Candidates: {n_candidates}")
            
            # Define reward function
            def reward_fn_wrapper(V0, phi, theta, a, b):
                sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                cue = pt.Cue(cue_ball_id="cue")
                shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
                
                try:
                    shot.cue.set_state(V0=V0, phi=phi, theta=theta, a=a, b=b)
                    if not simulate_with_timeout(shot, timeout=2):
                        return -50
                except Exception as e:
                    return -500
                
                score = analyze_shot_for_reward(
                    shot=shot,
                    last_state=last_state_snapshot,
                    player_targets=my_targets
                )
                
                return score
            
            # Step 1: Generate and evaluate candidates
            candidates = generate_candidate_actions(balls, my_targets, table, n_candidates)
            
            best_candidate = None
            best_candidate_score = -float('inf')
            
            eval_start = time.time()
            max_eval_time = time_budget * 0.3
            
            print(f"[MCTS] Evaluating {len(candidates)} candidates...")
            for i, candidate in enumerate(candidates):
                if time.time() - eval_start > max_eval_time:
                    print(f"[MCTS] Candidate eval timeout after {i} candidates")
                    break
                
                score = reward_fn_wrapper(**candidate)
                if score > best_candidate_score:
                    best_candidate_score = score
                    best_candidate = candidate
            
            print(f"[MCTS] Best candidate score: {best_candidate_score:.2f}")
            
            # Step 2: Decide whether to optimize
            remaining_decision_time = time_budget - (time.time() - decision_start_time)
            
            # If we have a great candidate and are under pressure, use it
            if best_candidate_score >= 60 and (time_pressure > 0.85 or remaining_decision_time < 1.2):
                action = best_candidate
                print(f"[MCTS] Using great candidate (score: {best_candidate_score:.2f})")
            # If we have good candidate and very low time, use it
            elif best_candidate_score >= 40 and remaining_decision_time < 0.8:
                action = best_candidate
                print(f"[MCTS] Using good candidate due to time constraint")
            # Otherwise, try to optimize if we have time
            elif remaining_decision_time > 1.0:
                print(f"[MCTS] Running Bayesian optimization...")
                seed = np.random.randint(1e6)
                optimizer = self._create_optimizer(reward_fn_wrapper, seed)
                
                # Seed with best candidates
                for candidate in candidates[:min(5, len(candidates))]:
                    try:
                        optimizer.probe(params=candidate, lazy=True)
                    except:
                        pass
                
                try:
                    optimizer.maximize(
                        init_points=initial_search,
                        n_iter=opt_search
                    )
                    
                    best_result = optimizer.max
                    best_params = best_result['params']
                    best_score = best_result['target']
                    
                    # Use optimized if better
                    if best_score > best_candidate_score:
                        action = {
                            'V0': float(best_params['V0']),
                            'phi': float(best_params['phi']),
                            'theta': float(best_params['theta']),
                            'a': float(best_params['a']),
                            'b': float(best_params['b']),
                        }
                        print(f"[MCTS] Optimized (score: {best_score:.2f})")
                    else:
                        action = best_candidate
                        print(f"[MCTS] Candidate better (score: {best_candidate_score:.2f})")
                except Exception as e:
                    print(f"[MCTS] Optimization failed: {e}, using candidate")
                    action = best_candidate if best_candidate else self._random_action()
            else:
                # No time for optimization
                if best_candidate_score >= 10:
                    action = best_candidate
                    print(f"[MCTS] Using candidate (score: {best_candidate_score:.2f}, no time)")
                else:
                    action = self._random_action()
                    print(f"[MCTS] Random (no good solution)")
            
            # Record decision time
            decision_time = time.time() - decision_start_time
            self.time_manager.record_decision(decision_time)
            
            print(f"[MCTS] Decision took {decision_time:.2f}s")
            print(f"[MCTS] Action: V0={action['V0']:.2f}, phi={action['phi']:.2f}, "
                  f"theta={action['theta']:.2f}")
            
            return action
        
        except Exception as e:
            print(f"[MCTS] Error: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()
