"""
DynaHeuristic.py - Dynamic Heuristic Agent with Adaptive Time Management

This agent dynamically adjusts its search depth based on:
1. Time remaining in the game (3-minute limit)
2. Game state (early game vs late game)
3. Position complexity

Key Features:
- Adaptive search budget allocation
- Time tracking per game
- More exploration when time allows
- Quick decisions when time is running out
"""

import math
import numpy as np
import pooltool as pt
import copy
import time
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import signal


# ============ Timeout Protection ============
class SimulationTimeoutError(Exception):
    """Physics simulation timeout exception"""
    pass


def _timeout_handler(signum, frame):
    """Timeout signal handler"""
    raise SimulationTimeoutError("Physics simulation timeout")


def simulate_with_timeout(shot, timeout=3):
    """Physics simulation with timeout protection"""
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)
    
    try:
        pt.simulate(shot, inplace=True)
        signal.alarm(0)
        return True
    except SimulationTimeoutError:
        print(f"[WARNING] Physics simulation timeout (>{timeout}s), skipping")
        return False
    except Exception as e:
        signal.alarm(0)
        raise e
    finally:
        signal.signal(signal.SIGALRM, old_handler)


# ============ Reward Function ============
def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """Analyze shot result and calculate reward score"""
    # 1. Basic analysis
    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
    
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
    
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # 2. Analyze first contact
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
    
    # 3. Analyze cushion contact
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
        
    # 4. Calculate reward
    score = 0
    
    if cue_pocketed and eight_pocketed:
        score -= 150
    elif cue_pocketed:
        score -= 100
    elif eight_pocketed:
        if player_targets == ['8']:
            score += 100
        else:
            score -= 150
            
    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 30
        
    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20
    
    if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
        score = 10
        
    return score


# ============ Dynamic Heuristic Agent ============
class DynamicHeuristicAgent:
    """Dynamic Heuristic Agent with Adaptive Time Management"""
    
    def __init__(self):
        """Initialize agent"""
        # Search space
        self.pbounds = {
            'V0': (0.5, 8.0),
            'phi': (0, 360),
            'theta': (0, 90), 
            'a': (-0.5, 0.5),
            'b': (-0.5, 0.5)
        }
        
        # Time management
        self.GAME_TIME_LIMIT = 180.0  # 3 minutes = 180 seconds
        self.game_start_time = None
        self.total_decision_time = 0.0
        self.decision_count = 0
        
        # Dynamic search parameters
        self.MIN_INITIAL_SEARCH = 3
        self.MAX_INITIAL_SEARCH = 15
        self.MIN_OPT_SEARCH = 2
        self.MAX_OPT_SEARCH = 8
        self.MIN_CANDIDATES = 8
        self.MAX_CANDIDATES = 20
        
        self.ALPHA = 1e-2
        
        # Table dimensions
        self.TABLE_WIDTH = 1.12
        self.TABLE_LENGTH = 2.24
        self.BALL_RADIUS = 0.028575
        
        print("DynamicHeuristicAgent (Adaptive Time Management) initialized.")
    
    
    def reset_game_timer(self):
        """Reset timer for a new game"""
        self.game_start_time = time.time()
        self.total_decision_time = 0.0
        self.decision_count = 0
        print("[DynamicAgent] Game timer reset")
    
    
    def get_time_budget(self):
        """Calculate time budget for current decision"""
        if self.game_start_time is None:
            self.reset_game_timer()
        
        elapsed = time.time() - self.game_start_time
        remaining = self.GAME_TIME_LIMIT - elapsed
        
        # Estimate remaining decisions (assume average 30 shots per game)
        estimated_remaining_decisions = max(1, 30 - self.decision_count)
        
        # Allocate time budget (with safety margin)
        time_budget = (remaining * 0.8) / estimated_remaining_decisions
        
        # Clamp between 0.5 and 10 seconds
        time_budget = max(0.5, min(10.0, time_budget))
        
        return time_budget, remaining
    
    
    def get_adaptive_search_params(self, time_budget, game_state):
        """Determine search parameters based on time budget and game state"""
        # Base parameters on time budget
        if time_budget > 8.0:
            # Plenty of time - thorough search
            initial_search = self.MAX_INITIAL_SEARCH
            opt_search = self.MAX_OPT_SEARCH
            n_candidates = self.MAX_CANDIDATES
        elif time_budget > 5.0:
            # Good amount of time - balanced search
            initial_search = int(self.MAX_INITIAL_SEARCH * 0.7)
            opt_search = int(self.MAX_OPT_SEARCH * 0.7)
            n_candidates = int(self.MAX_CANDIDATES * 0.7)
        elif time_budget > 3.0:
            # Moderate time - reduced search
            initial_search = int(self.MAX_INITIAL_SEARCH * 0.5)
            opt_search = int(self.MAX_OPT_SEARCH * 0.5)
            n_candidates = int(self.MAX_CANDIDATES * 0.5)
        elif time_budget > 1.5:
            # Limited time - minimal search
            initial_search = int(self.MAX_INITIAL_SEARCH * 0.3)
            opt_search = int(self.MAX_OPT_SEARCH * 0.3)
            n_candidates = int(self.MAX_CANDIDATES * 0.5)
        else:
            # Very limited time - fast heuristic only
            initial_search = self.MIN_INITIAL_SEARCH
            opt_search = self.MIN_OPT_SEARCH
            n_candidates = self.MIN_CANDIDATES
        
        # Adjust based on game state complexity
        n_remaining = game_state.get('n_remaining_balls', 7)
        if n_remaining <= 2:
            # End game - critical shots, increase search if time allows
            initial_search = min(initial_search + 2, self.MAX_INITIAL_SEARCH)
            opt_search = min(opt_search + 1, self.MAX_OPT_SEARCH)
        
        # Ensure minimums
        initial_search = max(initial_search, self.MIN_INITIAL_SEARCH)
        opt_search = max(opt_search, self.MIN_OPT_SEARCH)
        n_candidates = max(n_candidates, self.MIN_CANDIDATES)
        
        return initial_search, opt_search, n_candidates
    
    
    def _random_action(self):
        """Generate random shot action"""
        import random
        action = {
            'V0': round(random.uniform(0.5, 8.0), 2),
            'phi': round(random.uniform(0, 360), 2),
            'theta': round(random.uniform(0, 90), 2),
            'a': round(random.uniform(-0.5, 0.5), 3),
            'b': round(random.uniform(-0.5, 0.5), 3)
        }
        return action
    
    
    def _get_ball_position(self, ball):
        """Get 2D position of ball (x, y)"""
        return ball.state.rvw[0][:2]
    
    
    def _distance_2d(self, pos1, pos2):
        """Calculate 2D Euclidean distance"""
        return np.linalg.norm(np.array(pos1) - np.array(pos2))
    
    
    def _angle_between_points(self, from_pos, to_pos):
        """Calculate angle in degrees from from_pos to to_pos"""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        angle = math.degrees(math.atan2(dy, dx))
        return angle % 360
    
    
    def _has_clear_path(self, from_pos, to_pos, balls, ignore_ids):
        """Check if there's a clear path between two positions"""
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
            
            ball_pos = self._get_ball_position(ball)
            to_ball = ball_pos - from_pos
            projection = np.dot(to_ball, path_dir)
            
            if projection < 0 or projection > path_length:
                continue
            
            closest_point = from_pos + projection * path_dir
            dist = np.linalg.norm(ball_pos - closest_point)
            
            if dist < 2.5 * self.BALL_RADIUS:
                return False
        
        return True
    
    
    def _calculate_shot_difficulty(self, cue_pos, target_pos, pocket_pos, balls, target_id):
        """Calculate difficulty score for a shot (lower is easier)"""
        difficulty = 0
        
        cue_to_target_dist = self._distance_2d(cue_pos, target_pos)
        difficulty += cue_to_target_dist * 10
        
        target_to_pocket_dist = self._distance_2d(target_pos, pocket_pos)
        difficulty += target_to_pocket_dist * 20
        
        if not self._has_clear_path(cue_pos, target_pos, balls, ['cue', target_id]):
            difficulty += 50
        
        cue_to_target_angle = self._angle_between_points(cue_pos, target_pos)
        target_to_pocket_angle = self._angle_between_points(target_pos, pocket_pos)
        cut_angle = abs(target_to_pocket_angle - cue_to_target_angle)
        cut_angle = min(cut_angle, 360 - cut_angle)
        
        if cut_angle > 60:
            difficulty += (cut_angle - 60) * 2
        
        return difficulty
    
    
    def _select_best_target(self, balls, my_targets, table):
        """Select the best target ball to shoot at"""
        cue_pos = self._get_ball_position(balls['cue'])
        remaining_targets = [bid for bid in my_targets if balls[bid].state.s != 4]
        
        if not remaining_targets:
            return None
        
        pocket_positions = {pid: pocket.center[:2] for pid, pocket in table.pockets.items()}
        
        best_target = None
        best_score = float('inf')
        
        for target_id in remaining_targets:
            target_pos = self._get_ball_position(balls[target_id])
            min_difficulty = float('inf')
            
            for pocket_id, pocket_pos in pocket_positions.items():
                difficulty = self._calculate_shot_difficulty(
                    cue_pos, target_pos, pocket_pos, balls, target_id
                )
                min_difficulty = min(min_difficulty, difficulty)
            
            if min_difficulty < best_score:
                best_score = min_difficulty
                best_target = target_id
        
        return best_target
    
    
    def _calculate_ghost_ball_position(self, target_pos, pocket_pos):
        """Calculate ghost ball position for aiming"""
        target_pos = np.array(target_pos)
        pocket_pos = np.array(pocket_pos)
        
        direction = target_pos - pocket_pos
        dist = np.linalg.norm(direction)
        
        if dist < 1e-6:
            return target_pos
        
        direction = direction / dist
        ghost_pos = target_pos + direction * (2 * self.BALL_RADIUS)
        
        return ghost_pos
    
    
    def _generate_candidate_shots(self, balls, my_targets, table, n_candidates=10):
        """Generate candidate shots using heuristics"""
        cue_pos = self._get_ball_position(balls['cue'])
        target_id = self._select_best_target(balls, my_targets, table)
        
        if target_id is None:
            return [self._random_action() for _ in range(n_candidates)]
        
        target_pos = self._get_ball_position(balls[target_id])
        pocket_positions = {pid: pocket.center[:2] for pid, pocket in table.pockets.items()}
        
        candidates = []
        
        # Sort pockets by distance to target
        pocket_list = sorted(
            pocket_positions.items(),
            key=lambda x: self._distance_2d(target_pos, x[1])
        )
        
        for pocket_id, pocket_pos in pocket_list:
            ghost_pos = self._calculate_ghost_ball_position(target_pos, pocket_pos)
            phi = self._angle_between_points(cue_pos, ghost_pos)
            dist = self._distance_2d(cue_pos, target_pos)
            
            # Speed heuristic
            if dist < 0.5:
                V0_base = 2.0
            elif dist < 1.0:
                V0_base = 3.0
            elif dist < 1.5:
                V0_base = 4.0
            else:
                V0_base = 5.0
            
            # Generate variations
            for v_offset in [-0.5, 0, 0.5]:
                for phi_offset in [-8, -4, 0, 4, 8]:
                    for theta_val in [0.0, 5.0]:
                        V0 = np.clip(V0_base + v_offset, 0.5, 8.0)
                        phi_adjusted = (phi + phi_offset) % 360
                        
                        candidate = {
                            'V0': V0,
                            'phi': phi_adjusted,
                            'theta': theta_val,
                            'a': 0.0,
                            'b': 0.0
                        }
                        candidates.append(candidate)
                        
                        if len(candidates) >= n_candidates:
                            break
                    if len(candidates) >= n_candidates:
                        break
                if len(candidates) >= n_candidates:
                    break
            if len(candidates) >= n_candidates:
                break
        
        while len(candidates) < n_candidates:
            candidates.append(self._random_action())
        
        return candidates[:n_candidates]
    
    
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
    
    
    def decision(self, balls=None, my_targets=None, table=None):
        """Make decision with dynamic time management"""
        decision_start_time = time.time()
        
        if balls is None:
            print(f"[DynamicAgent] No ball information, using random action.")
            return self._random_action()
        
        try:
            # Get time budget
            time_budget, time_remaining = self.get_time_budget()
            
            # Save state snapshot
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            
            # Check if all targets are pocketed
            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                my_targets = ["8"]
                print("[DynamicAgent] All targets cleared, switching to black 8")
            
            # Game state for adaptive parameters
            game_state = {
                'n_remaining_balls': len(remaining_own)
            }
            
            # Get adaptive search parameters
            initial_search, opt_search, n_candidates = self.get_adaptive_search_params(
                time_budget, game_state
            )
            
            print(f"[DynamicAgent] Time: {time_remaining:.1f}s remaining, "
                  f"Budget: {time_budget:.1f}s, "
                  f"Search: {initial_search}+{opt_search}, Candidates: {n_candidates}")
            
            # Generate smart candidates
            candidates = self._generate_candidate_shots(balls, my_targets, table, n_candidates)
            
            # Define reward function
            def reward_fn_wrapper(V0, phi, theta, a, b):
                sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                cue = pt.Cue(cue_ball_id="cue")
                shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
                
                try:
                    shot.cue.set_state(V0=V0, phi=phi, theta=theta, a=a, b=b)
                    if not simulate_with_timeout(shot, timeout=3):
                        return 0
                except Exception as e:
                    return -500
                
                score = analyze_shot_for_reward(
                    shot=shot,
                    last_state=last_state_snapshot,
                    player_targets=my_targets
                )
                
                return score
            
            # Quick evaluation of candidates
            best_candidate = None
            best_candidate_score = -float('inf')
            
            eval_start = time.time()
            for i, candidate in enumerate(candidates):
                # Check time budget
                if time.time() - eval_start > time_budget * 0.4:
                    print(f"[DynamicAgent] Time limit during candidate eval, stopping at {i+1}/{len(candidates)}")
                    break
                
                score = reward_fn_wrapper(**candidate)
                if score > best_candidate_score:
                    best_candidate_score = score
                    best_candidate = candidate
            
            # If we found a good candidate and time is tight, use it
            if best_candidate_score >= 40 and time_budget < 2.0:
                print(f"[DynamicAgent] Using good candidate (score: {best_candidate_score:.2f}) due to time constraint")
                self.decision_count += 1
                self.total_decision_time += time.time() - decision_start_time
                return best_candidate
            
            # If we have time, do Bayesian optimization
            remaining_time = time_budget - (time.time() - decision_start_time)
            if remaining_time > 1.0:
                print(f"[DynamicAgent] Running optimization with {initial_search}+{opt_search} iterations...")
                seed = np.random.randint(1e6)
                optimizer = self._create_optimizer(reward_fn_wrapper, seed)
                
                # Seed with best candidates
                for candidate in candidates[:min(3, len(candidates))]:
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
                    print(f"[DynamicAgent] Optimized decision (score: {best_score:.2f})")
                else:
                    action = best_candidate
                    print(f"[DynamicAgent] Using candidate (score: {best_candidate_score:.2f})")
            else:
                # Not enough time for optimization
                if best_candidate_score >= 10:
                    action = best_candidate
                    print(f"[DynamicAgent] Using candidate (score: {best_candidate_score:.2f}, no time for opt)")
                else:
                    print(f"[DynamicAgent] No good solution, using random")
                    action = self._random_action()
            
            # Update counters
            self.decision_count += 1
            decision_time = time.time() - decision_start_time
            self.total_decision_time += decision_time
            
            print(f"[DynamicAgent] Decision took {decision_time:.2f}s, "
                  f"Total: {self.total_decision_time:.1f}s")
            
            return action
        
        except Exception as e:
            print(f"[DynamicAgent] Error during decision: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()



