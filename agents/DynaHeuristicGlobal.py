"""
DynaHeuristicGlobal.py - Global Time Budget Agent

This agent manages time across ALL games in an evaluation session, not per-game.
Key innovation: Shares a global time budget of 3 minutes × n_games across the entire evaluation.

Strategy:
1. Global time pool: 180s × n_games
2. Adaptive allocation: Use more time for complex positions, less for simple ones
3. Progressive urgency: Increase speed as global time runs low
4. No per-game limits: Can spend 5 minutes on one game, 1 minute on another

Benefits:
- Better time utilization across evaluation
- Can afford thorough search when needed
- Speeds up when time is running low
- Maximizes overall performance
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
        
        # Global time tracking
        self.total_time_budget = 0.0  # Will be set by initialize()
        self.start_time = None
        self.total_decisions = 0
        self.total_games = 0
        self.current_game = 0
        
        # Statistics
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
            return 5.0  # Default if not initialized
        
        elapsed = time.time() - self.start_time
        remaining = self.total_time_budget - elapsed
        
        # Safety margin: stop using time at 95% to ensure completion
        safe_remaining = remaining * 0.95
        
        if safe_remaining <= 0:
            return 0.3  # Minimal time for emergency decisions
        
        # Estimate remaining decisions across all remaining games
        # Assume average 30 decisions per game
        games_remaining = max(1, self.total_games - self.current_game)
        estimated_decisions_remaining = games_remaining * 30
        
        # Base allocation
        base_budget = safe_remaining / estimated_decisions_remaining
        
        # Adjust based on game state complexity
        complexity_multiplier = 1.0
        if game_state:
            n_remaining = game_state.get('n_remaining_balls', 7)
            if n_remaining <= 2:
                # Critical end-game shots deserve more time
                complexity_multiplier = 1.5
            elif n_remaining >= 6:
                # Early game can be faster
                complexity_multiplier = 0.8
        
        time_budget = base_budget * complexity_multiplier
        
        # Clamp to reasonable range
        time_budget = max(0.3, min(15.0, time_budget))
        
        return time_budget
    
    def get_remaining_time(self):
        """Get remaining time in budget"""
        if self.start_time is None:
            return float('inf')
        elapsed = time.time() - self.start_time
        return self.total_time_budget - elapsed
    
    def get_time_pressure(self):
        """Get time pressure factor (0-1, higher = more pressure)"""
        if self.start_time is None:
            return 0.0
        elapsed = time.time() - self.start_time
        used_ratio = elapsed / self.total_time_budget
        return min(1.0, used_ratio)
    
    def record_decision(self, decision_time):
        """Record time spent on a decision"""
        self.time_spent += decision_time
        self.decisions_made += 1
    
    def start_new_game(self):
        """Mark start of new game"""
        self.current_game += 1
        if self.current_game % 10 == 0:
            elapsed = time.time() - self.start_time if self.start_time else 0
            remaining = self.total_time_budget - elapsed
            print(f"[GlobalTimeManager] Game {self.current_game}/{self.total_games}, "
                  f"Time: {elapsed:.0f}s used, {remaining:.0f}s remaining, "
                  f"Decisions: {self.decisions_made}")
    
    def get_stats(self):
        """Get current statistics"""
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


# Global instance
global_time_manager = GlobalTimeManager()


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
        return False
    except Exception as e:
        signal.alarm(0)
        raise e
    finally:
        signal.signal(signal.SIGALRM, old_handler)


# ============ Reward Function ============
def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """Analyze shot result and calculate reward score"""
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


# ============ Global Dynamic Agent ============
class GlobalDynamicAgent:
    """Agent with global time budget management"""
    
    def __init__(self):
        """Initialize agent"""
        self.pbounds = {
            'V0': (0.5, 8.0),
            'phi': (0, 360),
            'theta': (0, 90), 
            'a': (-0.5, 0.5),
            'b': (-0.5, 0.5)
        }
        
        # Search parameters (wider range than per-game agent)
        self.MIN_INITIAL_SEARCH = 2
        self.MAX_INITIAL_SEARCH = 25  # Can afford more with global budget
        self.MIN_OPT_SEARCH = 1
        self.MAX_OPT_SEARCH = 15      # Can afford more with global budget
        self.MIN_CANDIDATES = 6
        self.MAX_CANDIDATES = 30      # More candidates when time allows
        
        self.ALPHA = 1e-2
        
        # Table dimensions
        self.TABLE_WIDTH = 1.12
        self.TABLE_LENGTH = 2.24
        self.BALL_RADIUS = 0.028575
        
        # Get global time manager
        self.time_manager = global_time_manager
        
        print("GlobalDynamicAgent (Global Time Budget) initialized.")
    
    
    def get_adaptive_search_params(self, time_budget, game_state, time_pressure):
        """Determine search parameters based on time budget and pressure"""
        # Base parameters on time budget
        if time_budget > 10.0:
            # Lots of time - maximize search
            initial_search = self.MAX_INITIAL_SEARCH
            opt_search = self.MAX_OPT_SEARCH
            n_candidates = self.MAX_CANDIDATES
        elif time_budget > 7.0:
            # Good time - thorough search
            initial_search = int(self.MAX_INITIAL_SEARCH * 0.8)
            opt_search = int(self.MAX_OPT_SEARCH * 0.8)
            n_candidates = int(self.MAX_CANDIDATES * 0.8)
        elif time_budget > 4.0:
            # Moderate time - balanced
            initial_search = int(self.MAX_INITIAL_SEARCH * 0.6)
            opt_search = int(self.MAX_OPT_SEARCH * 0.6)
            n_candidates = int(self.MAX_CANDIDATES * 0.6)
        elif time_budget > 2.0:
            # Limited time - reduced
            initial_search = int(self.MAX_INITIAL_SEARCH * 0.4)
            opt_search = int(self.MAX_OPT_SEARCH * 0.4)
            n_candidates = int(self.MAX_CANDIDATES * 0.5)
        elif time_budget > 1.0:
            # Low time - minimal
            initial_search = int(self.MAX_INITIAL_SEARCH * 0.2)
            opt_search = int(self.MAX_OPT_SEARCH * 0.2)
            n_candidates = int(self.MAX_CANDIDATES * 0.4)
        else:
            # Critical time - emergency mode
            initial_search = self.MIN_INITIAL_SEARCH
            opt_search = self.MIN_OPT_SEARCH
            n_candidates = self.MIN_CANDIDATES
        
        # Adjust for time pressure (global urgency)
        if time_pressure > 0.9:
            # Critical pressure - cut search significantly
            initial_search = max(self.MIN_INITIAL_SEARCH, initial_search // 3)
            opt_search = max(self.MIN_OPT_SEARCH, opt_search // 3)
            n_candidates = max(self.MIN_CANDIDATES, n_candidates // 2)
        elif time_pressure > 0.8:
            # High pressure - reduce search
            initial_search = max(self.MIN_INITIAL_SEARCH, initial_search // 2)
            opt_search = max(self.MIN_OPT_SEARCH, opt_search // 2)
            n_candidates = max(self.MIN_CANDIDATES, int(n_candidates * 0.7))
        
        # Adjust for game state
        n_remaining = game_state.get('n_remaining_balls', 7)
        if n_remaining <= 2 and time_pressure < 0.7:
            # Critical shots with time available - increase search
            initial_search = min(initial_search + 3, self.MAX_INITIAL_SEARCH)
            opt_search = min(opt_search + 2, self.MAX_OPT_SEARCH)
        
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
        """Get 2D position of ball"""
        return ball.state.rvw[0][:2]
    
    
    def _distance_2d(self, pos1, pos2):
        """Calculate 2D distance"""
        return np.linalg.norm(np.array(pos1) - np.array(pos2))
    
    
    def _angle_between_points(self, from_pos, to_pos):
        """Calculate angle in degrees"""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        angle = math.degrees(math.atan2(dy, dx))
        return angle % 360
    
    
    def _has_clear_path(self, from_pos, to_pos, balls, ignore_ids):
        """Check clear path"""
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
        """Calculate shot difficulty"""
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
        """Select best target ball"""
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
        """Calculate ghost ball position"""
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
        """Generate candidate shots"""
        cue_pos = self._get_ball_position(balls['cue'])
        target_id = self._select_best_target(balls, my_targets, table)
        
        if target_id is None:
            return [self._random_action() for _ in range(n_candidates)]
        
        target_pos = self._get_ball_position(balls[target_id])
        pocket_positions = {pid: pocket.center[:2] for pid, pocket in table.pockets.items()}
        
        candidates = []
        
        pocket_list = sorted(
            pocket_positions.items(),
            key=lambda x: self._distance_2d(target_pos, x[1])
        )
        
        for pocket_id, pocket_pos in pocket_list:
            ghost_pos = self._calculate_ghost_ball_position(target_pos, pocket_pos)
            phi = self._angle_between_points(cue_pos, ghost_pos)
            dist = self._distance_2d(cue_pos, target_pos)
            
            if dist < 0.5:
                V0_base = 2.0
            elif dist < 1.0:
                V0_base = 3.0
            elif dist < 1.5:
                V0_base = 4.0
            else:
                V0_base = 5.0
            
            for v_offset in [-0.8, -0.4, 0, 0.4, 0.8]:
                for phi_offset in [-10, -5, 0, 5, 10]:
                    for theta_val in [0.0, 5.0, 10.0]:
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
        """Make decision with global time management"""
        decision_start_time = time.time()
        
        if balls is None:
            return self._random_action()
        
        try:
            # Get global time budget and pressure
            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                my_targets = ["8"]
            
            game_state = {'n_remaining_balls': len(remaining_own)}
            
            time_budget = self.time_manager.get_time_budget(game_state)
            time_pressure = self.time_manager.get_time_pressure()
            remaining_time = self.time_manager.get_remaining_time()
            
            # Get adaptive parameters
            initial_search, opt_search, n_candidates = self.get_adaptive_search_params(
                time_budget, game_state, time_pressure
            )
            
            print(f"[GlobalAgent] Budget: {time_budget:.1f}s, Pressure: {time_pressure:.2f}, "
                  f"Remaining: {remaining_time:.0f}s, Search: {initial_search}+{opt_search}, "
                  f"Candidates: {n_candidates}")
            
            # Save state
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            
            # Generate candidates
            candidates = self._generate_candidate_shots(balls, my_targets, table, n_candidates)
            
            # Reward function
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
            
            # Evaluate candidates
            best_candidate = None
            best_candidate_score = -float('inf')
            
            eval_start = time.time()
            max_eval_time = time_budget * 0.3
            
            for i, candidate in enumerate(candidates):
                if time.time() - eval_start > max_eval_time:
                    break
                
                score = reward_fn_wrapper(**candidate)
                if score > best_candidate_score:
                    best_candidate_score = score
                    best_candidate = candidate
            
            # Decide whether to optimize
            remaining_decision_time = time_budget - (time.time() - decision_start_time)
            
            if best_candidate_score >= 50 and (time_pressure > 0.85 or remaining_decision_time < 1.5):
                # Good candidate + (high pressure OR low time) = use it
                action = best_candidate
                print(f"[GlobalAgent] Using candidate (score: {best_candidate_score:.2f}, pressure/time)")
            elif remaining_decision_time > 1.0:
                # Enough time for optimization
                print(f"[GlobalAgent] Optimizing...")
                seed = np.random.randint(1e6)
                optimizer = self._create_optimizer(reward_fn_wrapper, seed)
                
                # Seed with best candidates
                for candidate in candidates[:min(5, len(candidates))]:
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
                    print(f"[GlobalAgent] Optimized (score: {best_score:.2f})")
                else:
                    action = best_candidate
                    print(f"[GlobalAgent] Candidate better (score: {best_candidate_score:.2f})")
            else:
                # Not enough time
                if best_candidate_score >= 10:
                    action = best_candidate
                    print(f"[GlobalAgent] Using candidate (score: {best_candidate_score:.2f}, no time)")
                else:
                    action = self._random_action()
                    print(f"[GlobalAgent] Random (no good solution)")
            
            # Record decision time
            decision_time = time.time() - decision_start_time
            self.time_manager.record_decision(decision_time)
            
            print(f"[GlobalAgent] Decision took {decision_time:.2f}s")
            
            return action
        
        except Exception as e:
            print(f"[GlobalAgent] Error: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()



