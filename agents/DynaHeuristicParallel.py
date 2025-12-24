"""
DynaHeuristicParallel.py - Adaptive Multi-Core Parallel Agent

Key Features:
1. Automatic CPU core detection and utilization (default: 16 cores)
2. Parallel candidate evaluation using ProcessPoolExecutor
3. Self-learning time budget allocation
4. Adaptive time management based on historical performance
5. Safe time limit enforcement (never exceeds 3 minutes per game)

Strategy:
- Learn from historical decision times
- Dynamically adjust time budget to maximize search
- Use all available CPU cores for parallel evaluation
- Progressively aggressive time allocation with safety margins
- Aim for 85-95% time utilization without timeout risk
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


# ============ Adaptive Time Manager with Learning ============
class AdaptiveTimeManager:
    """Self-learning time manager that adapts to machine performance"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AdaptiveTimeManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        # Time tracking
        self.total_time_budget = 0.0
        self.start_time = None
        self.total_games = 0
        self.current_game = 0
        self.decisions_made = 0
        
        # Learning: historical decision time tracking
        self.decision_time_history = deque(maxlen=100)  # Last 100 decisions
        self.game_decision_counts = deque(maxlen=20)    # Last 20 games
        
        # Adaptive parameters
        self.time_safety_margin = 0.95  # Start conservative
        self.estimated_avg_decisions_per_game = 30.0
        self.predicted_decision_time = 5.0  # Initial estimate
        
        # Performance tracking
        self.games_completed = 0
        self.total_decisions = 0
        
    def initialize(self, n_games, time_per_game=180.0):
        """Initialize for new evaluation run"""
        self.total_time_budget = n_games * time_per_game
        self.start_time = time.time()
        self.total_games = n_games
        self.current_game = 0
        self.decisions_made = 0
        self.games_completed = 0
        
        # Reset learning data for new run
        self.decision_time_history.clear()
        self.game_decision_counts.clear()
        self.time_safety_margin = 0.95  # Start conservative
        
        print(f"[AdaptiveTimeManager] Initialized with {self.total_time_budget:.0f}s budget for {n_games} games")
        print(f"[AdaptiveTimeManager] Target time utilization: 85-95%")
    
    def learn_from_decision(self, decision_time):
        """Learn from completed decision"""
        self.decision_time_history.append(decision_time)
        self.decisions_made += 1
        self.total_decisions += 1
        
        # Update predicted decision time (exponential moving average)
        if len(self.decision_time_history) >= 5:
            recent_avg = np.mean(list(self.decision_time_history)[-10:])
            self.predicted_decision_time = 0.7 * self.predicted_decision_time + 0.3 * recent_avg
        
        # Adapt safety margin based on utilization
        if self.decisions_made >= 20:
            self._adapt_safety_margin()
    
    def _adapt_safety_margin(self):
        """Adaptively adjust safety margin based on performance"""
        if self.start_time is None:
            return
        
        elapsed = time.time() - self.start_time
        utilization = elapsed / self.total_time_budget
        
        # If we're using too little time, become more aggressive
        if utilization < 0.6 and self.games_completed > 2:
            self.time_safety_margin = min(0.98, self.time_safety_margin + 0.01)
        # If we're cutting it close, become more conservative
        elif utilization > 0.92:
            self.time_safety_margin = max(0.90, self.time_safety_margin - 0.02)
    
    def end_game(self, decisions_in_game):
        """Record game completion"""
        self.game_decision_counts.append(decisions_in_game)
        self.games_completed += 1
        self.current_game += 1
        self.decisions_made = 0  # Reset for next game
        
        # Update average decisions per game estimate
        if len(self.game_decision_counts) >= 3:
            self.estimated_avg_decisions_per_game = np.mean(self.game_decision_counts)
    
    def get_time_budget(self, game_state=None):
        """Calculate adaptive time budget for current decision"""
        if self.start_time is None:
            return 8.0
        
        elapsed = time.time() - self.start_time
        remaining = self.total_time_budget - elapsed
        
        # Apply adaptive safety margin
        safe_remaining = remaining * self.time_safety_margin
        
        if safe_remaining <= 0:
            return 0.3
        
        # Estimate remaining decisions
        games_remaining = max(1, self.total_games - self.current_game)
        
        # Use learned average if available
        if len(self.game_decision_counts) >= 3:
            decisions_in_current_game = self.decisions_made
            estimated_remaining_in_game = max(1, self.estimated_avg_decisions_per_game - decisions_in_current_game)
            estimated_decisions_remaining = estimated_remaining_in_game + (games_remaining - 1) * self.estimated_avg_decisions_per_game
        else:
            estimated_decisions_remaining = games_remaining * 30
        
        # Base budget allocation
        base_budget = safe_remaining / max(1, estimated_decisions_remaining)
        
        # Adjust based on game state complexity
        complexity_multiplier = 1.0
        if game_state:
            n_remaining = game_state.get('n_remaining_balls', 7)
            if n_remaining <= 2:
                complexity_multiplier = 1.5  # Critical endgame
            elif n_remaining >= 6:
                complexity_multiplier = 0.8  # Early game
        
        # Progressive time allocation: use more time early if we have budget
        utilization = elapsed / self.total_time_budget if self.total_time_budget > 0 else 0
        if utilization < 0.3 and games_remaining > 5:
            # We have plenty of time, be generous
            complexity_multiplier *= 1.3
        elif utilization > 0.8:
            # Running low on time, be conservative
            complexity_multiplier *= 0.7
        
        time_budget = base_budget * complexity_multiplier
        
        # Clamp to reasonable range
        return max(0.3, min(20.0, time_budget))
    
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
        """Notify start of new game"""
        if self.decisions_made > 0:
            self.end_game(self.decisions_made)
        
        if self.current_game % 10 == 0 and self.start_time:
            elapsed = time.time() - self.start_time
            remaining = self.total_time_budget - elapsed
            utilization = elapsed / self.total_time_budget * 100
            print(f"[AdaptiveTimeManager] Game {self.current_game}/{self.total_games}")
            print(f"  Time: {elapsed:.0f}s used ({utilization:.1f}%), {remaining:.0f}s remaining")
            print(f"  Decisions: {self.total_decisions}, Avg: {self.predicted_decision_time:.2f}s/decision")
            print(f"  Safety margin: {self.time_safety_margin:.2%}")
    
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


# Global instance
adaptive_time_manager = AdaptiveTimeManager()


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


# ============ Parallel Evaluation Helper ============
def evaluate_candidate_action(args):
    """Worker function for parallel candidate evaluation"""
    candidate, balls_state, table_state, last_state, player_targets = args
    
    try:
        # Reconstruct objects from serialized state
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls_state.items()}
        sim_table = copy.deepcopy(table_state)
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
        
        shot.cue.set_state(**candidate)
        
        if not simulate_with_timeout(shot, timeout=2):
            return (candidate, -50)
        
        # Calculate reward
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
        
        return (candidate, score)
        
    except Exception as e:
        return (candidate, -500)


# ============ Parallel Dynamic Agent ============
class ParallelDynamicAgent:
    """Multi-core parallel agent with adaptive time management"""
    
    def __init__(self, n_cores=None):
        """Initialize agent
        
        Args:
            n_cores: Number of CPU cores to use. If None, auto-detect (default 16)
        """
        # CPU core detection
        if n_cores is None:
            try:
                n_cores = len(os.sched_getaffinity(0))  # Available cores
            except:
                n_cores = os.cpu_count() or 16  # Total cores or default 16
        
        self.n_cores = min(n_cores, 32)  # Cap at 32 to avoid overhead
        
        # Search parameters - more aggressive since we have parallel processing
        self.MIN_INITIAL_SEARCH = 3
        self.MAX_INITIAL_SEARCH = 40   # Higher than non-parallel version
        self.MIN_OPT_SEARCH = 2
        self.MAX_OPT_SEARCH = 25       # Higher than non-parallel version
        self.MIN_CANDIDATES = 12       # More candidates for parallel eval
        self.MAX_CANDIDATES = 80       # Much higher with parallel
        
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
        
        self.time_manager = adaptive_time_manager
        
        # Process pool for parallel evaluation (created on demand)
        self.executor = None
        
        print(f"ParallelDynamicAgent initialized with {self.n_cores} CPU cores")
        print(f"  Max search: Init={self.MAX_INITIAL_SEARCH}, Opt={self.MAX_OPT_SEARCH}, Candidates={self.MAX_CANDIDATES}")
    
    def __del__(self):
        """Cleanup process pool"""
        if self.executor is not None:
            self.executor.shutdown(wait=False)
    
    def get_adaptive_search_params(self, time_budget, game_state, time_pressure):
        """Determine search parameters based on available time and resources"""
        # With parallel processing, we can afford more search
        if time_budget > 12.0:
            initial_search = self.MAX_INITIAL_SEARCH
            opt_search = self.MAX_OPT_SEARCH
            n_candidates = self.MAX_CANDIDATES
        elif time_budget > 8.0:
            initial_search = int(self.MAX_INITIAL_SEARCH * 0.8)
            opt_search = int(self.MAX_OPT_SEARCH * 0.8)
            n_candidates = int(self.MAX_CANDIDATES * 0.8)
        elif time_budget > 5.0:
            initial_search = int(self.MAX_INITIAL_SEARCH * 0.6)
            opt_search = int(self.MAX_OPT_SEARCH * 0.6)
            n_candidates = int(self.MAX_CANDIDATES * 0.6)
        elif time_budget > 3.0:
            initial_search = int(self.MAX_INITIAL_SEARCH * 0.4)
            opt_search = int(self.MAX_OPT_SEARCH * 0.4)
            n_candidates = int(self.MAX_CANDIDATES * 0.5)
        elif time_budget > 1.5:
            initial_search = int(self.MAX_INITIAL_SEARCH * 0.25)
            opt_search = int(self.MAX_OPT_SEARCH * 0.3)
            n_candidates = int(self.MAX_CANDIDATES * 0.4)
        else:
            initial_search = self.MIN_INITIAL_SEARCH
            opt_search = self.MIN_OPT_SEARCH
            n_candidates = self.MIN_CANDIDATES
        
        # Adjust for time pressure
        if time_pressure > 0.9:
            initial_search = max(self.MIN_INITIAL_SEARCH, initial_search // 2)
            opt_search = max(self.MIN_OPT_SEARCH, opt_search // 2)
            n_candidates = max(self.MIN_CANDIDATES, int(n_candidates * 0.6))
        elif time_pressure > 0.8:
            initial_search = max(self.MIN_INITIAL_SEARCH, int(initial_search * 0.7))
            opt_search = max(self.MIN_OPT_SEARCH, int(opt_search * 0.7))
            n_candidates = max(self.MIN_CANDIDATES, int(n_candidates * 0.75))
        
        # Adjust for game state
        n_remaining = game_state.get('n_remaining_balls', 7)
        if n_remaining <= 2 and time_pressure < 0.7:
            initial_search = min(initial_search + 5, self.MAX_INITIAL_SEARCH)
            opt_search = min(opt_search + 3, self.MAX_OPT_SEARCH)
        
        initial_search = max(initial_search, self.MIN_INITIAL_SEARCH)
        opt_search = max(opt_search, self.MIN_OPT_SEARCH)
        n_candidates = max(n_candidates, self.MIN_CANDIDATES)
        
        return initial_search, opt_search, n_candidates
    
    def _random_action(self):
        import random
        return {
            'V0': round(random.uniform(0.5, 8.0), 2),
            'phi': round(random.uniform(0, 360), 2),
            'theta': round(random.uniform(0, 90), 2),
            'a': round(random.uniform(-0.5, 0.5), 3),
            'b': round(random.uniform(-0.5, 0.5), 3)
        }
    
    def _get_ball_position(self, ball):
        return ball.state.rvw[0][:2]
    
    def _distance_2d(self, pos1, pos2):
        return np.linalg.norm(np.array(pos1) - np.array(pos2))
    
    def _angle_between_points(self, from_pos, to_pos):
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        angle = math.degrees(math.atan2(dy, dx))
        return angle % 360
    
    def _has_clear_path(self, from_pos, to_pos, balls, ignore_ids):
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
    
    def _select_best_targets(self, balls, my_targets, table, top_n=3):
        """Select top N best target balls"""
        cue_pos = self._get_ball_position(balls['cue'])
        remaining_targets = [bid for bid in my_targets if balls[bid].state.s != 4]
        
        if not remaining_targets:
            return []
        
        pocket_positions = {pid: pocket.center[:2] for pid, pocket in table.pockets.items()}
        
        target_scores = []
        for target_id in remaining_targets:
            target_pos = self._get_ball_position(balls[target_id])
            min_difficulty = float('inf')
            
            for pocket_id, pocket_pos in pocket_positions.items():
                difficulty = self._calculate_shot_difficulty(
                    cue_pos, target_pos, pocket_pos, balls, target_id
                )
                min_difficulty = min(min_difficulty, difficulty)
            
            target_scores.append((target_id, min_difficulty))
        
        target_scores.sort(key=lambda x: x[1])
        return [tid for tid, _ in target_scores[:top_n]]
    
    def _calculate_ghost_ball_position(self, target_pos, pocket_pos):
        target_pos = np.array(target_pos)
        pocket_pos = np.array(pocket_pos)
        
        direction = target_pos - pocket_pos
        dist = np.linalg.norm(direction)
        
        if dist < 1e-6:
            return target_pos
        
        direction = direction / dist
        ghost_pos = target_pos + direction * (2 * self.BALL_RADIUS)
        
        return ghost_pos
    
    def _generate_candidate_shots(self, balls, my_targets, table, n_candidates=40):
        """Generate diverse candidates from multiple targets"""
        cue_pos = self._get_ball_position(balls['cue'])
        
        # Select multiple best targets for diversity
        top_targets = self._select_best_targets(balls, my_targets, table, top_n=min(4, len([bid for bid in my_targets if balls[bid].state.s != 4])))
        
        if not top_targets:
            return [self._random_action() for _ in range(n_candidates)]
        
        pocket_positions = {pid: pocket.center[:2] for pid, pocket in table.pockets.items()}
        candidates = []
        
        candidates_per_target = max(1, n_candidates // len(top_targets))
        
        for target_id in top_targets:
            target_pos = self._get_ball_position(balls[target_id])
            
            pocket_list = sorted(
                pocket_positions.items(),
                key=lambda x: self._distance_2d(target_pos, x[1])
            )
            
            for pocket_id, pocket_pos in pocket_list[:3]:  # Top 3 pockets
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
                                return candidates[:n_candidates]
        
        while len(candidates) < n_candidates:
            candidates.append(self._random_action())
        
        return candidates[:n_candidates]
    
    def _evaluate_candidates_parallel(self, candidates, balls, table, last_state, my_targets, timeout):
        """Evaluate candidates in parallel using process pool"""
        if self.executor is None:
            self.executor = ProcessPoolExecutor(max_workers=self.n_cores)
        
        # Prepare arguments for parallel evaluation
        eval_args = [
            (candidate, balls, table, last_state, my_targets)
            for candidate in candidates
        ]
        
        try:
            # Submit all tasks and wait with timeout
            futures = [self.executor.submit(evaluate_candidate_action, arg) for arg in eval_args]
            
            results = []
            for future in futures:
                try:
                    result = future.result(timeout=timeout/len(candidates) + 1.0)
                    results.append(result)
                except FutureTimeoutError:
                    results.append((None, -100))
                except Exception:
                    results.append((None, -500))
            
            return results
            
        except Exception as e:
            print(f"[ParallelAgent] Parallel evaluation error: {e}")
            return [(c, -500) for c in candidates]
    
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
        """Make decision with parallel evaluation and adaptive time management"""
        decision_start_time = time.time()
        
        if balls is None:
            return self._random_action()
        
        try:
            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                my_targets = ["8"]
            
            game_state = {'n_remaining_balls': len(remaining_own)}
            
            time_budget = self.time_manager.get_time_budget(game_state)
            time_pressure = self.time_manager.get_time_pressure()
            remaining_time = self.time_manager.get_remaining_time()
            
            initial_search, opt_search, n_candidates = self.get_adaptive_search_params(
                time_budget, game_state, time_pressure
            )
            
            print(f"[ParallelAgent] Budget: {time_budget:.1f}s, Pressure: {time_pressure:.2f}, "
                  f"Remaining: {remaining_time:.0f}s")
            print(f"  Search: {initial_search}+{opt_search}, Candidates: {n_candidates}, Cores: {self.n_cores}")
            
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            
            # Generate candidates
            candidates = self._generate_candidate_shots(balls, my_targets, table, n_candidates)
            
            # Parallel evaluation of candidates
            eval_start = time.time()
            max_eval_time = time_budget * 0.4  # Use 40% of time for candidate eval
            
            print(f"[ParallelAgent] Evaluating {len(candidates)} candidates in parallel...")
            results = self._evaluate_candidates_parallel(
                candidates, balls, table, last_state_snapshot, my_targets,
                timeout=max_eval_time
            )
            
            # Find best candidate
            best_candidate = None
            best_candidate_score = -float('inf')
            
            for candidate, score in results:
                if candidate is not None and score > best_candidate_score:
                    best_candidate_score = score
                    best_candidate = candidate
            
            eval_time = time.time() - eval_start
            print(f"[ParallelAgent] Parallel eval took {eval_time:.2f}s, best score: {best_candidate_score:.2f}")
            
            remaining_decision_time = time_budget - (time.time() - decision_start_time)
            
            # Decide whether to optimize further
            if best_candidate_score >= 60:
                action = best_candidate
                print(f"[ParallelAgent] Excellent candidate, using directly")
            elif best_candidate_score >= 45 and remaining_decision_time < 2.0:
                action = best_candidate
                print(f"[ParallelAgent] Good candidate + low time")
            elif remaining_decision_time > 1.5:
                print(f"[ParallelAgent] Running Bayesian optimization...")
                
                # Define reward function (will use sequential evaluation for BO)
                def reward_fn_wrapper(V0, phi, theta, a, b):
                    candidate = {'V0': V0, 'phi': phi, 'theta': theta, 'a': a, 'b': b}
                    results = self._evaluate_candidates_parallel(
                        [candidate], balls, table, last_state_snapshot, my_targets,
                        timeout=2.0
                    )
                    return results[0][1] if results else -500
                
                seed = np.random.randint(1e6)
                optimizer = self._create_optimizer(reward_fn_wrapper, seed)
                
                # Seed with best candidates
                for candidate, score in sorted(results, key=lambda x: x[1], reverse=True)[:5]:
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
                    print(f"[ParallelAgent] Optimized (score: {best_score:.2f})")
                else:
                    action = best_candidate
                    print(f"[ParallelAgent] Candidate better (score: {best_candidate_score:.2f})")
            else:
                if best_candidate_score >= 10:
                    action = best_candidate
                    print(f"[ParallelAgent] Using candidate (no time for optimization)")
                else:
                    action = self._random_action()
                    print(f"[ParallelAgent] Random (no good solution found)")
            
            decision_time = time.time() - decision_start_time
            self.time_manager.learn_from_decision(decision_time)
            
            print(f"[ParallelAgent] Decision took {decision_time:.2f}s")
            
            return action
        
        except Exception as e:
            print(f"[ParallelAgent] Error: {e}")
            import traceback
            traceback.print_exc()
            
            # Still record time even on error
            decision_time = time.time() - decision_start_time
            self.time_manager.learn_from_decision(decision_time)
            
            return self._random_action()




