"""
Heuristic.py - Fast Heuristic-Based Agent with Limited Search

This agent uses domain knowledge and efficient search to make quick decisions
while maintaining competitive performance against BasicAgent.

Key Features:
1. Smart target ball selection based on difficulty
2. Geometric shot calculation using "ghost ball" method
3. Reduced Bayesian Optimization search space
4. Fast decision making (<1 second per shot)
"""

import math
import numpy as np
import pooltool as pt
import copy
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
    """
    Analyze shot result and calculate reward score
    
    Returns:
        float: Reward score
            +50/ball (own pocketed), +100 (legal black 8), +10 (legal no pocket)
            -100 (cue pocketed), -150 (illegal black 8), -30 (foul)
    """
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
    
    # First ball foul check
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


# ============ Heuristic Agent ============
class HeuristicAgent:
    """Fast Heuristic-Based Agent with Limited Search"""
    
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
        
        # Optimization parameters (reduced from BasicAgent)
        self.INITIAL_SEARCH = 5   # Reduced from 20
        self.OPT_SEARCH = 3       # Reduced from 10
        self.ALPHA = 1e-2
        
        # Table dimensions (standard 8-ball table)
        self.TABLE_WIDTH = 1.12  # meters
        self.TABLE_LENGTH = 2.24  # meters
        self.BALL_RADIUS = 0.028575  # meters
        
        print("HeuristicAgent (Fast, Heuristic-based) initialized.")

    
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
        
        # Check all balls not in ignore list
        for bid, ball in balls.items():
            if bid in ignore_ids or ball.state.s == 4:  # Skip ignored or pocketed balls
                continue
            
            ball_pos = self._get_ball_position(ball)
            
            # Vector from line start to ball
            to_ball = ball_pos - from_pos
            
            # Project onto path direction
            projection = np.dot(to_ball, path_dir)
            
            # Check if projection is within path segment
            if projection < 0 or projection > path_length:
                continue
            
            # Calculate perpendicular distance
            closest_point = from_pos + projection * path_dir
            dist = np.linalg.norm(ball_pos - closest_point)
            
            # If ball blocks path (within 2 ball radii)
            if dist < 2.5 * self.BALL_RADIUS:
                return False
        
        return True
    
    
    def _calculate_shot_difficulty(self, cue_pos, target_pos, pocket_pos, balls, target_id):
        """Calculate difficulty score for a shot (lower is easier)"""
        difficulty = 0
        
        # Distance from cue to target
        cue_to_target_dist = self._distance_2d(cue_pos, target_pos)
        difficulty += cue_to_target_dist * 10  # Weight: 10
        
        # Distance from target to pocket
        target_to_pocket_dist = self._distance_2d(target_pos, pocket_pos)
        difficulty += target_to_pocket_dist * 20  # Weight: 20 (more important)
        
        # Check clear path from cue to target
        if not self._has_clear_path(cue_pos, target_pos, balls, ['cue', target_id]):
            difficulty += 50  # Penalty for blocked path
        
        # Calculate cut angle (0 = straight shot, 90 = maximum cut)
        cue_to_target_angle = self._angle_between_points(cue_pos, target_pos)
        target_to_pocket_angle = self._angle_between_points(target_pos, pocket_pos)
        cut_angle = abs(target_to_pocket_angle - cue_to_target_angle)
        cut_angle = min(cut_angle, 360 - cut_angle)  # Take smaller angle
        
        # Penalize difficult cut angles (>60 degrees)
        if cut_angle > 60:
            difficulty += (cut_angle - 60) * 2
        
        return difficulty
    
    
    def _select_best_target(self, balls, my_targets, table):
        """Select the best target ball to shoot at"""
        cue_pos = self._get_ball_position(balls['cue'])
        
        # Get remaining target balls
        remaining_targets = [bid for bid in my_targets if balls[bid].state.s != 4]
        
        if not remaining_targets:
            return None
        
        # Get all pocket positions
        pocket_positions = {pid: pocket.center[:2] for pid, pocket in table.pockets.items()}
        
        best_target = None
        best_score = float('inf')
        
        for target_id in remaining_targets:
            target_pos = self._get_ball_position(balls[target_id])
            
            # Find easiest pocket for this target
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
        
        # Direction from pocket to target
        direction = target_pos - pocket_pos
        dist = np.linalg.norm(direction)
        
        if dist < 1e-6:
            return target_pos
        
        direction = direction / dist
        
        # Ghost ball is 2 ball radii away from target (opposite to pocket)
        ghost_pos = target_pos + direction * (2 * self.BALL_RADIUS)
        
        return ghost_pos
    
    
    def _generate_candidate_shots(self, balls, my_targets, table, n_candidates=10):
        """Generate candidate shots using heuristics"""
        cue_pos = self._get_ball_position(balls['cue'])
        
        # Select best target
        target_id = self._select_best_target(balls, my_targets, table)
        
        if target_id is None:
            # No targets left, return random candidates
            return [self._random_action() for _ in range(n_candidates)]
        
        target_pos = self._get_ball_position(balls[target_id])
        
        # Get all pockets
        pocket_positions = {pid: pocket.center[:2] for pid, pocket in table.pockets.items()}
        
        candidates = []
        
        # Generate shots for each pocket
        for pocket_id, pocket_pos in pocket_positions.items():
            # Calculate ghost ball position
            ghost_pos = self._calculate_ghost_ball_position(target_pos, pocket_pos)
            
            # Calculate angle from cue to ghost ball
            phi = self._angle_between_points(cue_pos, ghost_pos)
            
            # Calculate distance to determine speed
            dist = self._distance_2d(cue_pos, target_pos)
            
            # Speed heuristic: slower for close shots, faster for far shots
            if dist < 0.5:
                V0_base = 2.0
            elif dist < 1.0:
                V0_base = 3.0
            elif dist < 1.5:
                V0_base = 4.0
            else:
                V0_base = 5.0
            
            # Generate variations around the base shot
            for v_offset in [-0.5, 0, 0.5]:
                for phi_offset in [-5, 0, 5]:
                    V0 = np.clip(V0_base + v_offset, 0.5, 8.0)
                    phi_adjusted = (phi + phi_offset) % 360
                    
                    candidate = {
                        'V0': V0,
                        'phi': phi_adjusted,
                        'theta': 0.0,  # Top spin for control
                        'a': 0.0,      # Center hit
                        'b': 0.0       # Center hit
                    }
                    candidates.append(candidate)
                    
                    if len(candidates) >= n_candidates:
                        break
                if len(candidates) >= n_candidates:
                    break
            if len(candidates) >= n_candidates:
                break
        
        # Fill remaining with random if needed
        while len(candidates) < n_candidates:
            candidates.append(self._random_action())
        
        return candidates[:n_candidates]
    
    
    def _create_optimizer(self, reward_function, seed):
        """Create Bayesian optimizer"""
        gpr = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=self.ALPHA,
            n_restarts_optimizer=5,  # Reduced from 10
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
        """Make decision using heuristics and limited search"""
        if balls is None:
            print(f"[HeuristicAgent] No ball information, using random action.")
            return self._random_action()
        
        try:
            # Save state snapshot
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            
            # Check if all targets are pocketed
            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                my_targets = ["8"]
                print("[HeuristicAgent] All targets cleared, switching to black 8")
            
            # Generate smart candidates
            print(f"[HeuristicAgent] Generating candidate shots for targets: {my_targets}")
            candidates = self._generate_candidate_shots(balls, my_targets, table, n_candidates=8)
            
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
            
            # Evaluate candidates quickly
            best_candidate = None
            best_candidate_score = -float('inf')
            
            print(f"[HeuristicAgent] Evaluating {len(candidates)} candidates...")
            for candidate in candidates:
                score = reward_fn_wrapper(**candidate)
                if score > best_candidate_score:
                    best_candidate_score = score
                    best_candidate = candidate
            
            # If we found a good candidate, use it
            if best_candidate_score >= 10:
                print(f"[HeuristicAgent] Found good shot (score: {best_candidate_score:.2f})")
                print(f"[HeuristicAgent] Decision: V0={best_candidate['V0']:.2f}, "
                      f"phi={best_candidate['phi']:.2f}, theta={best_candidate['theta']:.2f}")
                return best_candidate
            
            # Otherwise, do limited Bayesian optimization
            print(f"[HeuristicAgent] Running limited optimization...")
            seed = np.random.randint(1e6)
            optimizer = self._create_optimizer(reward_fn_wrapper, seed)
            
            # Seed optimizer with best candidates
            for candidate in candidates[:3]:
                try:
                    optimizer.probe(params=candidate, lazy=True)
                except:
                    pass
            
            optimizer.maximize(
                init_points=self.INITIAL_SEARCH,
                n_iter=self.OPT_SEARCH
            )
            
            best_result = optimizer.max
            best_params = best_result['params']
            best_score = best_result['target']
            
            if best_score < 10:
                print(f"[HeuristicAgent] No good solution found (best: {best_score:.2f}). Using random.")
                return self._random_action()
            
            action = {
                'V0': float(best_params['V0']),
                'phi': float(best_params['phi']),
                'theta': float(best_params['theta']),
                'a': float(best_params['a']),
                'b': float(best_params['b']),
            }
            
            print(f"[HeuristicAgent] Decision (score: {best_score:.2f}): "
                  f"V0={action['V0']:.2f}, phi={action['phi']:.2f}, "
                  f"theta={action['theta']:.2f}, a={action['a']:.3f}, b={action['b']:.3f}")
            return action
        
        except Exception as e:
            print(f"[HeuristicAgent] Error during decision: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()



