"""
Brain Engine Module
===================
PPO-based RL decision engine with hybrid traffic rules:
  1. Emergency Override  (highest priority)
  2. RL Brain prediction (hold / switch)
  3. Smart Hold          (others empty → stay)
  4. Empty Lane Skip     (current empty → jump)
  5. Max Phase Timer     (force switch after N seconds)

Supports both v1 (6-dim obs) and v2 (14-dim obs) models.
Uses wall-clock time so behaviour is independent of video FPS.
"""

import os
import time
import numpy as np
from stable_baselines3 import PPO


class BrainEngine:
    """RL traffic-light decision engine."""

    def __init__(self, model_path="ppo_traffic_brain", max_phase_seconds=60, min_phase_seconds=3, yellow_seconds=1.5):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Auto-detect v2 model (14-dim advanced reward), fallback to v1 (6-dim)
        v2_path = os.path.join(script_dir, model_path + "_v2")
        v1_path = os.path.join(script_dir, model_path)
        
        if os.path.exists(v2_path + ".zip"):
            full_path = v2_path
            self.obs_version = 2
            print(f"  Loading RL brain v2 (Advanced) from {full_path} ...")
        else:
            full_path = v1_path
            self.obs_version = 1
            print(f"  Loading RL brain v1 (Original) from {full_path} ...")

        self.brain = PPO.load(full_path, device='cpu')
        self.max_phase_seconds = max_phase_seconds
        self.min_phase_seconds = min_phase_seconds  # Prevents insane flickering
        self.yellow_seconds = yellow_seconds

        # State
        self.current_phase = 3
        self.phase_start_time = time.time()
        self.transitioning_to = -1
        self.yellow_start_time = 0
        
        self.total_switches = 0
        self.total_emergencies = 0
        self.locked_green_time = self.min_phase_seconds  # الوقت الأخضر المثبّت لهذا المسار
        
        # v2: تتبع وقت الانتظار التراكمي التقريبي لكل مسار
        self.accumulated_wait_estimate = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        self.last_lane_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    # ------------------------------------------------------------------
    @property
    def time_on_phase(self):
        return time.time() - self.phase_start_time

    # ------------------------------------------------------------------
    def decide(self, lane_counts, emergency_lane=-1):
        """
        Return decision dict:
            phase   – new / current green-lane index
            status  – short human-readable string
            reason  – detailed explanation
            color   – BGR tuple for the HUD
        """
        elapsed = self.time_on_phase
        # Map real seconds → integer steps the model understands (1 step ≈ 10 s)
        time_steps = min(int(elapsed / 10), 100)

        # تحديث تقدير وقت الانتظار التراكمي
        # السيارات الواقفة تتراكم عليها ثوانٍ الانتظار مع مرور الوقت
        for i in range(4):
            if i != self.current_phase:  # المسارات التي إشارتها حمراء
                self.accumulated_wait_estimate[i] += lane_counts.get(i, 0) * 1.0
            else:
                # المسار الأخضر: تقليل وقت الانتظار تدريجياً
                self.accumulated_wait_estimate[i] = max(0, self.accumulated_wait_estimate[i] * 0.5)
        
        if self.obs_version == 2:
            # === v2: فضاء مراقبة موسّع 14 بُعد ===
            obs = np.array([
                lane_counts.get(0, 0),   # [0-3] الطوابير
                lane_counts.get(1, 0),
                lane_counts.get(2, 0),
                lane_counts.get(3, 0),
                0, 0, 0, 0,              # [4-7] المقتربات (تقدير بسيط من الفرق)
                self.accumulated_wait_estimate[0],  # [8-11] وقت الانتظار التراكمي
                self.accumulated_wait_estimate[1],
                self.accumulated_wait_estimate[2],
                self.accumulated_wait_estimate[3],
                self.current_phase,       # [12] الإشارة الحالية
                time_steps,              # [13] مدة البقاء
            ], dtype=np.float32)
            
            # تقدير المقتربات: الفرق بين القراءة الحالية والسابقة
            for i in range(4):
                diff = lane_counts.get(i, 0) - self.last_lane_counts.get(i, 0)
                obs[4 + i] = max(0, diff)  # السيارات الجديدة = مقتربات
            
            self.last_lane_counts = dict(lane_counts)
        else:
            # === v1: فضاء مراقبة أصلي 6 أبعاد ===
            obs = np.array([
                lane_counts.get(0, 0),
                lane_counts.get(1, 0),
                lane_counts.get(2, 0),
                lane_counts.get(3, 0),
                self.current_phase,
                time_steps,
            ], dtype=np.int32)

        # Check if we are currently in a Yellow phase transition
        if self.transitioning_to != -1:
            yellow_elapsed = time.time() - self.yellow_start_time
            if yellow_elapsed >= self.yellow_seconds:
                # Transition complete: turn new lane Green
                self.current_phase = self.transitioning_to
                self.transitioning_to = -1
                self.phase_start_time = time.time()
                self.total_switches += 1
                # === تثبيت الوقت الأخضر لحظة الفتح ===
                q_new = lane_counts.get(self.current_phase, 0)
                self.locked_green_time = min(
                    self.min_phase_seconds + q_new * 2.0,
                    self.max_phase_seconds
                )
                return self._result(f"GREEN ({self.locked_green_time:.0f}s | {q_new} cars)",
                                    f"Locked green for {q_new} cars",
                                    (0, 255, 0))
            else:
                # Still in Yellow phase
                return self._result(f"YELLOW PHASE ({self.yellow_seconds - yellow_elapsed:.1f}s)",
                                    f"Switching to Lane {self.transitioning_to}",
                                    (0, 255, 255), is_yellow=True)

        # === 1. Emergency Override ====================================
        if emergency_lane != -1:
            if emergency_lane != self.current_phase:
                # Start yellow transition to emergency lane immediately
                self.transitioning_to = emergency_lane
                self.yellow_start_time = time.time()
                self.total_emergencies += 1
                return self._result("EMERGENCY PREP",
                                    f"Stopping traffic for Lane {emergency_lane}",
                                    (0, 255, 255), is_yellow=True)
            return self._result("EMERGENCY OVERRIDE",
                                f"Emergency in Lane {emergency_lane}",
                                (0, 0, 255))

        # === استخدام الوقت الأخضر المثبّت ===
        q_cur = lane_counts.get(self.current_phase, 0)
        
        # Enforce Locked Green Time
        # إذا صار عدد السيارات 0 → صفّر الوقت وانتقل فوراً
        if q_cur == 0 and elapsed >= self.min_phase_seconds:
            self.locked_green_time = 0  # تصفير الوقت
        
        if elapsed < self.locked_green_time:
            remaining = self.locked_green_time - elapsed
            return self._result(f"GREEN ({remaining:.0f}s | {q_cur} cars)",
                                f"Locked green: {self.locked_green_time:.0f}s",
                                (0, 255, 0))

        # === 2. RL Brain ==============================================
        action, _ = self.brain.predict(obs, deterministic=True)

        q_others = sum(lane_counts.get(i, 0) for i in range(4)) - q_cur
        should_switch = (action == 1) or (elapsed >= self.max_phase_seconds)
        
        # Log the AI's internal thought process to the terminal
        if int(elapsed * 10) % 10 == 0:  
            print(f"  [AI] Phase:{self.current_phase} | Queues:{obs[:4]} | Action:{'SWITCH' if action==1 else 'HOLD'} | GreenTime:{self.locked_green_time:.0f}s")

        if should_switch:
            # === 3. Smart Hold ========================================
            if q_others == 0 and q_cur > 0:
                self.phase_start_time = time.time()
                return self._result("SMART HOLD",
                                    "Other lanes empty – holding",
                                    (0, 200, 200))

            # === 4. Switch to BUSIEST lane ============================
            next_p = self._busiest_lane(lane_counts)
            if next_p == -1:  # All other lanes are empty
                self.phase_start_time = time.time()
                return self._result("HOLD (Others Empty)", "No cars waiting elsewhere", (0, 200, 200))

            # Start Yellow Transition
            self.transitioning_to = next_p
            self.yellow_start_time = time.time()
            reason = (f"Max time reached" if elapsed >= self.max_phase_seconds
                      else f"AI switching to Lane {next_p} ({lane_counts.get(next_p,0)} cars)")
            return self._result(f"YELLOW PHASE ({self.yellow_seconds:.1f}s)",
                                reason, (0, 255, 255), is_yellow=True)
        else:
            # === 5. Imbalance Override =================================
            busiest = self._busiest_lane(lane_counts)
            if busiest != -1:
                q_busiest = lane_counts.get(busiest, 0)
                # إذا المسار الأحمر فيه أكثر من 4 سيارات زيادة عن الأخضر
                if q_busiest > q_cur + 4 and elapsed >= self.locked_green_time:
                    self.transitioning_to = busiest
                    self.yellow_start_time = time.time()
                    return self._result(f"YELLOW PHASE ({self.yellow_seconds:.1f}s)",
                                        f"Imbalance! Lane {busiest} has {q_busiest} cars",
                                        (0, 255, 255), is_yellow=True)
            
            # === 6. Skip if current lane empty ========================
            if q_cur == 0 and q_others > 0:
                next_p = self._busiest_lane(lane_counts)
                if next_p != -1:
                    self.transitioning_to = next_p
                    self.yellow_start_time = time.time()
                    return self._result(f"YELLOW PHASE ({self.yellow_seconds:.1f}s)",
                                        f"Current lane empty, switching to Lane {next_p}",
                                        (0, 255, 255), is_yellow=True)

            remaining = max(0, self.max_phase_seconds - elapsed)
            return self._result(f"HOLDING GREEN ({remaining:.0f}s)",
                                f"AI holding (L{self.current_phase}:{q_cur} cars)",
                                (0, 255, 0))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _busiest_lane(self, lane_counts):
        """Find the lane with the MOST cars waiting (excluding current green). Returns -1 if all empty."""
        best_lane = -1
        best_count = 0
        for i in range(4):
            if i != self.current_phase:
                count = lane_counts.get(i, 0)
                if count > best_count:
                    best_count = count
                    best_lane = i
        return best_lane

    def _result(self, status, reason, color, is_yellow=False):
        return {
            'phase': self.current_phase,
            'status': status,
            'reason': reason,
            'color': color,
            'is_yellow': is_yellow
        }

    def get_stats(self):
        return {
            'total_switches': self.total_switches,
            'total_emergencies': self.total_emergencies,
            'current_phase': self.current_phase,
        }
