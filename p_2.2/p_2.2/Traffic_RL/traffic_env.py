import os
import sys
import traci
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sumolib import checkBinary

class TrafficEnv(gym.Env):
    """
    بيئة التحكم بإشارات المرور باستخدام التعلم المعزز (PPO).
    
    === نظام المكافآت المتقدم (Advanced Reward Shaping) ===
    1. عقوبة وقت الانتظار التراكمي   (Accumulated Delay Penalty)
    2. ضريبة التبديل                  (Switch Penalty)
    3. مكافأة العبور                   (Throughput Reward)
    4. معاقبة إهدار الوقت الأخضر       (Wasted Green Penalty)
    5. مكافأة كفاءة الوقت الأخضر       (Green Time Efficiency Reward)
    6. رؤية المستقبل                   (Approaching Vehicles in State)
    
    === فضاء المراقبة (Observation Space) - 14 بُعد ===
    [0-3]   عدد السيارات الواقفة في كل مسار (Queue)
    [4-7]   عدد السيارات المقتربة في كل مسار (Approaching)
    [8-11]  وقت الانتظار التراكمي لكل مسار (Accumulated Waiting Time)
    [12]    الإشارة الحالية (Current Phase: 0-3)
    [13]    الوقت على نفس الإشارة (Time on Same Phase)
    """
    
    def __init__(self):
        super(TrafficEnv, self).__init__()
        
        # 0 = إبقاء الإشارة، 1 = انتقال
        self.action_space = spaces.Discrete(2) 
        
        # === فضاء المراقبة الموسّع: 14 بُعد ===
        self.observation_space = spaces.Box(
            low=0, high=10000, shape=(14,), dtype=np.float32
        )
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = "sim.sumocfg"
        
        # تذكر: استخدم 'sumo' للتدريب، و 'sumo-gui' للاختبار
        self.sumoCmd = [checkBinary('sumo-gui'), "-c", self.config_path, 
                        "--start", "--no-warnings"]
        
        self.tls_id = None
        self.phases = []
        self.phase_lanes = {0: [], 1: [], 2: [], 3: []} 
        
        self.step_count = 0
        self.current_phase = 0
        self.time_on_same_phase = 0
        
        # === متغيرات التتبع الجديدة ===
        self.vehicles_before_step = set()  # لحساب مكافأة العبور (Throughput)
        self.cars_passed_on_green = 0      # لحساب كفاءة الوقت الأخضر
        
        # === معاملات المكافآت (Reward Hyperparameters) ===
        # مُصغّرة لتسهيل التعلم (النطاق المثالي: -100 إلى +100 لكل حلقة)
        self.REWARD_EMERGENCY_OVERRIDE = +5      # مكافأة الاستجابة للطوارئ
        self.PENALTY_SWITCH = -1                 # ضريبة التبديل (مخفضة لتشجيع التبديل عند الحاجة)
        self.REWARD_THROUGHPUT_PER_CAR = +0.5    # مكافأة لكل سيارة تعبر
        self.PENALTY_WASTED_GREEN = -3           # عقوبة الإشارة الخضراء لشارع فارغ
        self.PENALTY_DELAY_WEIGHT = -0.002       # وزن عقوبة الانتظار التراكمي
        self.REWARD_GREEN_EFFICIENCY = +3        # مكافأة كفاءة الوقت الأخضر العالية
        self.PENALTY_QUEUE = -0.1                # عقوبة الازدحام الأساسية (لكل سيارة)
        self.PENALTY_IMBALANCE = -0.3            # عقوبة عدم التوازن (مسار أخضر فارغ + أحمر مزدحم)
        self.MAX_PHASE_STEPS = 8                 # الحد الأقصى: 8 خطوات = 80 ثانية

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        try:
            traci.close()
        except:
            pass
            
        traci.start(self.sumoCmd)
        self.tls_id = traci.trafficlight.getIDList()[0]
        
        num_signals = len(traci.trafficlight.getRedYellowGreenState(self.tls_id))
        signals_per_dir = num_signals // 4
        self.phases = []
        
        links = traci.trafficlight.getControlledLinks(self.tls_id)
        
        for i in range(4):
            state = ['r'] * num_signals
            start_idx = i * signals_per_dir
            lanes_for_this_phase = set()
            
            for j in range(start_idx, start_idx + signals_per_dir):
                state[j] = 'G'
                if j < len(links) and len(links[j]) > 0:
                    incoming_lane = links[j][0][0]
                    lanes_for_this_phase.add(incoming_lane)
                    
            self.phases.append("".join(state))
            self.phase_lanes[i] = list(lanes_for_this_phase)
            
        self.step_count = 0
        self.current_phase = 0
        self.time_on_same_phase = 0
        self.cars_passed_on_green = 0
        
        # تسجيل جميع المركبات الحالية في المحاكاة
        self.vehicles_before_step = set(traci.vehicle.getIDList())
        
        traci.trafficlight.setRedYellowGreenState(
            self.tls_id, self.phases[self.current_phase]
        )
        
        return self._get_observation(), {}

    # ------------------------------------------------------------------
    # === فضاء المراقبة الموسّع (14 بُعد) ===
    # ------------------------------------------------------------------
    def _get_observation(self):
        obs = np.zeros(14, dtype=np.float32)
        
        for i in range(4):
            total_queue = 0        # السيارات الواقفة (سرعة < 0.1 م/ث)
            total_approaching = 0  # السيارات المقتربة (متحركة)
            total_wait = 0.0       # وقت الانتظار التراكمي
            
            for lane in self.phase_lanes[i]:
                # عدد السيارات الواقفة تماماً
                total_queue += traci.lane.getLastStepHaltingNumber(lane)
                
                # عدد جميع السيارات في المسار (المقتربة = الكل - الواقفة)
                all_vehicles = traci.lane.getLastStepVehicleNumber(lane)
                total_approaching += max(0, all_vehicles - traci.lane.getLastStepHaltingNumber(lane))
                
                # وقت الانتظار التراكمي لجميع السيارات في هذا المسار
                vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
                for veh_id in vehicle_ids:
                    total_wait += traci.vehicle.getAccumulatedWaitingTime(veh_id)
            
            obs[i] = total_queue            # [0-3] الطوابير
            obs[4 + i] = total_approaching  # [4-7] المقتربات
            obs[8 + i] = total_wait         # [8-11] وقت الانتظار التراكمي
            
        obs[12] = self.current_phase        # [12] الإشارة الحالية
        obs[13] = self.time_on_same_phase   # [13] مدة البقاء على نفس الإشارة
        return obs

    # ------------------------------------------------------------------
    # === خطوة المحاكاة مع نظام المكافآت المتقدم ===
    # ------------------------------------------------------------------
    def step(self, action):
        obs_before = self._get_observation()
        reward = 0.0
        did_switch = False
        
        # =========================================================
        # 1. نظام الطوارئ السيادي (Emergency Override)
        # =========================================================
        emergency_phase = -1
        
        for phase_idx in range(4):
            for lane in self.phase_lanes[phase_idx]:
                vehicles = traci.lane.getLastStepVehicleIDs(lane)
                for veh in vehicles:
                    if traci.vehicle.getVehicleClass(veh) == "emergency":
                        emergency_phase = phase_idx
                        break
            if emergency_phase != -1:
                break
                
        # =========================================================
        # 2. اتخاذ القرار
        # =========================================================
        if emergency_phase != -1:
            # === حالة الطوارئ: تجاهل كل شيء وافتح المسرب فوراً ===
            if self.current_phase != emergency_phase:
                did_switch = True
            self.current_phase = emergency_phase
            self.time_on_same_phase = 0 
            self.cars_passed_on_green = 0
            reward += self.REWARD_EMERGENCY_OVERRIDE
            
        else:
            # === الحالة العادية ===
            queue_on_current_green = obs_before[self.current_phase]
            total_other_cars = sum(obs_before[:4]) - queue_on_current_green
            
            # إيجاد المسار الأكثر ازدحاماً (للتبديل الذكي)
            max_red_queue = 0
            busiest_lane = -1
            for i in range(4):
                if i != self.current_phase and obs_before[i] > max_red_queue:
                    max_red_queue = obs_before[i]
                    busiest_lane = i
            
            if action == 1 or self.time_on_same_phase >= self.MAX_PHASE_STEPS:
                
                # ==== الاحتكار الذكي (Smart Hold) ====
                if total_other_cars == 0 and queue_on_current_green > 0:
                    self.time_on_same_phase = 0 
                # ====================================
                else:
                    # === مكافأة كفاءة الوقت الأخضر (Green Time Efficiency) ===
                    if self.time_on_same_phase > 0:
                        efficiency = self.cars_passed_on_green / self.time_on_same_phase
                        if efficiency >= 2.0:
                            reward += self.REWARD_GREEN_EFFICIENCY
                    
                    # === التبديل الذكي: اذهب للمسار الأكثر ازدحاماً ===
                    # بدلاً من التدوير التسلسلي الأعمى
                    if busiest_lane != -1:
                        next_phase = busiest_lane
                    else:
                        # إذا كل المسارات فارغة، تدوير عادي
                        next_phase = (self.current_phase + 1) % 4
                        
                    self.current_phase = next_phase
                    self.time_on_same_phase = 0
                    self.cars_passed_on_green = 0
                    did_switch = True
                    
            else:
                self.time_on_same_phase += 1
                
                # === عقوبة إهدار الوقت الأخضر (Wasted Green) ===
                if queue_on_current_green == 0 and obs_before[4 + self.current_phase] == 0:
                    reward += self.PENALTY_WASTED_GREEN

        # =========================================================
        # 3. تنفيذ القرار
        # =========================================================
        traci.trafficlight.setRedYellowGreenState(
            self.tls_id, self.phases[self.current_phase]
        )
        
        # === ضريبة التبديل (Switch Penalty) ===
        if did_switch:
            reward += self.PENALTY_SWITCH
        
        # =========================================================
        # 4. التقدم 10 ثوانٍ في المحاكاة
        # =========================================================
        vehicles_at_start = set(traci.vehicle.getIDList())
        
        for _ in range(10):
            traci.simulationStep()
            self.step_count += 1
            
        vehicles_at_end = set(traci.vehicle.getIDList())
        
        # =========================================================
        # 5. حساب المكافآت بعد تنفيذ الخطوة
        # =========================================================
        obs_after = self._get_observation()
        
        # === مكافأة العبور (Throughput Reward) ===
        # السيارات التي كانت موجودة في البداية واختفت = عبرت التقاطع بنجاح
        departed_vehicles = vehicles_at_start - vehicles_at_end
        throughput = len(departed_vehicles)
        reward += throughput * self.REWARD_THROUGHPUT_PER_CAR
        self.cars_passed_on_green += throughput
        
        # === عقوبة وقت الانتظار التراكمي (Accumulated Delay Penalty) ===
        # مجموع أوقات الانتظار لجميع المسارات
        total_accumulated_wait = sum(obs_after[8:12])
        reward += total_accumulated_wait * self.PENALTY_DELAY_WEIGHT
        
        # === عقوبة الازدحام الأساسية (Queue Penalty) ===
        total_queue = sum(obs_after[:4])
        reward += total_queue * self.PENALTY_QUEUE
        
        # === عقوبة عدم التوازن (Imbalance Penalty) ===
        # إذا المسار الأخضر فيه قليل سيارات بينما مسار أحمر مزدحم جداً
        green_queue = obs_after[self.current_phase]
        max_red_q = max(obs_after[i] for i in range(4) if i != self.current_phase)
        imbalance = max(0, max_red_q - green_queue)  # الفرق بين أكثر مسار أحمر مزدحم والمسار الأخضر
        if imbalance > 3:  # فقط إذا الفرق كبير (أكثر من 3 سيارات)
            reward += imbalance * self.PENALTY_IMBALANCE
        
        # =========================================================
        # 6. حالة الانتهاء
        # =========================================================
        terminated = False
        truncated = self.step_count >= 2000 
        
        if truncated:
            traci.close()
            
        return obs_after, reward, terminated, truncated, {}