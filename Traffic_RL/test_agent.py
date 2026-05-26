import os
from stable_baselines3 import PPO
from traffic_env import TrafficEnv

# تحديد المسار الحالي للبحث عن المودل
script_dir = os.path.dirname(os.path.abspath(__file__))

# محاولة تحميل النموذج الجديد (v2)، وإذا لم يوجد يحمّل القديم
model_path_v2 = os.path.join(script_dir, "ppo_traffic_brain_v2")
model_path_v1 = os.path.join(script_dir, "ppo_traffic_brain")

if os.path.exists(model_path_v2 + ".zip"):
    model_path = model_path_v2
    print("Using v2 model (Advanced Reward Shaping)")
else:
    model_path = model_path_v1
    print("Using v1 model (Original)")

print("Loading the Environment...")
env = TrafficEnv()

print("Loading the Trained Brain...")
model = PPO.load(model_path)

obs, info = env.reset()

print("AI is now controlling the traffic!")
step = 0
while step < 2000:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()
        
    step += 1

print("Testing Finished.")
env.close()