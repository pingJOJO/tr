import os
from stable_baselines3 import PPO
from traffic_env import TrafficEnv

# تحديد المسار الحالي لضمان حفظ المودل هنا دائماً
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "ppo_traffic_brain_v2")

if __name__ == '__main__':
    print("=" * 56)
    print("  PPO Traffic Brain - Training v2")
    print("  Advanced Reward Shaping + 14-dim Observation")
    print("=" * 56)

    print("\nInitializing training environment...")
    env = TrafficEnv()

    print("Building the AI Brain (PPO Model)...")
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,       # أصغر = تحديث أسرع وتدريب أسرع
        batch_size=64,
        n_epochs=5,        # أقل = أسرع لكل تحديث
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        device="cpu",      # PPO مع MlpPolicy أسرع على CPU (الشبكة صغيرة جداً)
    )

    print("Starting training phase (200,000 steps)...")
    model.learn(total_timesteps=200000)

    print("\nTraining finished! Saving the brain...")
    model.save(model_path)
    print(f"Model saved successfully at: {model_path}.zip")
    print("=" * 56)