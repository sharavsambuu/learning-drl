import gymnasium as gym
import gymnasium_robotics

env = gym.make("FetchPush-v4", render_mode="human")

def run_rollout():
    observation, info = env.reset()
    
    total_reward = 0
    steps        = 0
    terminated   = False
    truncated    = False

    while not terminated and not truncated:
        action = env.action_space.sample()
        
        observation, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        steps += 1

    print(f"\n=== Rollout Finished ===")
    print(f"Total Steps: {steps}")
    print(f"Total Reward: {total_reward}")
    
    is_success = info.get("is_success", 0.0)
    print(f"Did agent succeed? : {'YES' if is_success == 1.0 else 'NO'}")
    
    if "distance_threshold" in info:
        print(f"Goal Threshold: {info['distance_threshold']}")

if __name__ == "__main__":
    for i in range(20):
        run_rollout()
    
    env.close()