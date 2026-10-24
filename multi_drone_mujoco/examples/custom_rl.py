import os
import time
import gymnasium as gym
import numpy as np
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from multi_drone_mujoco.envs.base_aviary import _generate_aviary_xml
from multi_drone_mujoco.envs.custom_aviary import CustomAviary
from multi_drone_mujoco.utils.enums import DroneModel, Physics, ActionType, ObservationType

def add_obstacles(env):
    init_xyzs = np.array([[0.0, 0.0, 0.5]])
    init_rpys = np.array([[0.0, 0.0, 0.0]])
    xml = env.xml_str

    np.random.seed(42)

    obstacle_xml = ""
    actuator_xml = ""

    colors = [
        "0.9 0.2 0.2 0.9",
        "0.2 0.8 0.2 0.9",
        "0.2 0.3 0.9 0.9",
        "0.9 0.6 0.1 0.9",
        "0.7 0.2 0.9 0.9",
        "0.1 0.8 0.8 0.9"
    ]

    for i in range(10):
        x = np.random.uniform(-1.5, 1.5)
        y = np.random.uniform(-1.5, 1.5)

        if abs(x) < 0.3 and abs(y) < 0.3:
            x += 0.5 * np.sign(x + 0.01)

        h = np.random.uniform(0.4, 1.5)
        r = np.random.uniform(0.03, 0.07)
        color = colors[i % len(colors)]

        # Obstacle body
        obstacle_xml += f'''
        <body name="obs_{i}" pos="{x:.2f} {y:.2f} {h/2:.2f}">

            <joint
                name="obs_{i}_joint"
                type="slide"
                axis="1 0 0"
                range="-0.5 0.5"
                limited="true"/>

            <geom
                type="cylinder"
                size="{r:.3f} {h/2:.2f}"
                rgba="{color}"/>
            
            <inertial pos="0 0 0" mass="0.1" diaginertia="0.0001 0.0001 0.0001"/>

        </body>
        '''

        # Actuator
        actuator_xml += f'''
        <position
            name="obs_{i}_motor"
            joint="obs_{i}_joint"
            kp="100"/>
        '''

    for i in range(10, 25):
        x = np.random.uniform(-1.5, 1.5)
        y = np.random.uniform(-1.5, 1.5)

        if abs(x) < 0.3 and abs(y) < 0.3:
            x += 0.5 * np.sign(x + 0.01)

        h = np.random.uniform(0.4, 1.5)
        r = np.random.uniform(0.03, 0.07)
        color = colors[i % len(colors)]

        # Obstacle body
        obstacle_xml += f'''
        <body name="obs_{i}" pos="{x:.2f} {y:.2f} {h/2:.2f}">
            <geom
                type="cylinder"
                size="{r:.3f} {h/2:.2f}"
                rgba="{color}"/>

        </body>
        '''

    # Add obstacles to worldbody
    xml = xml.replace(
        "</worldbody>",
        obstacle_xml + "  </worldbody>"
    )

    # Add actuators to actuator section
    xml = xml.replace(
        "</actuator>",
        actuator_xml + "  </actuator>"
    )

    # Create new model
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    env.model = model
    env.data = data
    env._renderer = None

    mujoco.mj_forward(env.model, env.data)
    env._updateAndStoreKinematicInformation()

    # Store actuator IDs
    env.obstacle_actuator_ids = np.array([
        mujoco.mj_name2id(
            env.model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            f"obs_{i}_motor"
        )
        for i in range(10)
    ])

def main():
    print("="*60)
    print("Training PPO on CustomAviary")
    print("="*60)

    # Create the training environment
    env = CustomAviary(record=False, gui=False)
    add_obstacles(env)
    
    # Create the evaluation environment
    eval_env = CustomAviary(record=False, gui=False)
    add_obstacles(eval_env)
    
    # Setup evaluation callback to log reward progression
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./logs/',
        log_path='./logs/',
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    # Initialize the PPO model
    # We use a simple MLP policy. 
    model = PPO("MultiInputPolicy", env, verbose=1)
    
    # Train for 100,000 steps. This is relatively short but should show the reward trend.
    model.learn(total_timesteps=100000, callback=eval_callback)
    
    print("Training finished. Evaluating final policy...")
    
    # Evaluate a few episodes
    obs, info = eval_env.reset()
    total_reward = 0
    episodes = 0
    while episodes < 5:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"Episode {episodes+1} finished with reward: {total_reward:.2f}")
            obs, info = eval_env.reset()
            total_reward = 0
            episodes += 1

    env.close()
    eval_env.close()

def view_env():
    print("="*60)
    print("Viewing CustomAviary Environment")
    print("="*60)

    env = CustomAviary(record=False, gui=False)
    add_obstacles(env)

    env.reset()

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():

            t = env.data.time

            # Move the first 10 obstacles sinusoidally
            for i, actuator_id in enumerate(env.obstacle_actuator_ids):
                amplitude = 0.5
                frequency = 0.2

                env.data.ctrl[actuator_id] = (
                    amplitude * np.sin(
                        2 * np.pi * frequency * t
                    )
                )

            mujoco.mj_step(env.model, env.data)
            viewer.sync()

            time.sleep(0.01)

    env.close()

if __name__ == "__main__":
    view_env()  # View the environment with obstacles
