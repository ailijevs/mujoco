#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import sys
import os
import numpy as np
import math

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 full_viewer.py <model.xml>")
        print("\nAvailable models to try:")
        print("  model/humanoid/humanoid.xml      - Humanoid character")
        print("  model/car/car.xml                - Car simulation")
        print("  model/flex/flag.xml              - Flexible flag")
        print("  model/hammock/hammock.xml        - Soft body hammock")
        print("  model/humanoid/100_humanoids.xml - Stress test!")
        print("  model/cube/cube_3x3x3.xml        - Rubik's cube")
        print("  model/balloons/balloons.xml      - Balloon physics")
        return
    
    model_path = sys.argv[1]
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found")
        return
    
    print(f"Loading {model_path}...")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    print(f"✅ Model loaded successfully!")
    print(f"   📦 Bodies: {model.nbody}")
    print(f"   🔗 Joints: {model.njnt}")
    print(f"   🎯 DOF: {model.nv}")
    print(f"   ⚡ Timestep: {model.opt.timestep:.4f}s")
    
    print("\n🎮 Controls:")
    print("   🖱️  Left mouse + drag  = Rotate camera")
    print("   🖱️  Right mouse + drag = Pan camera")  
    print("   🖱️  Scroll wheel       = Zoom")
    print("   ❌ Close window        = Exit")
    
    print("\n🚀 Starting interactive simulation...")
    
    # Set better starting position (standing upright)
    data.qpos[2] = 1.3  # Lift torso higher
    mujoco.mj_forward(model, data)

    def walking_controller(model, data, step):
        """Simple walking pattern"""
        time = step * 0.01
        
        # Only apply control if the model has actuators
        if model.nu > 0:
            # Simple rhythmic movement for legs
            walk_speed = 3.0
            for i in range(min(model.nu, 8)):
                if i == 0:  # Left hip
                    data.ctrl[i] = 0.4 * math.sin(walk_speed * time)
                elif i == 1:  # Right hip  
                    data.ctrl[i] = 0.4 * math.sin(walk_speed * time + math.pi)
                elif i == 2:  # Left knee
                    data.ctrl[i] = 0.3 * abs(math.sin(walk_speed * time))
                elif i == 3:  # Right knee
                    data.ctrl[i] = 0.3 * abs(math.sin(walk_speed * time + math.pi))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        while viewer.is_running():
            # Apply walking control
            walking_controller(model, data, step)
            
            # Step the physics
            mujoco.mj_step(model, data)
            viewer.sync()
            step += 1
            
            # Show progress every 2000 steps
            if step % 2000 == 0:
                height = data.qpos[2]
                print(f"Step {step:,}: Humanoid height = {height:.2f}m")

if __name__ == "__main__":
    main()
