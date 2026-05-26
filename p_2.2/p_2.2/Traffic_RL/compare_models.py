import os
import sys
import xml.etree.ElementTree as ET
import traci
from sumolib import checkBinary
import numpy as np
from stable_baselines3 import PPO
from traffic_env import TrafficEnv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = "sim.sumocfg"

# Auto-detect v2 model (advanced reward), fallback to v1
_v2_path = os.path.join(SCRIPT_DIR, "ppo_traffic_brain_v2")
_v1_path = os.path.join(SCRIPT_DIR, "ppo_traffic_brain")
MODEL_PATH = _v2_path if os.path.exists(_v2_path + ".zip") else _v1_path

def parse_tripinfo(xml_file):
    if not os.path.exists(xml_file):
        return None
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    metrics = {
        'total_vehicles': 0,
        'total_wait_time': 0,
        'total_time_loss': 0,
        'total_stops': 0,
        'total_fuel': 0.0,
        'total_co2': 0.0,
        'emg_wait_time': 0,
        'emg_time_loss': 0,
    }
    
    for trip in root.findall('tripinfo'):
        metrics['total_vehicles'] += 1
        metrics['total_wait_time'] += float(trip.get('waitingTime', 0))
        metrics['total_time_loss'] += float(trip.get('timeLoss', 0))
        metrics['total_stops'] += int(trip.get('waitingCount', 0))
        
        # Check if it's the emergency vehicle
        if trip.get('id') == 'emg_hero':
            metrics['emg_wait_time'] = float(trip.get('waitingTime', 0))
            metrics['emg_time_loss'] = float(trip.get('timeLoss', 0))
            
        # Emissions
        emissions = trip.find('emissions')
        if emissions is not None:
            # SUMO outputs in mg, convert to grams
            metrics['total_fuel'] += float(emissions.get('fuel_abs', 0)) / 1000.0
            metrics['total_co2'] += float(emissions.get('CO2_abs', 0)) / 1000.0
            
    if metrics['total_vehicles'] > 0:
        metrics['avg_wait_time'] = metrics['total_wait_time'] / metrics['total_vehicles']
        metrics['avg_time_loss'] = metrics['total_time_loss'] / metrics['total_vehicles']
        metrics['avg_stops'] = metrics['total_stops'] / metrics['total_vehicles']
    else:
        metrics['avg_wait_time'] = 0
        metrics['avg_time_loss'] = 0
        metrics['avg_stops'] = 0
        
    return metrics

def run_simulation(mode="static"):
    output_file = f"tripinfo_{mode}.xml"
    # Added --device.emissions.probability 1.0 to get emissions for all vehicles
    sumo_cmd = [
        checkBinary('sumo'), 
        "-c", CONFIG_PATH, 
        "--no-warnings", 
        "--tripinfo-output", output_file,
        "--device.emissions.probability", "1.0"
    ]
    
    if mode == "ai":
        env = TrafficEnv()
        # Override the GUI command with the headless one
        env.sumoCmd = sumo_cmd
        try:
            model = PPO.load(MODEL_PATH)
        except Exception as e:
            print(f"Error loading AI model: {e}")
            return None
        
        obs, _ = env.reset()
        tls_id = env.tls_id
    else:
        traci.start(sumo_cmd)
        tls_id = traci.trafficlight.getIDList()[0]
        
    controlled_lanes = set(traci.trafficlight.getControlledLanes(tls_id))
    
    max_queue = 0
    total_queue = 0
    step_count = 0
    phase_switches = 0
    
    last_phase = traci.trafficlight.getRedYellowGreenState(tls_id)
    
    timeout = 20000 # simulation seconds
    
    print(f"--- Running {mode.upper()} Simulation ---")
    
    current_time = 0
    while traci.simulation.getMinExpectedNumber() > 0 and current_time < timeout:
        if mode == "ai":
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            current_time += 10 # TrafficEnv steps 10 seconds per action
        else:
            traci.simulationStep()
            current_time += 1
            
        # Track Queue Length
        current_queue = 0
        for lane in controlled_lanes:
            current_queue += traci.lane.getLastStepHaltingNumber(lane)
            
        max_queue = max(max_queue, current_queue)
        total_queue += current_queue
        step_count += 1
        
        # Track Phase Switches
        current_phase = traci.trafficlight.getRedYellowGreenState(tls_id)
        if current_phase != last_phase:
            phase_switches += 1
            last_phase = current_phase
            
        if mode == "ai" and (terminated or truncated):
            break
            
    try:
        traci.close()
    except Exception:
        pass
        
    metrics = parse_tripinfo(output_file)
    if metrics:
        metrics['max_queue'] = max_queue
        metrics['avg_queue'] = total_queue / step_count if step_count > 0 else 0
        metrics['phase_switches'] = phase_switches
        
    return metrics

def print_comparison(static_m, ai_m):
    print("\n" + "="*90)
    print(f"{'Metric':<35} | {'Static (No AI)':<15} | {'AI (PPO)':<15} | {'Improvement':<15}")
    print("-" * 90)
    
    def print_row(name, m_static, m_ai, unit="", lower_is_better=True):
        if m_static == 0:
            imp = "N/A"
        else:
            diff = m_static - m_ai if lower_is_better else m_ai - m_static
            pct = (diff / m_static) * 100
            imp = f"{pct:+.1f}%"
            
        val_static = f"{m_static:.2f}{unit}"
        val_ai = f"{m_ai:.2f}{unit}"
        print(f"{name:<35} | {val_static:<15} | {val_ai:<15} | {imp:<15}")

    print_row("Avg Waiting Time", static_m['avg_wait_time'], ai_m['avg_wait_time'], " s")
    print_row("Max Queue Length", static_m['max_queue'], ai_m['max_queue'], " cars")
    print_row("Avg Queue Length", static_m['avg_queue'], ai_m['avg_queue'], " cars")
    print_row("Avg Stops per Vehicle", static_m['avg_stops'], ai_m['avg_stops'], "")
    print_row("Total Time Loss", static_m['total_time_loss'], ai_m['total_time_loss'], " s")
    print_row("Fuel Consumption", static_m['total_fuel'], ai_m['total_fuel'], " g")
    print_row("CO2 Emissions", static_m['total_co2'], ai_m['total_co2'], " g")
    print_row("Phase Switches", static_m['phase_switches'], ai_m['phase_switches'], "", lower_is_better=False) 
    # Phase switches: lower isn't strictly better or worse without context, but usually fewer means less flickering, though AI might switch optimally.
    print("-" * 90)
    print_row("[EMERGENCY] Ambulance Wait Time", static_m['emg_wait_time'], ai_m['emg_wait_time'], " s")
    print("=" * 90 + "\n")

if __name__ == "__main__":
    print("Starting Benchmarking Process...")
    static_metrics = run_simulation("static")
    ai_metrics = run_simulation("ai")
    
    if static_metrics and ai_metrics:
        print_comparison(static_metrics, ai_metrics)
    else:
        print("Failed to gather metrics.")
