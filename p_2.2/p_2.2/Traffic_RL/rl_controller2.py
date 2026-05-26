import os
import sys
import traci
from sumolib import checkBinary

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "sim.sumocfg")

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Error: Please declare environment variable 'SUMO_HOME'")

sumoCmd = [checkBinary('sumo-gui'), "-c", config_path, "--start", "--delay", "100"]

print("Connecting to SUMO...")
traci.start(sumoCmd)

tls_id = traci.trafficlight.getIDList()[0]
num_signals = len(traci.trafficlight.getRedYellowGreenState(tls_id))
signals_per_dir = num_signals // 4 

middle_east_phases = []
for i in range(4):
    state = ['r'] * num_signals 
    start_idx = i * signals_per_dir
    for j in range(start_idx, start_idx + signals_per_dir):
        state[j] = 'G'
    middle_east_phases.append("".join(state))

controlled_lanes = list(set(traci.trafficlight.getControlledLanes(tls_id)))
print(f"Controlled lanes connected to junction: {controlled_lanes}")

step = 0
current_phase_index = 0
frames_in_current_phase = 0
PHASE_DURATION = 100 

while step < 1000:
    traci.simulationStep() 
    
    total_waiting_cars = 0
    lane_queues = {} 
    
    for lane in controlled_lanes:
        halting_cars = traci.lane.getLastStepHaltingNumber(lane)
        lane_queues[lane] = halting_cars
        total_waiting_cars += halting_cars

    if frames_in_current_phase >= PHASE_DURATION:
        current_phase_index = (current_phase_index + 1) % 4 
        frames_in_current_phase = 0
        
        print("-" * 40)
        print(f"Switched to Phase: {current_phase_index}")
        print(f"Total waiting cars at intersection: {total_waiting_cars}")
        print("-" * 40)

    traci.trafficlight.setRedYellowGreenState(tls_id, middle_east_phases[current_phase_index])
    
    frames_in_current_phase += 1
    step += 1

traci.close()
print("Simulation ended successfully.")