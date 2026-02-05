import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import datetime

class PersistentErrorSimulator:
    """
    Realistic avionics simulator where sensor errors PERSIST for multiple samples
    Once a sensor becomes faulty, it stays faulty until it's "fixed"
    """
    
    def __init__(self, 
                 error_start_probability=0.001,  # Probability a NEW fault starts
                 mean_fault_duration=50,          # Average samples a fault lasts
                 fault_duration_std=20):          # Variation in fault duration
        """
        Initialize simulator with persistent faults
        
        Parameters:
        -----------
        error_start_probability : float
            Probability that a NEW fault begins at any sample (0.001 = 0.1%)
        mean_fault_duration : int
            Average number of samples a fault persists
        fault_duration_std : int
            Standard deviation of fault duration
        """
        self.error_start_probability = error_start_probability
        self.mean_fault_duration = mean_fault_duration
        self.fault_duration_std = fault_duration_std
        self.sample_count = 0
        
        # Flight phases
        self.phases = [
            {'name': 'CRUISE_LOW', 'duration': 300, 'target_alt': 10000, 'target_speed': 252},
            {'name': 'CLIMB', 'duration': 240, 'target_alt': 15000, 'target_speed': 248},
            {'name': 'CRUISE_HIGH', 'duration': 300, 'target_alt': 15000, 'target_speed': 248},
            {'name': 'DESCENT', 'duration': 240, 'target_alt': 10000, 'target_speed': 255},
            {'name': 'CRUISE_LOW', 'duration': 300, 'target_alt': 10000, 'target_speed': 252},
        ]
        
        self.flight_phase = 0
        self.phase_progress = 0
        
        # Current state
        self.altitude = 10000.0
        self.airspeed = 252.0
        self.temperature = 15.0
        
        self.prev_altitude = self.altitude
        self.prev_airspeed = self.airspeed
        self.prev_temperature = self.temperature
        
        # ACTIVE FAULTS - tracks ongoing sensor failures
        self.active_faults = {
            'altitude': {'active': False, 'remaining_samples': 0, 'error_type': None, 'start_sample': None},
            'airspeed': {'active': False, 'remaining_samples': 0, 'error_type': None, 'start_sample': None},
            'temperature': {'active': False, 'remaining_samples': 0, 'error_type': None, 'start_sample': None}
        }
        
        # Statistics
        self.fault_events = []
        self.stats = {
            'total_samples': 0,
            'samples_with_any_error': 0,
            'fault_events_started': 0
        }
    
    def get_current_phase(self):
        return self.phases[self.flight_phase]
    
    def advance_phase(self):
        self.phase_progress = 0
        self.flight_phase = (self.flight_phase + 1) % len(self.phases)
    
    def calculate_base_values(self):
        """Calculate correct sensor values"""
        phase = self.get_current_phase()
        progress = self.phase_progress / phase['duration']
        
        # Altitude
        if phase['name'] == 'CLIMB':
            target_alt = 10000 + (5000 * progress)
        elif phase['name'] == 'DESCENT':
            target_alt = 15000 - (5000 * progress)
        else:
            target_alt = phase['target_alt']
        
        # Airspeed
        if phase['name'] == 'CLIMB':
            target_speed = 252 - (4 * progress)
        elif phase['name'] == 'DESCENT':
            target_speed = 248 + (7 * progress)
        else:
            target_speed = phase['target_speed']
        
        # Smooth transitions
        self.altitude = 0.95 * self.prev_altitude + 0.05 * target_alt
        self.airspeed = 0.95 * self.prev_airspeed + 0.05 * target_speed
        
        # Temperature
        base_temp_at_10k = 15.0
        lapse_rate = -0.002
        self.temperature = base_temp_at_10k + (self.altitude - 10000) * lapse_rate
        
        self.prev_altitude = self.altitude
        self.prev_airspeed = self.airspeed
        self.prev_temperature = self.temperature
    
    def add_normal_noise(self, altitude, airspeed, temperature):
        """Add small random noise"""
        altitude += np.random.normal(0, 5)
        airspeed += np.random.normal(0, 0.3)
        temperature += np.random.normal(0, 0.15)
        return altitude, airspeed, temperature
    
    def start_new_fault(self, sensor):
        """Start a new persistent fault on a sensor"""
        # Choose error type
        error_type = np.random.choice(['spike', 'drift', 'stuck', 'dropout', 'noise_burst'])
        
        # Determine how long this fault will last
        duration = int(max(10, np.random.normal(self.mean_fault_duration, self.fault_duration_std)))
        
        # Activate the fault
        self.active_faults[sensor] = {
            'active': True,
            'remaining_samples': duration,
            'error_type': error_type,
            'start_sample': self.sample_count
        }
        
        # Track fault event
        self.fault_events.append({
            'sensor': sensor,
            'error_type': error_type,
            'start_sample': self.sample_count,
            'duration': duration
        })
        
        self.stats['fault_events_started'] += 1
    
    def update_faults(self):
        """Update all active faults - decrement timers"""
        for sensor in ['altitude', 'airspeed', 'temperature']:
            if self.active_faults[sensor]['active']:
                self.active_faults[sensor]['remaining_samples'] -= 1
                
                # Check if fault is over
                if self.active_faults[sensor]['remaining_samples'] <= 0:
                    self.active_faults[sensor]['active'] = False
    
    def check_for_new_faults(self):
        """Check if any new faults should start"""
        for sensor in ['altitude', 'airspeed', 'temperature']:
            # Only start new fault if sensor is currently healthy
            if not self.active_faults[sensor]['active']:
                if np.random.random() < self.error_start_probability:
                    self.start_new_fault(sensor)
    
    def apply_error_to_sensor(self, value, sensor_type, error_type):
        """Apply specific error type to a sensor"""
        
        if error_type == 'spike':
            # Continuous spikes
            magnitude = np.random.uniform(2, 5)
            if sensor_type == 'altitude':
                value += magnitude * 100 * np.random.choice([-1, 1])
            elif sensor_type == 'airspeed':
                value += magnitude * 5 * np.random.choice([-1, 1])
            else:
                value += magnitude * 2 * np.random.choice([-1, 1])
        
        elif error_type == 'drift':
            # Persistent drift - gets worse over time
            drift = np.random.uniform(0.5, 2.0)
            if sensor_type == 'altitude':
                value += drift * 50
            elif sensor_type == 'airspeed':
                value += drift * 2
            else:
                value += drift * 0.5
        
        elif error_type == 'stuck':
            # Stuck at a value
            if sensor_type == 'altitude':
                value = self.prev_altitude
            elif sensor_type == 'airspeed':
                value = self.prev_airspeed
            else:
                value = self.prev_temperature
        
        elif error_type == 'dropout':
            # Persistent low reading
            if sensor_type == 'altitude':
                value = value * 0.5
            elif sensor_type == 'airspeed':
                value = value * 0.7
            else:
                value = value - 10
        
        elif error_type == 'noise_burst':
            # High noise
            if sensor_type == 'altitude':
                value += np.random.normal(0, 50)
            elif sensor_type == 'airspeed':
                value += np.random.normal(0, 5)
            else:
                value += np.random.normal(0, 2)
        
        return value
    
    def get_sample(self):
        """Generate one sample with persistent faults"""
        
        # Calculate base values
        self.calculate_base_values()
        
        # Get sensor readings
        altitude = self.altitude
        airspeed = self.airspeed
        temperature = self.temperature
        
        # Add normal noise
        altitude, airspeed, temperature = self.add_normal_noise(altitude, airspeed, temperature)
        
        # Check for new faults starting
        self.check_for_new_faults()
        
        # Apply active faults
        altitude_error = self.active_faults['altitude']['active']
        airspeed_error = self.active_faults['airspeed']['active']
        temperature_error = self.active_faults['temperature']['active']
        
        error_details = []
        
        if altitude_error:
            error_type = self.active_faults['altitude']['error_type']
            altitude = self.apply_error_to_sensor(altitude, 'altitude', error_type)
            error_details.append(f"altitude:{error_type}")
        
        if airspeed_error:
            error_type = self.active_faults['airspeed']['error_type']
            airspeed = self.apply_error_to_sensor(airspeed, 'airspeed', error_type)
            error_details.append(f"airspeed:{error_type}")
        
        if temperature_error:
            error_type = self.active_faults['temperature']['error_type']
            temperature = self.apply_error_to_sensor(temperature, 'temperature', error_type)
            error_details.append(f"temperature:{error_type}")
        
        # Update fault timers
        self.update_faults()
        
        # Create sample
        has_any_error = altitude_error or airspeed_error or temperature_error
        
        sample = {
            'timestamp': datetime.datetime.now().isoformat(),
            'sample_number': self.sample_count,
            'airspeed_sensor': airspeed,
            'altitude_sensor': altitude,
            'temperature_sensor': temperature,
            'flight_phase': self.get_current_phase()['name'],
            'has_any_error': int(has_any_error),
            'altitude_has_error': int(altitude_error),
            'airspeed_has_error': int(airspeed_error),
            'temperature_has_error': int(temperature_error),
            'num_sensors_with_error': int(altitude_error) + int(airspeed_error) + int(temperature_error),
            'error_details': ';'.join(error_details) if error_details else 'none',
            # NEW: Track if this is part of an ongoing fault
            'altitude_fault_age': self.active_faults['altitude']['remaining_samples'] if altitude_error else 0,
            'airspeed_fault_age': self.active_faults['airspeed']['remaining_samples'] if airspeed_error else 0,
            'temperature_fault_age': self.active_faults['temperature']['remaining_samples'] if temperature_error else 0,
        }
        
        # Update statistics
        self.sample_count += 1
        self.phase_progress += 1
        self.stats['total_samples'] += 1
        if has_any_error:
            self.stats['samples_with_any_error'] += 1
        
        # Check phase advance
        if self.phase_progress >= self.get_current_phase()['duration']:
            self.advance_phase()
        
        return sample
    
    def generate_dataset(self, num_samples, output_file, verbose=True):
        """Generate complete dataset"""
        
        if verbose:
            print("="*80)
            print("PERSISTENT ERROR SIMULATOR - Realistic Sensor Faults")
            print("="*80)
            print(f"Generating {num_samples:,} samples...")
            print(f"Fault start probability: {self.error_start_probability*100:.2f}%")
            print(f"Average fault duration: {self.mean_fault_duration} samples")
            print("="*80)
        
        samples = []
        for i in range(num_samples):
            sample = self.get_sample()
            samples.append(sample)
            
            if verbose and i % 5000 == 0:
                active_faults = [s for s in ['altitude', 'airspeed', 'temperature'] 
                               if self.active_faults[s]['active']]
                status = f"Active faults: {active_faults}" if active_faults else "All sensors OK"
                print(f"[{i:6d}] {status}")
        
        df = pd.DataFrame(samples)
        df.to_csv(output_file, index=False)
        
        if verbose:
            print("\n" + "="*80)
            print("GENERATION COMPLETE")
            print("="*80)
            print(f"\nTotal samples: {len(df):,}")
            print(f"Normal samples: {(df['has_any_error']==0).sum():,} ({(df['has_any_error']==0).sum()/len(df)*100:.2f}%)")
            print(f"Samples with errors: {(df['has_any_error']==1).sum():,} ({(df['has_any_error']==1).sum()/len(df)*100:.2f}%)")
            
            print(f"\n📊 FAULT EVENTS:")
            print(f"  Total fault events started: {len(self.fault_events)}")
            print(f"  Average fault duration: {np.mean([f['duration'] for f in self.fault_events]):.1f} samples")
            
            print(f"\n📊 ERROR BREAKDOWN BY NUMBER OF SENSORS:")
            for num_errors in range(4):
                count = (df['num_sensors_with_error'] == num_errors).sum()
                print(f"  {num_errors} sensor(s) with error: {count:,} samples ({count/len(df)*100:.2f}%)")
            
            print(f"\n📊 INDIVIDUAL SENSOR ERROR COUNTS:")
            print(f"  Altitude sensor errors:     {df['altitude_has_error'].sum():,}")
            print(f"  Airspeed sensor errors:     {df['airspeed_has_error'].sum():,}")
            print(f"  Temperature sensor errors:  {df['temperature_has_error'].sum():,}")
            
            print(f"\n💾 File saved: {output_file}")
            print("="*80)
        
        return df


def main():
    """Generate dataset with persistent faults"""
    
    # Create simulator
    # Lower start probability but longer duration = realistic faults
    simulator = PersistentErrorSimulator(
        error_start_probability=0.001,  # 0.1% chance of NEW fault per sample
        mean_fault_duration=50,         # Faults last ~50 samples on average
        fault_duration_std=20           # Some variation
    )
    
    output_file = '/Users/anurag_77y/Desktop/Avionics fault recognization/persistent_errors_30000.csv'
    
    df = simulator.generate_dataset(
        num_samples=30000,
        output_file=output_file,
        verbose=True
    )
    
    # Analyze persistence
    print("\n" + "="*80)
    print("PERSISTENCE ANALYSIS")
    print("="*80)
    
    # Find error runs for altitude
    altitude_errors = df['altitude_has_error'].values
    runs = []
    current_run = 0
    
    for i in range(len(altitude_errors)):
        if altitude_errors[i] == 1:
            current_run += 1
        else:
            if current_run > 0:
                runs.append(current_run)
            current_run = 0
    
    if current_run > 0:
        runs.append(current_run)
    
    if runs:
        print(f"\n✅ ALTITUDE SENSOR ERROR PERSISTENCE:")
        print(f"  Number of fault events: {len(runs)}")
        print(f"  Average fault duration: {np.mean(runs):.1f} samples")
        print(f"  Longest fault: {max(runs)} samples")
        print(f"  Shortest fault: {min(runs)} samples")
        print(f"\n  First 10 fault durations: {runs[:10]}")
    
    # Show examples
    print("\n" + "="*80)
    print("EXAMPLE: Persistent Fault Pattern")
    print("="*80)
    
    # Find a fault sequence
    first_fault_idx = df[df['altitude_has_error'] == 1].index[0] if len(df[df['altitude_has_error'] == 1]) > 0 else 0
    example_range = df.iloc[first_fault_idx:first_fault_idx+10]
    
    print("\nShowing 10 consecutive samples during altitude sensor fault:")
    print("-" * 80)
    for _, row in example_range.iterrows():
        alt_status = "❌ ERROR" if row['altitude_has_error'] else "✓ OK"
        air_status = "❌ ERROR" if row['airspeed_has_error'] else "✓ OK"
        temp_status = "❌ ERROR" if row['temperature_has_error'] else "✓ OK"
        
        print(f"Sample {row['sample_number']:5d}: Alt: {alt_status:9s} | Air: {air_status:9s} | Temp: {temp_status:9s}")
    
    print("\n👀 Notice: Altitude error PERSISTS across multiple samples!")
    print("   This is REALISTIC - a faulty sensor stays faulty!")
    
    # Save additional files
    print("\n" + "="*80)
    print("SAVING ADDITIONAL FILES")
    print("="*80)
    
    # Fault events log
    fault_log = pd.DataFrame(simulator.fault_events)
    fault_log.to_csv('/Users/anurag_77y/Desktop/Avionics fault recognization/fault_events_log.csv', index=False)
    print(f"✓ Fault events log: fault_events_log.csv ({len(fault_log)} events)")
    
    # Normal only
    normal_df = df[df['has_any_error'] == 0]
    normal_df.to_csv('/Users/anurag_77y/Desktop/Avionics fault recognization/persistent_normal_only.csv', index=False)
    print(f"✓ Normal samples: persistent_normal_only.csv ({len(normal_df):,} rows)")
    
    # Errors only
    error_df = df[df['has_any_error'] == 1]
    error_df.to_csv('/Users/anurag_77y/Desktop/Avionics fault recognization/persistent_errors_only.csv', index=False)
    print(f"✓ Error samples: persistent_errors_only.csv ({len(error_df):,} rows)")
    
    print("\n" + "="*80)
    print("✅ DONE! Dataset with PERSISTENT faults created!")
    print("="*80)
    print("""
KEY FEATURES:
✓ Sensor faults PERSIST for realistic durations (20-70 samples)
✓ Once a sensor fails, it stays failed until "repaired"
✓ Multiple sensors can fail simultaneously
✓ Faults have patterns, not random single-sample errors
✓ Much more realistic for training anomaly detection!
""")


if __name__ == "__main__":
    main()
