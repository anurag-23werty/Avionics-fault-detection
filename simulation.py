import pandas as pd
import numpy as np
import time
import datetime
import os
from scipy.signal import savgol_filter

class FlightSimulator:
    """
    Real-time avionics sensor simulator with occasional sensor errors
    for training anomaly detection models
    """
    
    def __init__(self, sampling_rate=1.0, error_probability=0.02):
        """
        Initialize flight simulator
        
        Parameters:
        -----------
        sampling_rate : float
            Samples per second (Hz)
        error_probability : float
            Probability of sensor error per reading (0.02 = 2% error rate)
        """
        self.sampling_rate = sampling_rate
        self.error_probability = error_probability
        self.sample_count = 0
        self.flight_phase = 0
        self.phase_progress = 0
        
        # Flight phases
        self.phases = [
            {'name': 'CRUISE_LOW', 'duration': 300, 'target_alt': 10000, 'target_speed': 252},
            {'name': 'CLIMB', 'duration': 240, 'target_alt': 15000, 'target_speed': 248},
            {'name': 'CRUISE_HIGH', 'duration': 300, 'target_alt': 15000, 'target_speed': 248},
            {'name': 'DESCENT', 'duration': 240, 'target_alt': 10000, 'target_speed': 255},
            {'name': 'CRUISE_LOW', 'duration': 300, 'target_alt': 10000, 'target_speed': 252},
        ]
        
        # Current state
        self.altitude = 10000.0
        self.airspeed = 252.0
        self.temperature = 15.0
        
        # Previous values for smoothing
        self.prev_altitude = self.altitude
        self.prev_airspeed = self.airspeed
        self.prev_temperature = self.temperature
        
        # Error tracking
        self.errors_generated = []
        
    def get_current_phase(self):
        """Get current flight phase"""
        return self.phases[self.flight_phase]
    
    def advance_phase(self):
        """Move to next flight phase"""
        self.phase_progress = 0
        self.flight_phase = (self.flight_phase + 1) % len(self.phases)
        
    def calculate_base_values(self):
        """Calculate base sensor values based on flight phase"""
        phase = self.get_current_phase()
        
        # Calculate progress through current phase (0 to 1)
        progress = self.phase_progress / phase['duration']
        
        # Altitude interpolation
        if phase['name'] == 'CLIMB':
            target_alt = 10000 + (5000 * progress)
        elif phase['name'] == 'DESCENT':
            target_alt = 15000 - (5000 * progress)
        else:
            target_alt = phase['target_alt']
        
        # Airspeed interpolation
        if phase['name'] == 'CLIMB':
            target_speed = 252 - (4 * progress)
        elif phase['name'] == 'DESCENT':
            target_speed = 248 + (7 * progress)
        else:
            target_speed = phase['target_speed']
        
        # Smooth transitions
        self.altitude = 0.95 * self.prev_altitude + 0.05 * target_alt
        self.airspeed = 0.95 * self.prev_airspeed + 0.05 * target_speed
        
        # Temperature based on altitude (lapse rate)
        base_temp_at_10k = 15.0
        lapse_rate = -0.002  # °C per foot
        self.temperature = base_temp_at_10k + (self.altitude - 10000) * lapse_rate
        
        # Store for next iteration
        self.prev_altitude = self.altitude
        self.prev_airspeed = self.airspeed
        self.prev_temperature = self.temperature
        
    def add_normal_noise(self, altitude, airspeed, temperature):
        """Add normal sensor noise"""
        altitude += np.random.normal(0, 5)
        airspeed += np.random.normal(0, 0.3)
        temperature += np.random.normal(0, 0.15)
        
        return altitude, airspeed, temperature
    
    def generate_error(self):
        """Generate different types of sensor errors"""
        error_types = [
            'spike', 'drift', 'stuck', 'dropout', 'noise_burst'
        ]
        
        error_type = np.random.choice(error_types)
        affected_sensor = np.random.choice(['airspeed', 'altitude', 'temperature'])
        
        error_info = {
            'sample': self.sample_count,
            'timestamp': datetime.datetime.now().isoformat(),
            'type': error_type,
            'sensor': affected_sensor,
            'phase': self.get_current_phase()['name']
        }
        
        return error_type, affected_sensor, error_info
    
    def apply_error(self, altitude, airspeed, temperature, error_type, affected_sensor):
        """Apply specific error to sensor reading"""
        
        if error_type == 'spike':
            # Sudden spike in reading (sensor glitch)
            magnitude = np.random.uniform(2, 5)
            if affected_sensor == 'altitude':
                altitude += magnitude * 100 * np.random.choice([-1, 1])
            elif affected_sensor == 'airspeed':
                airspeed += magnitude * 5 * np.random.choice([-1, 1])
            else:
                temperature += magnitude * 2 * np.random.choice([-1, 1])
                
        elif error_type == 'drift':
            # Gradual drift (calibration error)
            drift = np.random.uniform(0.5, 2.0)
            if affected_sensor == 'altitude':
                altitude += drift * 50
            elif affected_sensor == 'airspeed':
                airspeed += drift * 2
            else:
                temperature += drift * 0.5
                
        elif error_type == 'stuck':
            # Sensor stuck at previous value
            if affected_sensor == 'altitude':
                altitude = self.prev_altitude
            elif affected_sensor == 'airspeed':
                airspeed = self.prev_airspeed
            else:
                temperature = self.prev_temperature
                
        elif error_type == 'dropout':
            # Reading drops to unrealistic value
            if affected_sensor == 'altitude':
                altitude = altitude * 0.5  # Half the altitude
            elif affected_sensor == 'airspeed':
                airspeed = airspeed * 0.7  # 30% speed drop
            else:
                temperature = temperature - 10  # Sudden cold
                
        elif error_type == 'noise_burst':
            # Sudden increase in noise
            if affected_sensor == 'altitude':
                altitude += np.random.normal(0, 50)
            elif affected_sensor == 'airspeed':
                airspeed += np.random.normal(0, 5)
            else:
                temperature += np.random.normal(0, 2)
        
        return altitude, airspeed, temperature
    
    def get_sample(self):
        """Generate one sample of sensor data"""
        # Calculate base values
        self.calculate_base_values()
        
        # Get current values
        altitude = self.altitude
        airspeed = self.airspeed
        temperature = self.temperature
        
        # Add normal noise
        altitude, airspeed, temperature = self.add_normal_noise(altitude, airspeed, temperature)
        
        # Decide if error should occur
        has_error = np.random.random() < self.error_probability
        error_info = None
        
        if has_error:
            error_type, affected_sensor, error_info = self.generate_error()
            altitude, airspeed, temperature = self.apply_error(
                altitude, airspeed, temperature, error_type, affected_sensor
            )
            self.errors_generated.append(error_info)
        
        # Create sample dictionary
        sample = {
            'timestamp': datetime.datetime.now().isoformat(),
            'sample_number': self.sample_count,
            'airspeed_sensor': airspeed,
            'altitude_sensor': altitude,
            'temperature_sensor': temperature,
            'flight_phase': self.get_current_phase()['name'],
            'has_error': int(has_error),
            'error_type': error_info['type'] if error_info else 'none',
            'error_sensor': error_info['sensor'] if error_info else 'none'
        }
        
        # Increment counters
        self.sample_count += 1
        self.phase_progress += 1
        
        # Check if phase should advance
        if self.phase_progress >= self.get_current_phase()['duration']:
            self.advance_phase()
        
        return sample
    
    def generate_batch(self, num_samples):
        """Generate a batch of samples"""
        samples = []
        for _ in range(num_samples):
            samples.append(self.get_sample())
        return pd.DataFrame(samples)
    
    def run_continuous(self, output_file, duration_seconds=None, num_samples=None, 
                      real_time=False, verbose=True):
        """
        Run continuous simulation
        
        Parameters:
        -----------
        output_file : str
            Path to output CSV file
        duration_seconds : int
            Run for this many seconds (if real_time=True)
        num_samples : int
            Generate this many samples (alternative to duration)
        real_time : bool
            Wait between samples to simulate real-time
        verbose : bool
            Print progress information
        """
        
        if verbose:
            print("="*80)
            print("FLIGHT SIMULATOR - CONTINUOUS MODE")
            print("="*80)
            print(f"Sampling rate: {self.sampling_rate} Hz")
            print(f"Error probability: {self.error_probability*100:.1f}%")
            print(f"Output file: {output_file}")
            
            if real_time:
                print(f"Mode: Real-time (duration: {duration_seconds}s)")
            else:
                print(f"Mode: Fast generation (samples: {num_samples})")
            print("="*80)
        
        # Initialize output file
        first_sample = True
        
        # Determine how many samples to generate
        if num_samples is None and duration_seconds is not None:
            num_samples = int(duration_seconds * self.sampling_rate)
        elif num_samples is None:
            num_samples = 1000  # Default
        
        start_time = time.time()
        
        for i in range(num_samples):
            sample = self.get_sample()
            
            # Write to file
            df = pd.DataFrame([sample])
            df.to_csv(output_file, mode='a', header=first_sample, index=False)
            first_sample = False
            
            # Verbose output
            if verbose and (i % 100 == 0 or sample['has_error']):
                status = "⚠️ ERROR" if sample['has_error'] else "✓ OK"
                print(f"[{i:6d}] {status} | Phase: {sample['flight_phase']:12s} | "
                      f"Alt: {sample['altitude_sensor']:7.1f}ft | "
                      f"Speed: {sample['airspeed_sensor']:6.2f}kt | "
                      f"Temp: {sample['temperature_sensor']:5.2f}°C")
                
                if sample['has_error']:
                    print(f"         └─ {sample['error_type']} error in {sample['error_sensor']}")
            
            # Real-time delay
            if real_time:
                time.sleep(1.0 / self.sampling_rate)
        
        elapsed = time.time() - start_time
        
        if verbose:
            print("="*80)
            print("SIMULATION COMPLETE")
            print("="*80)
            print(f"Total samples: {num_samples}")
            print(f"Elapsed time: {elapsed:.2f}s")
            print(f"Samples/second: {num_samples/elapsed:.1f}")
            print(f"Errors generated: {len(self.errors_generated)}")
            print(f"Error rate: {len(self.errors_generated)/num_samples*100:.2f}%")
            print(f"Output saved to: {output_file}")
            print("="*80)
        
        return pd.read_csv(output_file)


def main():
    """Main function to run the simulator"""
    
    # Create simulator with 2% error rate
    simulator = FlightSimulator(
        sampling_rate=1.0,  # 1 Hz
        error_probability=0.02  # 2% error rate
    )
    
    # Generate 30,000 samples for training
    print("\n" + "="*80)
    print("GENERATING TRAINING DATASET WITH ERRORS")
    print("="*80)
    
    output_file = 'avionics_with_errors_30000.csv'
    
    df = simulator.run_continuous(
        output_file=output_file,
        num_samples=30000,
        real_time=False,
        verbose=True
    )
    
    # Generate error report
    print("\n" + "="*80)
    print("ERROR ANALYSIS")
    print("="*80)
    
    error_df = df[df['has_error'] == 1]
    
    print(f"\nTotal samples: {len(df)}")
    print(f"Normal samples: {len(df[df['has_error'] == 0])} ({len(df[df['has_error'] == 0])/len(df)*100:.2f}%)")
    print(f"Error samples: {len(error_df)} ({len(error_df)/len(df)*100:.2f}%)")
    
    print("\nError types distribution:")
    print(error_df['error_type'].value_counts())
    
    print("\nAffected sensors distribution:")
    print(error_df['error_sensor'].value_counts())
    
    print("\nErrors by flight phase:")
    print(error_df['flight_phase'].value_counts())
    
    # Save error summary
    error_summary = {
        'total_samples': len(df),
        'normal_samples': len(df[df['has_error'] == 0]),
        'error_samples': len(error_df),
        'error_rate': len(error_df)/len(df)*100,
        'error_types': error_df['error_type'].value_counts().to_dict(),
        'affected_sensors': error_df['error_sensor'].value_counts().to_dict(),
        'errors_by_phase': error_df['flight_phase'].value_counts().to_dict()
    }
    
    # Create detailed error log
    error_log_path = 'error_log.csv'
    error_df.to_csv(error_log_path, index=False)
    print(f"\nDetailed error log saved to: {error_log_path}")
    
    # Create summary report
    report_path = 'simulation_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FLIGHT SIMULATOR - ERROR INJECTION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("SIMULATION PARAMETERS\n")
        f.write("-"*80 + "\n")
        f.write(f"Sampling rate: 1.0 Hz\n")
        f.write(f"Error probability: 2.0%\n")
        f.write(f"Total samples: {len(df)}\n\n")
        
        f.write("DATASET COMPOSITION\n")
        f.write("-"*80 + "\n")
        f.write(f"Normal samples: {len(df[df['has_error'] == 0])} ({len(df[df['has_error'] == 0])/len(df)*100:.2f}%)\n")
        f.write(f"Error samples: {len(error_df)} ({len(error_df)/len(df)*100:.2f}%)\n\n")
        
        f.write("ERROR TYPES\n")
        f.write("-"*80 + "\n")
        f.write("1. SPIKE: Sudden spike in reading (sensor glitch)\n")
        f.write("2. DRIFT: Gradual drift (calibration error)\n")
        f.write("3. STUCK: Sensor stuck at previous value\n")
        f.write("4. DROPOUT: Reading drops to unrealistic value\n")
        f.write("5. NOISE_BURST: Sudden increase in noise\n\n")
        
        f.write("ERROR TYPE DISTRIBUTION\n")
        f.write("-"*80 + "\n")
        for error_type, count in error_df['error_type'].value_counts().items():
            f.write(f"{error_type}: {count} ({count/len(error_df)*100:.1f}%)\n")
        
        f.write("\nAFFECTED SENSORS\n")
        f.write("-"*80 + "\n")
        for sensor, count in error_df['error_sensor'].value_counts().items():
            f.write(f"{sensor}: {count} ({count/len(error_df)*100:.1f}%)\n")
        
        f.write("\nERRORS BY FLIGHT PHASE\n")
        f.write("-"*80 + "\n")
        for phase, count in error_df['flight_phase'].value_counts().items():
            f.write(f"{phase}: {count} ({count/len(error_df)*100:.1f}%)\n")
        
        f.write("\nUSAGE FOR MODEL TRAINING\n")
        f.write("-"*80 + "\n")
        f.write("This dataset is suitable for:\n")
        f.write("1. Anomaly detection model training\n")
        f.write("2. Sensor fault detection\n")
        f.write("3. Predictive maintenance algorithms\n")
        f.write("4. Flight data validation systems\n\n")
        
        f.write("The 'has_error' column can be used as the target label for supervised learning.\n")
        f.write("Features include: airspeed_sensor, altitude_sensor, temperature_sensor, flight_phase\n")
    
    print(f"\nSimulation report saved to: {report_path}")
    
    # Create clean dataset (without error columns) for comparison
    clean_df = df[['timestamp', 'sample_number', 'airspeed_sensor', 
                   'altitude_sensor', 'temperature_sensor', 'flight_phase']].copy()
    clean_path = 'avionics_sensor_only_30000.csv'
    clean_df.to_csv(clean_path, index=False)
    print(f"Clean sensor data (no labels) saved to: {clean_path}")
    
    print("\n" + "="*80)
    print("ALL FILES GENERATED SUCCESSFULLY!")
    print("="*80)


if __name__ == "__main__":
    main()
