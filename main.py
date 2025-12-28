import numpy as np
import random
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 🟨 Laser Module
class LaserModule:
    def __init__(self, intensity=1.0):
        self.intensity = intensity

    def generate_interference(self, freq1, freq2):
        return abs(freq1 - freq2) * self.intensity

    def generate_resonant_interference(self, freq1, freq2, resonance_factor=0.1):
        """Interference with resonance: higher intensity at close frequencies"""
        diff = abs(freq1 - freq2)
        # Resonance equation: higher intensity at small differences
        if diff < 0.5:
            resonance = self.intensity * (1.0 / (diff + 0.001))
            return resonance
        return diff * self.intensity


# 🟦 Logic Gates (modified for wave detection)
class LaserLogicGate:
    def __init__(self, laser_module, threshold=1.0):
        self.laser = laser_module
        self.threshold = threshold

    def detect_resonance(self, f1, f2):
        """Detect resonance between two frequencies"""
        interference = self.laser.generate_resonant_interference(f1, f2)
        return interference > self.threshold * 0.5

    def detect_wave_coupling(self, frequencies):
        """Detect wave coupling in a set of frequencies"""
        coupled = []
        for i in range(len(frequencies)):
            for j in range(i+1, len(frequencies)):
                if self.detect_resonance(frequencies[i], frequencies[j]):
                    coupled.append((i, j))
        return coupled


# 🌊 Plasma Wave Simulator
class PlasmaWaveSimulator:
    def __init__(self, laser_module):
        self.laser = laser_module
        self.resonance_history = []
    
    def simulate_plasma_wave(self, plasma_params, steps=200):
        """
        Simulate wave propagation in cosmic plasma
        
        plasma_params: {
            'density': plasma density,
            'temperature': temperature,
            'magnetic_field': magnetic field strength,
            'base_frequency': base frequency
        }
        """
        # Basic plasma equations
        plasma_freq = np.sqrt(plasma_params['density']) * 5.0  # ω_p
        cyclotron_freq = plasma_params['magnetic_field'] * 3.0  # ω_c
        
        time = np.linspace(0, 20, steps)
        wave_amplitudes = []
        wave_interferences = []
        
        print(f"🌀 Simulating plasma: ω_p={plasma_freq:.2f}, ω_c={cyclotron_freq:.2f}")
        
        for t in time:
            # Basic plasma wave
            plasma_wave = np.sin(plasma_freq * t)
            
            # Magnetic field effect
            magnetic_effect = 0.3 * np.sin(cyclotron_freq * t)
            
            # Interference between frequencies (laser simulation)
            interference = self.laser.generate_resonant_interference(
                plasma_freq, 
                cyclotron_freq
            )
            
            # Detect resonance conditions
            if abs(plasma_freq - cyclotron_freq) < 1.0:
                self.resonance_history.append({
                    'time': t,
                    'freq_diff': abs(plasma_freq - cyclotron_freq),
                    'interference': interference
                })
            
            # Total amplitude with interference effect
            total_wave = plasma_wave + magnetic_effect + 0.1 * interference * np.random.randn()
            wave_amplitudes.append(total_wave)
            wave_interferences.append(interference)
        
        return {
            'time': time,
            'amplitudes': wave_amplitudes,
            'interferences': wave_interferences,
            'plasma_freq': plasma_freq,
            'cyclotron_freq': cyclotron_freq
        }
    
    def analyze_resonance_modes(self, frequencies):
        """Analyze resonance modes in plasma"""
        resonance_modes = []
        
        for i, f1 in enumerate(frequencies):
            resonance_peaks = []
            for j, f2 in enumerate(frequencies):
                if i != j:
                    diff = abs(f1 - f2)
                    # Resonance occurs at relatively small differences
                    if diff < 1.0:
                        strength = 1.0 / (diff + 0.001)
                        resonance_peaks.append({
                            'pair': (i, j),
                            'strength': strength,
                            'diff': diff
                        })
            
            if resonance_peaks:
                resonance_modes.append({
                    'frequency': f1,
                    'peaks': resonance_peaks,
                    'total_resonance': sum(p['strength'] for p in resonance_peaks)
                })
        
        return sorted(resonance_modes, key=lambda x: x['total_resonance'], reverse=True)


# 🧬 Physics Simulator (modified for hot plasma)
class PlasmaPhysicsSimulator:
    def __init__(self, laser_module):
        self.laser = laser_module
    
    def simulate_hot_plasma(self, particles, magnetic_field=1.0, steps=100, dt=0.01):
        """Simulate hot plasma in magnetic field"""
        trajectories = []
        wave_energies = []
        
        for step in range(steps):
            new_particles = []
            total_wave_energy = 0
            
            for i, (mass, charge, pos, vel) in enumerate(particles):
                # Lorentz force: F = q(E + v × B)
                lorentz_force = np.array([0.0, 0.0])
                
                # Magnetic field effect (v × B)
                if magnetic_field > 0:
                    b_field = np.array([0.0, magnetic_field])  # field in y direction
                    v_cross_b = np.array([
                        vel[1] * b_field[1],
                        -vel[0] * b_field[1]
                    ])
                    lorentz_force += charge * v_cross_b
                
                # Particle interactions (laser simulation)
                for j, (mass_j, charge_j, pos_j, _) in enumerate(particles):
                    if i != j:
                        r = pos_j - pos
                        dist = np.linalg.norm(r)
                        if dist > 0:
                            # Coulomb interaction with interference effect
                            coulomb = (charge * charge_j * r) / (dist ** 3)
                            
                            # Simulate laser interference effect
                            freq_i = np.linalg.norm(vel) * 10
                            freq_j = np.linalg.norm(particles[j][3]) * 10
                            interference = self.laser.generate_interference(freq_i, freq_j)
                            
                            lorentz_force += coulomb * (1 + 0.1 * interference)
                
                # Calculate acceleration and new position
                acc = lorentz_force / mass
                vel += acc * dt
                pos += vel * dt
                
                # Store wave energy
                wave_energy = 0.5 * mass * np.linalg.norm(vel)**2
                total_wave_energy += wave_energy
                
                new_particles.append((mass, charge, pos, vel))
            
            particles = new_particles
            trajectories.append([pos.copy() for _, _, pos, _ in particles])
            wave_energies.append(total_wave_energy)
        
        return np.array(trajectories), wave_energies


# 🖥️ Laser Computer with plasma capabilities
class LaserPlasmaComputer:
    def __init__(self, intensity=1.0):
        self.laser = LaserModule(intensity=intensity)
        self.logic = LaserLogicGate(self.laser)
        self.memory = {}
        self.wave_simulator = PlasmaWaveSimulator(self.laser)
        self.plasma_simulator = PlasmaPhysicsSimulator(self.laser)
    
    def simulate_cosmic_plasma_regions(self):
        """Simulate different cosmic plasma regions"""
        print("🌌 Simulating Wave Interactions in Cosmic Plasma")
        print("=" * 60)
        
        # Define typical cosmic plasma regions
        cosmic_regions = [
            {
                'name': 'Intergalactic Medium (IGM)',
                'density': 0.001,
                'temperature': 1e6,
                'magnetic_field': 0.001,
                'description': 'Very thin plasma, 90% of universe volume'
            },
            {
                'name': 'Galactic Halo',
                'density': 0.01,
                'temperature': 1e7,
                'magnetic_field': 0.01,
                'description': 'Hot gas around galaxies'
            },
            {
                'name': 'Galactic Disk',
                'density': 1.0,
                'temperature': 1e4,
                'magnetic_field': 0.1,
                'description': 'Star-forming region'
            },
            {
                'name': 'Quasar Jet',
                'density': 0.1,
                'temperature': 1e8,
                'magnetic_field': 1.0,
                'description': 'Relativistic jets from black hole'
            }
        ]
        
        results = {}
        
        for region in cosmic_regions:
            print(f"\n📍 Region: {region['name']}")
            print(f"   {region['description']}")
            print(f"   Density: {region['density']:.3f}, B: {region['magnetic_field']:.3f}")
            
            # Simulate waves
            wave_result = self.wave_simulator.simulate_plasma_wave({
                'density': region['density'],
                'temperature': region['temperature'],
                'magnetic_field': region['magnetic_field'],
                'base_frequency': 1.0
            }, steps=150)
            
            # Analyze resonance modes
            test_freqs = [
                wave_result['plasma_freq'],
                wave_result['cyclotron_freq'],
                wave_result['plasma_freq'] * 0.5,
                wave_result['cyclotron_freq'] * 2.0
            ]
            resonance_modes = self.wave_simulator.analyze_resonance_modes(test_freqs)
            
            results[region['name']] = {
                'wave': wave_result,
                'resonance': resonance_modes[:3] if resonance_modes else [],
                'params': region
            }
            
            # Display results
            if resonance_modes:
                print(f"   🔥 Resonance modes detected: {len(resonance_modes)}")
                for mode in resonance_modes[:2]:
                    print(f"     Frequency {mode['frequency']:.2f}: Resonance strength {mode['total_resonance']:.2f}")
        
        return results
    
    def visualize_results(self, results):
        """Visualize plasma simulation results"""
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Plasma waves in different regions
        ax1 = plt.subplot(2, 3, 1)
        colors = ['blue', 'green', 'red', 'purple']
        for idx, (region_name, data) in enumerate(results.items()):
            wave = data['wave']
            ax1.plot(wave['time'][:100], wave['amplitudes'][:100], 
                    color=colors[idx], alpha=0.7, linewidth=1.5,
                    label=f"{region_name[:15]}...")
        ax1.set_title("Plasma Waves in Different Cosmic Regions")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Amplitude")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. Frequency interference
        ax2 = plt.subplot(2, 3, 2)
        for idx, (region_name, data) in enumerate(results.items()):
            wave = data['wave']
            ax2.plot(wave['time'][:100], wave['interferences'][:100], 
                    color=colors[idx], alpha=0.7, linewidth=1.5)
        ax2.set_title("Laser Interference Intensity")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Intensity")
        ax2.grid(True, alpha=0.3)
        
        # 3. Resonance chart
        ax3 = plt.subplot(2, 3, 3)
        resonance_data = []
        labels = []
        for region_name, data in results.items():
            if data['resonance']:
                total_resonance = sum(mode['total_resonance'] for mode in data['resonance'])
                resonance_data.append(total_resonance)
                labels.append(region_name[:12])
        
        if resonance_data:
            bars = ax3.bar(range(len(resonance_data)), resonance_data, color='orange', alpha=0.7)
            ax3.set_title("Resonance Strength in Different Regions")
            ax3.set_ylabel("Total Resonance Strength")
            ax3.set_xticks(range(len(labels)))
            ax3.set_xticklabels(labels, rotation=45, ha='right')
            
            # Add values on bars
            for bar, val in zip(bars, resonance_data):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f"{val:.1f}", ha='center', va='bottom', fontsize=9)
        
        # 4. Frequency distribution
        ax4 = plt.subplot(2, 3, 4)
        all_frequencies = []
        for region_name, data in results.items():
            wave = data['wave']
            all_frequencies.extend([wave['plasma_freq'], wave['cyclotron_freq']])
        
        ax4.hist(all_frequencies, bins=10, alpha=0.7, color='teal', edgecolor='black')
        ax4.set_title("Plasma Frequency Distribution")
        ax4.set_xlabel("Frequency")
        ax4.set_ylabel("Count")
        ax4.grid(True, alpha=0.3)
        
        # 5. Hot plasma particle simulation
        ax5 = plt.subplot(2, 3, 5)
        
        # Create plasma particles (protons and electrons)
        particles = []
        for i in range(10):
            # Protons
            particles.append((
                1836.0,  # Proton mass (electron units)
                1.0,     # Positive charge
                np.array([random.uniform(-1, 1), random.uniform(-1, 1)]),  # Position
                np.array([random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)])  # Velocity
            ))
            # Electrons
            particles.append((
                1.0,     # Electron mass
                -1.0,    # Negative charge
                np.array([random.uniform(-1, 1), random.uniform(-1, 1)]),
                np.array([random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2)])
            ))
        
        # Simulate hot plasma
        trajectories, wave_energies = self.plasma_simulator.simulate_hot_plasma(
            particles, magnetic_field=0.5, steps=50
        )
        
        # Plot particle trajectories
        for i in range(min(5, len(trajectories[0]))):
            traj = trajectories[:, i]
            x_coords = [pos[0] for pos in traj]
            y_coords = [pos[1] for pos in traj]
            ax5.plot(x_coords, y_coords, marker='.', markersize=2, linewidth=0.8, alpha=0.6)
        
        ax5.set_title("Particle Trajectories in Hot Plasma")
        ax5.set_xlabel("X")
        ax5.set_ylabel("Y")
        ax5.grid(True, alpha=0.3)
        
        # 6. Wave energy
        ax6 = plt.subplot(2, 3, 6)
        ax6.plot(wave_energies, color='darkred', linewidth=2)
        ax6.set_title("Wave Energy Evolution in Plasma")
        ax6.set_xlabel("Time Step")
        ax6.set_ylabel("Energy")
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig


# 🚀 Main execution
if __name__ == "__main__":
    print("🚀 Starting Wave Interactions Simulation in Cosmic Plasma")
    print("=" * 60)
    
    start_time = time.time()
    
    # Create laser plasma computer
    plasma_computer = LaserPlasmaComputer(intensity=2.5)
    
    # Run simulation
    results = plasma_computer.simulate_cosmic_plasma_regions()
    
    # Display results
    print("\n" + "=" * 60)
    print("📊 Main Results Analysis:")
    print("=" * 60)
    
    # Statistical analysis
    total_resonance = 0
    resonance_regions = []
    
    for region_name, data in results.items():
        if data['resonance']:
            region_resonance = sum(mode['total_resonance'] for mode in data['resonance'])
            total_resonance += region_resonance
            resonance_regions.append((region_name, region_resonance))
            
            print(f"\n📍 {region_name}")
            print(f"   Plasma frequency: {data['wave']['plasma_freq']:.2f}")
            print(f"   Cyclotron frequency: {data['wave']['cyclotron_freq']:.2f}")
            
            if data['resonance']:
                best_resonance = max(data['resonance'], key=lambda x: x['total_resonance'])
                print(f"   Strongest resonance: {best_resonance['total_resonance']:.2f}")
    
    print(f"\n📈 Total resonance strength in all regions: {total_resonance:.2f}")
    
    if resonance_regions:
        strongest_region = max(resonance_regions, key=lambda x: x[1])
        print(f"🔥 Strongest resonance region: {strongest_region[0]} ({strongest_region[1]:.2f})")
    
    # Visualization
    print("\n🎨 Creating visualizations...")
    plasma_computer.visualize_results(results)
    
    # Execution time
    end_time = time.time()
    print(f"\n⏱️ Total execution time: {end_time - start_time:.2f} seconds")
    
    print("\n✅ Wave interactions and resonance simulation in cosmic plasma completed!")
