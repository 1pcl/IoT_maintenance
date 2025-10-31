# Maintenance Strategy Recommendation for Wireless Sensor Networks


## Scenarios

### Scenario: Smart Factory Monitoring

**Scenario Description:**
In the smart meter reading scenario, the dataset specifies $208$ sensor nodes deployed in a multi-floor residential building. 
This scenario focuses on automatic meter reading in a modern multi-story residential building. As illustrated in Fig.~\ref{fig:meter_topology}, a total of $208$ smart meter nodes are deployed across $27$ floors. Each node periodically collects and reports electricity consumption data. The scenario is characterized by sparse node distribution, low-frequency data collection, long system lifetime requirements, and relatively low-value individual data packets. 

This scenario addresses the complex maintenance requirements of modern industrial facilities with heterogeneous sensor deployments. As shown in Fig.~\ref{fig:factory_topology}, a total of $400$ sensor nodes are strategically distributed across four specialized monitoring zones within a $500 \times 500$ square meter factory area. The deployment encompasses distinct sensor types with varying characteristics: production equipment monitoring utilizes high-power sensors for real-time equipment status and operational parameters; environmental monitoring employs medium-power sensors for temperature and humidity tracking; security surveillance deploys low-power sensors with reliable long-distance communication; and infrastructure monitoring uses ultra-low-power sensors for periodic status checks of utility systems.

The scenario is characterized by diverse sampling frequencies (ranging from 60 to 1800 seconds), varied data packet sizes (400 to 2000 bytes), and zone-specific failure probabilities reflecting the distinct operational environments. This heterogeneous configuration presents complex maintenance optimization challenges, requiring tailored strategies for each sensor category while respecting the overall system budget of $1,500,000$.

**Topology Structure:**  
![Smart Meter Topology](./images/smart_meter_topology.png)


# Usage
If you need to run this project for recommending maintenance strategy of WSNs, then only need to configure [runtime environment](#runtime-environment)。


# Environment Configuration
## runtime environment
- Python 3.8+
- Required packages: numpy, matplotlib, scipy

# Directory Structure Description
```sh
|-- README.md            // Help
|-- images               // Topology
|-- display              // Visualization
|-- utilities.py         // Maintenance modules and the fault model 
|-- scene_generator.py   // Scene Generator (Smart Meter Network and Animal House Monitoring)
|-- calculate_cost.py    // Calculate the cost and expenses
|-- module_cost.py       // Calculate the cost and expenses of maintenance module
|-- simulator.py         // WSN simulation
|-- enhanced_genetic.py  // Enhanced Genetic Algorithm
|-- genetic.py           // Genetic Algorithm
|-- particle_swarm.py    // Particle Swarm Optimization Algorithm
|-- simulated_annealing.py     // Simulated Annealing Algorithm
|-- evaluation.py            // Evaluation of maintenance strategies
```




