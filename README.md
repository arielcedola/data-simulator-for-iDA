# data-simulator-for-iDA

Simulation of sensor data in streaming:

1) data_simulator_iDA_v1: Simulates real-time sensor data events, bringing the possibility of sending them directly to an Azure Event Hub (must be configured previosuly).

2) data_simulator: Simulates sensor data containing failures, specially designed for predictive maintenance studies by means of classification algorithms. The script generates data from 7 sensors, with similar characteristics to the turbofan sensors provided by [NASA](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository) and usually utilized in predictive maintenance models. Data are saved in a file and then can be sent to an Azure Event Hub (must be configured previosuly).
