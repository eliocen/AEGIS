# AEGIS v0.31 Step3C 鈥?Model/Runner Integration

Integrates the frozen Step2 calibrated-evidence and intervention-utility contract into the model and real training path for M4qcesi-c, M4qcesi-u, and M4qcesi. Calibration loss is 0.25 where enabled; utility-selector BCE is 0.50 where enabled; utility-target temperature is 0.05. The v0.29 harmful-transition objective remains disabled at effective weight 0.0.

Step3C is implementation verification only: no formal training, formal evaluation, checkpoint loading, hypothesis decision, or official-test access.
