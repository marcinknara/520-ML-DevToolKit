import numpy as np
from time import sleep
import neptune.new as neptune

run = neptune.init(project='bfink99/example-project-pytorch',
                   api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4MDQ4NzE2OS0zMTRmLTQ1NmEtOWVmMy04MzQ5OGE3MzQ2ODMifQ==')

# log score
run['single_metric'] = 0.62

for i in range(100):
    sleep(0.2) # to see logging live
    run['random_training_metric'].log(i * np.random.random())
    run['other_random_training_metric'].log(0.5 * i * np.random.random())
