import sys
import json
import os

sys.path.insert(0, 'src')
from etl import *

def main(targets):

    data_config = json.load(open('config/data-params.json'))

    if 'data' in targets:
        latency_data_prep(**data_config)
        loss_data_prep(**data_config)

    if 'test' in targets:
        loss_data_prep(**data_config)
        latency_linear_reg(latency_data_prep(**data_config))
        decision_tree(latency_data_prep(**data_config))
        svm(latency_data_prep(**data_config))
        
    if 'all' in targets:
        latency_data_prep(**data_config)
        loss_data_prep(**data_config)
        latency_linear_reg(latency_data_prep(**data_config))
        decision_tree(latency_data_prep(**data_config))
        svm(latency_data_prep(**data_config))

if __name__ == '__main__':

    targets = sys.argv[1:]
    main(targets)
