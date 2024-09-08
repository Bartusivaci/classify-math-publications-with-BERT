
"""
To use this implementation, you simply have to implement `get_classifications` such that it returns classifications.
You can then let your agent compete on the server by calling

    python3 server_interaction.py path/to/your/config.json
"""
import json
import logging

import requests
import time


def get_classifications(titles):
    """ TODO: Implement this function.
        It gets a list of titles (strings).
        It should return a the classifications as a list.
        The first element of the returned list should be the classifications for the first title (as a list of 5 strings).
        
        Example:
            titles = ['Computation and application of robust data-driven bandwidth selection for gradient function estimation', 'Core, least core and nucleolus for multiple scenario cooperative games']
            return_value = [['Operations research, mathematical programming', 'Computer science', 'Numerical analysis', 'Statistics', 'Biology and other natural sciences'], ['Manifolds and cell complexes', 'Probability theory and stochastic processes', 'Computer science', 'Operations research, mathematical programming', 'Game theory, economics, social and behavioral sciences']] 
    """
    raise NotImplementedError()


def run(config_file, action_function, parallel_runs=True):
    logger = logging.getLogger(__name__)

    with open(config_file, 'r') as fp:
        config = json.load(fp)

    actions = []
    for request_number in range(51):    # 50 runs are enough for full evaluation. Running much more puts unnecessary strain on the server's database.
        logger.info(f'Iteration {request_number} (sending {len(actions)} actions)')
        # send request
        response = requests.put(f'{config["url"]}/act/{config["env"]}', json={
            'agent': config['agent'],
            'pwd': config['pwd'],
            'actions': actions,
            'single_request': not parallel_runs,
        })
        if response.status_code == 200:
            response_json = response.json()
            for error in response_json['errors']:
                logger.error(f'Error message from server: {error}')
            for message in response_json['messages']:
                logger.info(f'Message from server: {message}')

            action_requests = response_json['action-requests']
            if not action_requests:
                logger.info('The server has no new action requests - waiting for 1 second.')
                time.sleep(1)  # wait a moment to avoid overloading the server and then try again
            # get actions for next request
            actions = []
            for action_request in action_requests:
                actions.append({'run': action_request['run'], 'action': action_function(action_request['percept'])})
        elif response.status_code == 503:
            logger.warning('Server is busy - retrying in 3 seconds')
            time.sleep(3)  # server is busy - wait a moment and then try again
        else:
            # other errors (e.g. authentication problems) do not benefit from a retry
            logger.error(f'Status code {response.status_code}. Stopping.')
            break

    print('Done - 50 runs are enough for full evaluation')


if __name__ == '__main__':
    import sys
    run(sys.argv[1], get_classifications)
