import neptune
from neptune.sessions import Session
from config import config

session = Session.with_default_backend(api_token=config.API_TOKEN)

project = session.get_project(project_qualified_name='davianyang/TinyImageNetChallenge')

def create_neptune_experiment(exp_no, model):
    PARAMS = {'batch_size': config.BATCH_SIZE,
          'shuffle': True,
          'activation': 'relu',
          'learning_rate': 1e-2,
          'optimizer': 'SGD'}
    
    experiment = project.create_experiment(name=f'Experiment-{str(exp_no)}-{model}',
                          description=f'{model} trained on Tiny Image Net',
                          tags=['classification', 'TinyImageNet', f'{model}'],
                          params=PARAMS)

    return experiment