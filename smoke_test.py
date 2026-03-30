from src.utils.logger import get_logger
from src.utils.common import read_yaml, ensure_dir

logger = get_logger('phase1_test', log_file='training.log')
logger.info('Phase 1 smoke test starting...')

config = read_yaml('configs/config.yaml')
logger.info(f'Config loaded: project name = {config["project"]["name"]}')

params = read_yaml('configs/params.yaml')
logger.info(f'Params loaded: test_size = {params["data_processing"]["test_size"]}')

ensure_dir('logs')
logger.info('All utilities working correctly')
print('SUCCESS — check logs/training.log for the output')