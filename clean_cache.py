import os
from glob import glob


def recursive_travel_all_directories(dir_path):
	subdir_paths = glob(dir_path + '*/')
	if subdir_paths:
		for subdir_path in subdir_paths:
			if subdir_path.find('__pycache__') > 0 or subdir_path.find('.idea') > 0:
				os.system('rm -rf %s' % subdir_path)
				print('delete %s' % subdir_path)
			else:
				recursive_travel_all_directories(subdir_path)
	else:
		return


recursive_travel_all_directories('./')
