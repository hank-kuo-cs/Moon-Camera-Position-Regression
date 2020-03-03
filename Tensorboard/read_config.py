import sys
import json


file_path = sys.argv[1]
with open(file_path, 'r') as f:
	data = json.loads(f.read())
	for k1, v1 in data.items():
		print(k1)
		for k2, v2 in v1.items():
			print('\t{0}: {1}'.format(k2, v2))
		print()

