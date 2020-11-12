import os
import random
window_size = 10
future_step = 0
ratio = 0.8
with open('keys_with_time_8.txt', 'r') as f:
	all_lines = f.readlines()

time_start = int(all_lines[0].strip().split()[-1])
time_end = int(all_lines[-1].strip().split()[-1])
time_interval = float(time_end - time_start)
time_split = int(time_interval * ratio + float(time_start))
normal_train_split = []
normal_test_split = []
abnormal_test_split = []
for i in range(0, len(all_lines)-(window_size+future_step)):
	key_seq = ''
	normal = True
	for j in range(window_size+future_step):
		key_seq = key_seq + all_lines[i+j].strip().split()[2] + ' '
		if j > window_size-1 and all_lines[i+j].strip().split()[1] != '-' and future_step != 0:
			normal = False
		elif future_step == 0 and all_lines[i+j].strip().split()[1] != '-':
			normal = False
	if normal and int(all_lines[i].strip().split()[-1]) <= time_split:
		normal_train_split.append(key_seq)
	elif normal and int(all_lines[i].strip().split()[-1]) > time_split:
		normal_test_split.append(key_seq)
	else:
		abnormal_test_split.append(key_seq)


n_train = len(normal_train_split)
n_normal_test = len(normal_test_split)
n_abnormal_test = len(abnormal_test_split)

random_normal = normal_train_split + normal_test_split
random.shuffle(random_normal)
normal_train_split = random_normal[:n_train]
normal_test_split = random_normal[n_train:]

data_dir = 'window_' + str(window_size) + 'future_' + str(future_step) + 'remove_8_random' + '/' 
if not os.path.isdir(data_dir):
	os.makedirs(data_dir)

normal_train_key = set()
with open(data_dir+'normal_train.txt', 'w') as f:
	for i, item in enumerate(normal_train_split):
		if i % 1000 == 0:
			print('writing normal train '+str(100*i/len(normal_train_split)) + '%')
		f.write(item+'\n')
		for key in item.strip().split():
			normal_train_key.add(key)

normal_test_key = set()
with open(data_dir+'normal_test.txt', 'w') as f:
	for i, item in enumerate(normal_test_split):
		if i % 1000 == 0:
			print('writing normal test '+str(100*i/len(normal_test_split)) + '%')
		f.write(item+'\n')
		for key in item.strip().split():
			normal_test_key.add(key)

abnormal_test_key = set()

with open(data_dir+'abnormal_test.txt', 'w') as f:
	for i, item in enumerate(abnormal_test_split):
		if i % 1000 == 0:
			print('writing abnormal '+str(100*i/len(abnormal_test_split)) + '%')
		f.write(item+'\n')
		for key in item.strip().split():
			abnormal_test_key.add(key)

