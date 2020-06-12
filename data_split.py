import os

window_size = 20
future_step = 5

with open('keys.txt', 'r') as f:
	all_lines = f.readlines()

normal_split = []
abnormal_split = []
for i in range(0, len(all_lines)-(window_size+future_step)):
	key_seq = ''
	normal = True
	for j in range(window_size+future_step):
		key_seq = key_seq + all_lines[i+j].strip().split()[2] + ' '
		if j > 19 and  all_lines[i+j].strip().split()[1] != '-':
			normal = False 
	if normal:
		normal_split.append(key_seq)
	else:
		abnormal_split.append(key_seq)


data_dir = 'window_' + str(window_size) + 'future_' + str(future_step)+'/' 
if not os.path.isdir(data_dir):
	os.makedirs(data_dir)

with open(data_dir+'normal.txt', 'w') as f:
	print('normal_split_len:')
	print(len(normal_split))
	for i, item in enumerate(normal_split):
		if i % 1000 == 0:
			print('writing normal '+str(100*i/len(normal_split)) + '%')
		f.write(item+'\n')

with open(data_dir+'abnormal.txt', 'w') as f:
	print('abnormal_split_len:')
	print(len(abnormal_split))
	for i, item in enumerate(abnormal_split):
		if i % 1000 == 0:
			print('writing abnormal '+str(100*i/len(abnormal_split)) + '%')
		f.write(item+'\n')