import os

window_size = 10

with open('keys_with_time_8.txt', 'r') as f:
	all_lines = f.readlines()

Slide = [5]
Step = [5]

for slide in Slide:
	for step in Step:
		dataset = []
		key_seq = []

		for i in range(0, len(all_lines)-(window_size+step+slide)):

			if all_lines[i].strip().split()[1] == '-':
				key_seq.append(all_lines[i].strip().split()[2])

			elif all_lines[i].strip().split()[1] != '-':
				if len(key_seq) < window_size + slide -1:
					key_seq = []
				else:
					dataset.append(key_seq)
					key_seq = []


		data_dir = 'loss_window_' + str(window_size) + 'future_' + str(step) + 'slide_' +str(slide)+ 'remove_8' + '/' 
		if not os.path.isdir(data_dir):
			os.makedirs(data_dir)

		with open(data_dir+'dataset.txt', 'w') as f:
			for i, item in enumerate(dataset):
				if i % 1000 == 0:
					print('writing prediction dataset '+str(100*i/len(dataset)) + '%')
				out = ''
				for j in item:
					out = out + j + ' ' 
				f.write(out+'\n')



