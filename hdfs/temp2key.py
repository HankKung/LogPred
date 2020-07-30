import pandas as pd


df_temp = pd.read_csv('HDFS.log_templates.csv')
df_stru = pd.read_csv('HDFS.log_structured.csv')

df_temp = pd.DataFrame(df_temp)
df_stru = pd.DataFrame(df_stru)

temp_len = df_temp.shape[0]

temp2key = dict()

for i in range(temp_len):
	temp2key[df_temp['EventId'][i]] = i

print('template stage done')

log_len = df_stru.shape[0]

keys=[]
no_block = 0
for i, row in df_stru.iterrows():
	
	block = row['ParameterList']
	for e in block:
		if 'blk' not in e:
			no_block += 1
			
	block = str(block)
	block_idx = block.index('blk_')
	
	# space_index = block.index(' ', block_idx)
	block = block[block_idx:]
	if block.find('\'') != -1:
		index_end = block.index('\'')
		if block.find(' ') > -1:
			if block.find(' ') < index_end:
				block = block[0:block.find(' ')]
			else:
				block = block[0:index_end]
		else: 
			block = block[0:index_end]
	block = block.strip()
	keys.append([])
	keys[i].append(str(row['LineId']))
	keys[i].append(str(temp2key[row['EventId']])) 
	keys[i].append(str(row['Time']))
	keys[i].append(block)
	if i % 1000000 == 0 :
		print('reading log '+str(100*i/log_len) + '%')
		print(block)

print('no block: ', no_block)
with open('keys_with_time.txt', 'w') as f:
	for i, item in enumerate(keys):
		if i % 200000 == 0 :
			print('writing log '+str(i/log_len) + '%')
		key = item[0]+' '+item[1]+' '+item[2] + ' '+ item[3] + '\n'
		f.write(key)
