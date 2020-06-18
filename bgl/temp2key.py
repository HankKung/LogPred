import pandas as pd


df_temp = pd.read_csv('BGL.log_templates.csv')
df_stru = pd.read_csv('BGL.log_structured.csv')

df_temp = pd.DataFrame(df_temp)
df_stru = pd.DataFrame(df_stru)

temp_len = df_temp.shape[0]

temp2key = dict()

for i in range(temp_len):
	temp2key[df_temp['EventId'][i]] = i

print('template stage done')

log_len = df_stru.shape[0]

keys=[]
for i, row in df_stru.iterrows():
	if i % 200000 == 0 :
		print('reading log '+str(100*i/log_len) + '%')
	keys.append([])
	keys[i].append(str(row['LineId']))
	keys[i].append(row['Label'])
	keys[i].append(str(temp2key[row['EventId']])) 
	keys[i].append(str(row['Timestamp']))

with open('keys.txt', 'w') as f:
	for i, item in enumerate(keys):
		if i % 200000 == 0 :
			print('writing log '+str(i/log_len) + '%')
		key = item[0]+' '+item[1]+' '+item[2] + item[3] + '\n'
		f.write(key)
