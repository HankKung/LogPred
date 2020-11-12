import pandas as pd


df_temp = pd.read_csv('BGL.log_templates.csv')
df_stru = pd.read_csv('BGL.log_structured.csv')

df_temp = pd.DataFrame(df_temp)
df_stru = pd.DataFrame(df_stru)

temp_len = df_temp.shape[0]

temp2key = dict()
n = 0
for i, row in df_temp.iterrows():
	if row['Occurrences'] > 8:
		temp2key[row['EventId']] = n
		n += 1
	else:
		temp2key[row['EventId']] = -1

# for i in range(temp_len):
# 	temp2key[df_temp['EventId'][i]] = i

print('template stage done')
print(n)

log_len = df_stru.shape[0]

keys=[]
n = 0
for i, row in df_stru.iterrows():

	if i % 200000 == 0 :
		print('reading log '+str(100*i/log_len) + '%')
	if i>0 and temp2key[row['EventId']] != -1 and keys[n-1][1] == row['Label'] and keys[n-1][2] == row['EventId'] and row['Timestamp'] - key[n-1][3]<2:
		continue
	if temp2key[row['EventId']] != -1:
		keys.append([])
		keys[n].append(str(row['LineId']))
		keys[n].append(row['Label'])
		keys[n].append(str(temp2key[row['EventId']])) 
		keys[n].append(str(row['Timestamp']))
		n += 1

with open('keys_with_time_8.txt', 'w') as f:
	for i, item in enumerate(keys):
		if i % 200000 == 0 :
			print('writing log '+str(i/log_len) + '%')
		key = item[0]+' '+item[1]+' '+item[2] + ' '+ item[3] + '\n'
		f.write(key)
