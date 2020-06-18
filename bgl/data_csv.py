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
node = dict()
out_node = []
out_key = []
out_temp = []
out_label = []
out_time = []
for i, row in df_stru.iterrows():
	if i % 200000 == 0 :
		print('reading log '+str(100*i/log_len) + '%')
	if str(row['Node']) not in node:
		node[str(row['Node'])] = len(node)
	out_node.append(node[str(row['Node'])])
	out_temp.append(str(row['EventTemplate']))
	out_label.append(row['Label'])
	out_key.append(str(temp2key[row['EventId']])) 
	out_time.append(row['Timestamp'])


out_dict = {
			"label": out_label,
			"timestamp": out_time,
			"node": out_node,
			"template": out_temp
			}

out_df = pd.DataFrame(out_dict)
out_df.to_csv('data.csv')
