
train_list_1 = []

with open('/Users/7ckung/Documents/GitHub/LogPred/data/hdfs_train', 'r') as f:
    for line in f.readlines():
        train_list_1.append(line)

print(len(train_list_1))



train_list_2 = []

with open('/Users/7ckung/Documents/GitHub/LogPred/data/hdfs_test_normal', 'r') as f:
    for line in f.readlines():
        train_list_2.append(line)

print(len(train_list_2))

total_line = len(train_list_1) + len(train_list_2)
part = int(total_line * 0.8)
part = part - len(train_list_1)

train_list = train_list_1 + train_list_2[:part]
val_list = train_list_2[part:]
print(len(train_list))
print(len(val_list))
with open('/Users/7ckung/Documents/GitHub/LogPred/data/hdfs_train_80.txt', 'w') as f:
    for line in train_list:
        f.write(line + '\n')
with open('/Users/7ckung/Documents/GitHub/LogPred/data/hdfs_test_normal_80.txt', 'w') as f:
    for line in val_list:
        f.write(line + '\n')