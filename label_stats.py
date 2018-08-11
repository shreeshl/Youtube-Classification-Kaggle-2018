
my_data = open('/Users/shreesh/Downloads/vocabulary.csv', 'r').read().split('\n')
dict_map = {}
for data in my_data:
    if len(data)==0:continue
    data = data.split(',')
    dict_map[data[0]] = data[3]

f = open('/Users/shreesh/yt8m/att/stats','r').read().split('\n')
ans = {}
for item in f:
    if len(item)==0:continue
    item = item.split(' ')
    ans[int(item[1])] = dict_map[item[0]]

ans = sorted(ans.items())[::-1]
for item in ans[:20]:
    print item

