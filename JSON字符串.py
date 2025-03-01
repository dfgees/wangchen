"""
             json就是一种在各个编程语言中流通的数据格式，负责不同编程语言中的数据传递和交互。类似于：
             国际通用语言：英语
             中国56个民族不同地区的通用语言：普通话
"""
"""
演示json数据和python字典的相互转换
"""
import json
                                                             #准备列表，列表内每一个元素都是字典，将其转化为json
date=[{"name":'王晨','age':18},{'name':'王一航','age':18}]
json_str=json.dumps(date,ensure_ascii=False)
print(type(json_str))
print(date[1])
print(json_str)
                                                               #准备字典，将字典转换为JSON
d={"name":"王一航","add":14105}
json_strs=json.dumps(d,ensure_ascii=False)
print(type(json_str))
print(json_str)
                                                             #将JSON字符串转换为python数据类型
s='[{"name":"王一航","add":14105}]'
l=json.loads(s)
print(type(l))
print(l)
                                                             #将JSON字符串转换为python数据类型
s='{"name":"王一航","add":14105}'
d=json.loads(s)
print(type(d))
print(d)
"""
JSON无非就是一个单独的字典或一个内部元素都是字典的列表
"""










