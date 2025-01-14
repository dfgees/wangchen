"""
print(666)#写一个整数字面量
print(13.14)#写一个浮点型字面量
print("王晨")#写一个字符串字面量


     我叫王晨
     今年18
     来自咸阳

#王晨写的注释
#定义一个变量，用来记录钱包余额
money=50
print("钱包里还有:",money,"元",)
money=money-10
print("money-20=",money-10)
#变量是记录数据用的
#数据类型：字符串类型、整形、浮点型

#方法一
print(type("王晨"))
print (type(18))
print(type(13.14))
#方法二
name="王晨"
name_type=type(name)
print(name_type)
#方法三
string_type=type(name)
print(string_type)
#以上为查看变量的三种方式
#变量没有类型，但存储的数据有类型

#数据类型转换
num_str=str(11)
print(type(num_str),num_str)

float_str=str(1.1)
print(type(float_str),float_str)#将字符串转化为数字

num=int("11")
print(type(num),num)

num2=float("11.35")
print(type(num2),num2)#将字符串转化为数字

float_num=float(11)
print(type(float_num),float_num)#整数转化为浮点数

int_num=int(11.1)
print(type(int_num),int_num)#浮点数转化为整形
money=100
print(money)

print("1+1=",1+1)#加法
print("2-1=",2-1)#减法
print("2/2=",2/2)#除法
print("3*2=",3*2)#乘法
print("11//2=",11//2)#除法取整
print("11%2=",11%2)#除法保余数
print("11**2=",11**2)#求平方
num += 1#num=num+1
print(num)
num -= 1
print(num)
num *=4
print(num)

aaa='王晨'
print(aaa)
print(type(aaa))#单引号定义
bbb="帅哥"
print(bbb)
print(type(bbb))#双引号定义
ccc="男生"
print(ccc)
print(type(ccc))#三引号定义

name='"王晨“'#单引号可以包含双引号
print(name)
name="'王晨'"#双引号可以包含单引号
print(name)
name="\"王晨"#转义字符\可以解除引号的作用
print(name)

print("王晨，"+"今年十八岁")#加号可以连接

age=18
message="王晨今年%s"%age
print(message)#   %:我要占位   s:将变量变成字符串放入占位的地方

name='王晨'
age=18
money=18.88
message="这个男人叫%s，今年%d,有%.2f元"%(name,age,money)
print(message)

print(f"这个男人叫{name},今年{age}岁，有{money}元")

print('1*2*3的结果是:%d'%(1*2*3))
print(f'1*2*3的结果是:{1*2*3}')
print("字符串在python中的类型名是:%s" % type("字符串"))
print(f"公司:{'传音博客'}，股票代码:{'003032'}，当前股价:{10.99}")
print("每日增长系数:%.1f,经过%d天的增长后，股价达到了:%.2f"%(1.2,7,71.63))

print("请告诉我你是谁？")
name=input()
print("我知道了，你是：%s"%name)

name=input("请告诉我你是谁:")
print("我知道了，你是：%s"%name)#input内的数据通按字符串类型处理，功能：获取键盘输入的数据，与print相反

#判断语句
bool_1=True
bool_2=False#定义变量储存布尔类型的数据
print(f"bool_1变量的内容是：{bool_1}，类型是：{type(bool_1)}")                                 #判断语句

age=10
if age>=18:
    print("我已经成年了")
    print("即将步入大年生活")
print("时间过的真快啊")

age=input("请输入你的年龄：")
age=int(age)
if age>=18:
    print("欢迎来到幼儿园，儿童免费，成人收费。")
    print(f"请输入你的年龄：{age}")
    print("您已成年，游玩需要补票10元。")
    print("祝您游玩愉快。")
else:
    print("您未成年，可以免费游玩")
print("祝您玩的高兴")

print("欢迎来到动物园。")
height=int(input("请输入你的身高："))
VIP=int(input("请输入你的VIP等级："))
if height<120:
    print("您的身高小于120cm，可以免费畅玩。")
elif VIP>3:
    print("您的VIP等级大于3，可以免费畅玩。")
else:
    print("不好意思，所有条件都不满足，需要购票10元。")
print("祝您玩得愉快。")


if int(input("请输入你的身高："))<120:
    print("您的身高小于120cm，可以免费畅玩。")
elif int(input("请输入你的VIP等级："))>3:
    print("您的VIP等级大于3，可以免费畅玩。")
else:
    print("不好意思，所有条件都不满足，需要购票10元。")

age=17
year=3
level=1
if age>=18:
    print("你是个成年人")
    if age<30:
        print("年龄达标了")
        if year>2:
            print("恭喜你，年龄和入职时间都达标了，可以领取奖品了。")
        elif level>3:
            print("恭喜你，年龄和级别达标，可以领取奖品了。")
        else:
            print("不好意思，尽管年龄达标，但是入职时间和级别不达标。")
    else:
        print("不好意思，年龄太大了")
else:
    print("不好意思，年龄太小了")
"""
# 第二周

# #构建一个随机的数字变量
# import random
# num=random.randint(1,10)
# print(num)
#
# guess_num=int(input("输入你要猜的数字:"))
# #通过if判断语句进行数字的猜测
# if guess_num==num:
#     print("恭喜，第一次就猜中了")
# else:
#     if guess_num>num:
#         print("你猜测的数字大了")
#     else:
#         print("你猜测的数字小了")
#     guess_num = int(input("再次输入你要猜的数字:"))
#     if guess_num == num:
#         print("恭喜，第二次就猜中了")
#     else:
#         if guess_num > num:
#             print("你猜测的数字大了")
#         else:
#             print("你猜测的数字小了")
#         guess_num = int(input("第三次次输入你要猜的数字:"))
#         if guess_num == num:
#             print("恭喜，第三次就猜中了")
#         else:
#             print("三次机会用完了，抱歉，你没有猜中。")
"""
i=0
while i<100:
    print("王一航，我爱你")
    i+=1

sum=1
i=1
while i<100:

    i+=1
    sum+=i
print(f"1~100的和为：{sum}")



#猜数游戏
#定义一个变量来记录循环的次数
count=0
#获取1~100的随机数字
import random
num=random.randint(1,100)
#print(num)
#通过一个布尔类型的变量，做循环是否继续的标记
flag=True
while flag:
    guess_num=int(input("请输入您猜测的数字:"))
    count+=1
    if guess_num==num:

        print("恭喜您猜中了")
    #设置终止的条件
        flag=False
    else:
        if guess_num>num:
            print("您输入的数字大了")
        else:
            print("您输入的数字小了")
print(f"您总共猜测了{count}次")

#表白100天
i=1
while i<=2:
    print(f'今天是第{i}天，准备表白')
    i+=1
    j=1
    while j<=2:
        print(f"今天送她第{j}朵玫瑰花")
        j+=1
    print("我喜欢你")
print(f"坚持到第{i-1}天，表白成功")

print("Hello")
print("World")
print("Hello", end=' ')
print("World", end=' ')#print不换行
"""



# print("Hello\tWorld")
# print("itheime\tbers")

# i=1
# while i<=9:
#
#       #内层循环的print语句，不要换行，通过\t制表符进行对齐
#       j=1
#       while j<=i:
#           print(f"{i}*{j}={i*j}\t",end='')
#           j+=1
#       i+=1
#       print()  # print空内容，就是输出一个换行

# name="wangchen"
# for x in name:
#     #将name的内容挨个取出赋予x临时变量
#     #就可以在循环体内对x进行处理
#     print(x)
# #无法定义循环条件，只能被动取出数据处理
#
# name='itheima is a brand of itcast'
# i=0
# for x in name:
#     if x=="a":
#         i+=1
# print(i)
#

#range语法                 range（num)
# for x in range(10):
#     print(x)

# #range(num1,num2)
# for x in range(5,10):
#     print(x)
# #range(numi,num2,step)意思是间隔step个数字输出
# for x in range(5,10,4):
#     print(x)

# for i in range(1,101):
#     print(f"今天是向王一航表白的第{i}天")
#     for j in range(1,11):
#         print(f"送了王一航第{j}朵玫瑰")
# print(f"王一航，我爱你")

# for i in range(1,10):
#     for j in range(1,10):
#         if i<=j:
#             print(f"{i}*{j}={i*j}\t",end='')
#     print()#乘法口诀表
#
# for i in range(1,10):
#     for j in range(1,10):
#         if i>=j:
#             print(f"{i}*{j}={i*j}\t",end='')
#     print()#乘法口诀表


#continue只终止本次循环，break打破所在循环让循环终止
# money=10000;
# for i in range(1,21):
#     import random
#     score=random.randint (1,10)
#
#     if score<5:
#         print(f"员工{i}绩效分{score}，不满足，不发工资")
#         continue
#     if money>=1000:
#         money-=1000
#         print(f"员工{i}，绩效分{score}，满足，发工资1000元")
#     else:
#         print(f"员工{i}对不起，公司没钱了，发不起工资了")
# def my_love():
#     print("赵树山 ，我爱你")

# my_love()

# def add(a,b,c):                                                                                                     #括号里的叫形式参数
#     result=a+b+c
#     print(f"{a}+{b}+{c}={result}")
# add(272458356356349595635635634655924,123456786532223456743549548459858,11111111111)                                      #  括号里的叫实际参数

#
# def say_hi():
#     print("你好啊")
# result=say_hi()
# print(f"无返回值类型，返回的内容是:{result}")
# print(f"无返回值类型，返回的内容类型是:{type(result)}")

# def say_hi2():
#     print("你好啊")
#     return None
# result=say_hi2()
# print(f"无返回值类型，返回的内容是:{result}")
# print(f"无返回值类型，返回的内容类型是:{type(result)}")
#
# def check_age(age):
#     if age>=18:
#         return "SECCESS"
#     else:
#         return None
# result=check_age(18)
# if not result:
#     print("未成年，不可以进入")

#函数的嵌套
# def fun_b():
#     print("---2---")
# def fun_c():
#     print("---3---")
#     fun_b()
#     print("---4---")
#
# fun_c()
#

# def test_a():
#     num=100
#     print(num)
# test_a()
# print(num)#处理函数体，局部变量就无法使用了

# num = 200
#
#
# def test_a():
#     print(f"test_a:{num}")
#
#
# def test_b():
#     global num  # global可以将局部变量变为全局变量
#     num = 500
#     print(f"test_b:{num}")
# test_a()
# test_b()
# print(num)


# money=5000000
# name=input('请输入您的名字：')
# def aaa(show_header):#查询函数
#     if show_header:
#         print("----------------查询金额----------------")
#     print(f"剩余的存款为{money}")
# def bbb(num):#存款函数
#     global money
#     money+=num
#     print('--------------------存款--------------------')
#     print(f"您存款金额为{num}")
#     aaa(False)
#
# def ccc(num):#取款函数
#     global money
#     money-=num
#     print('------------------- 取款--------------------')
#     print(f"您取款金额为{num}")
#     aaa(False)

# def main():
#     print('-------------------主菜单--------------------')
#     print(f"{name},您好，欢迎来到银行。请操作：")
#     print('查询余额\t[输入1]')
#     print('存款\t\t[输入2]')
#     print('取款\t\t[输入3]')
#     print('退出\t\t[输入4]')
#     return input("请输入您的选择：")
#
# while True:
#     keyboard_input=main()
#     if keyboard_input=='1':
#         aaa(True)
#         continue                                                        #通过continue回到主菜单
#     elif keyboard_input=='2':
#         num=int(input("您想输入多少钱？请输入："))
#         bbb(num)
#         continue
#     elif keyboard_input=='3':
#         num=int(input('您想取出多少钱？请输入：'))
#         ccc(num)
#         continue
#     else:
#         print("程序退出了")
#         break                                                  #通过break退出循环


# my_list=['wangchen','wangyihang','hanhengyue']
# print(my_list)
# print(type(my_list))
# print(my_list[0])
# print(my_list[1])
# print(my_list[2])
# print(my_list[-1])
# print(my_list[-2])
# print(my_list[-3])


# my_list=[['wangchen','wangyihang'],['hanhengyue']]
# print(my_list)
# print(type(my_list))#嵌套列表
#
# print(my_list[0][0])
# print(my_list[0][1])
# print(my_list[1][0])

# mylist=[1,2,3,4,5,6,7,8,9,10]
#
# index=mylist.index(1)
# print(f'1在下列列表中的下标索引值是：{index}')
# index=mylist.index(0)
# print(f'1在下列列表中的下标索引值是：{index}')

# my_list=['wangchen','wangyihang','hanhengyue']
# my_list[2]='wangyihang'
# print(my_list)                                                                     #修改元素
#
# my_list.insert(1,'wangchen')
# print(my_list)                                                                     #加入元素
#
# my_list.append('shuaige')
# print(my_list)                                                                     #追加元素
#
# my_list.extend(['wangchen','wangyihang'])
# print(my_list)                                                                     #追加多个元素
#
# del my_list[2]
# print(my_list)                                                                     #删除元素的第一个方法
#
# element=my_list.pop(1)
# print(f'通过pop方法取出元素后列表内容为{my_list},去除的元素为{element}')                   #删除元素的第二个方法
#
# my_list.remove('wangchen')
# print(my_list)                                                                     #删除某元素在列表中的第一个匹配项
#
# my_list.clear()
# print(my_list)                                                                     #清除整个列表
#
# my_list=['wangchen','wangyihang','wangchen','wangyihang']
# count=my_list.count('wangchen')
# print(f"列表内‘wangchen'元素有{count}个")                                             #统计列表内某元素的数量
#
# count=len(my_list)
# print(f'列表内的元素数量总共有{count}个')
# def list_while_func():
#     my_list=['wangchen','wangyihang','wangchen','wangyihang']
#
#                                                                                  #循环控制变量通过下标索引来控制，默认零
#                                                                                  #每一次循环将下标索引变量+1
#                                                                                  #循环条件：下标索引变量<列表的元素数量
#
#     index=0                                                                                   #初始值为零
#     while index<len(my_list):
#         element=my_list[index]
#         print(f"列表的元素：{element}")
#         index+=1
#
# def list_for_func():
#     my_list=[1,2,3,4,5,6,7,8,9]
#     for element in my_list:                                                            #for循环按照顺序依次输出2列表内的元素
#         print(f'列表的元素有：{element}')
#
# list_while_func()
# list_for_func()
#
# def list_while_func1():
#     my_list=[1,2,3,4,5,6,7,8,9]
#     index=0
#     while index<len(my_list):
#         if my_list[index]%2==1:
#             del my_list[index]
#             index+=1
#
#     print(my_list)
# list_while_func1()
#
# def list_for_func2():
#     my_list1=[1,2,3,4,5,6,7,8,9]
#     my_list2=[]
#
#     for element in my_list1:
#         if element%2==0:
#             my_list2.append(element)
#     print(my_list2)
# list_for_func2()
#
#
#                                                                                         #定义元组字面量:(元素,元素,元素)
#                                                                          #tuple:元组
# #元组的嵌套
# t1=((1,2,3,4,5),(6,7,8,9))
# num=t1[0][2]
# print(num)                                                                               #下标索引去取出内容
#
# t2=(1,2,3,4,5,6,7,8,9)
# index=t2.index(9)
# print(index)                                                                             #index查找方法
#
# num=t2.count(9)
# print(num)                                                                               #count统计方法（统计元组内某个元素的个数）
#
# num= len(t2)
# print(num)                                                                               #len统计元组内元素数量
#
# index=0
# while index<len(t2):
#     print(f'元组的元素有：{t2[index]}')
#     index+=1                                                                             #元组的遍历：while
#
# for element in t2:
#     print(f'元组的元素有{element}')
#
# t9=(1,2,3,[4,5,6],[7,8,9])
# print(f't9的内容是{t9}')
# t9[3][2]='wangchen'
# t9[4][1]='wangyihang'
# print(f't9的内容是{t9}')                                                                   #修改元组内列表的内容
#
#
#
# #第三周的课
# #字符串
#                                                                 #字符串无法修改的数据容器
my_str='wangyihang and wangchen'#字符串的下标
# value=my_str[11]
# value2=my_str[-12]
# print(value)
# print(value2)
#
# value3=my_str.index('wangchen')
# print(f"在字符串中查找'wangchen',其起始下标为{value3}")#查找下标
#
# value4=my_str.replace('wangchen','王晨')
# print(f'替换字符串后得到：{value4}')#替换内容，形成新的字符串
#
# value5=my_str.split(' ')
# print(f'将字符串进行split分割后得到{value5}')                                        #分割字符串内的内容
# #
# my_str='        wangyihang and wangchen       '
# print(my_str)
# value6=my_str.strip()
# print(f'字符串被strip，结果:{value6}')                                           #strip可以去处首尾空格(不传参)
#
# my_str='12wangyihang and wangchen21'
# value7=my_str.strip('12')
# print(f'字符串被strip，结果:{value7}')                                           #strip可以去除首尾内容（传参）
#
# my_str='wangyihang and wangchen'
# value8=my_str.count('w')
# print(f'w在字符串内出现了{value8}次')
#
# value9=len(my_str)
# print(f'字符串的长度是{value9}')
#
#                                                                        #   序列的切片
#                                                                       #列表、元组、字符串取出一个子序列
# #步长表示依此取出元素的间隔
# my_list=[0,1,2,3,4,5,6,7,8,9]
# result1=my_list[4:6]
# print(result1)#对list进行切片
#
# my_list=[0,1,2,3,4,5,6,7,8,9]
# result2=my_list[5:1:-1]
# print(result2)
#
#
# my_touple=(0,1,2,3,4,5,6,7,8,9)
# result3=my_touple[:3]
# print(result3)#对touple进行切片
#
# result6=my_touple[8:0:-2]
# print(result6)
#
# my_str="0123456789"
# result4=my_str[::3]
# print(result4)#对字符串进行切片
#
#
# result5=my_str[::-1]
# print(result5)#将序列反转
#

#集合
                                                                                                    #集合不支持元素重复，且元素无序

# my_set={'王晨','王一航','王一航爱王晨','王晨说王一航是舔狗','王晨','王一航','王一航爱王晨','王晨说王一航是舔狗'}
# my_set_empty=set()#定义空集合
# print(f'my_set的内容是{my_set},类型是{type(my_set)}')
# print(f'my_set_empty的内容是{my_set_empty},类型是{type(my_set_empty)}')
#
# my_set={'wangchen','wangyihang'}
# my_set.add('and')
# print(f'加入元素后结果是{my_set}')                                               #加入一个元素
#
# my_set.remove('wangchen')
# print(f'移除元素后为{my_set}')                                        #删除一个元素
#
# my_set={'王晨','王一航','王一航爱王晨','王晨说王一航是舔狗'}
# element=my_set.pop()
# print(f'集合被取出元素是{element},取出集合后集合为{my_set}')                           #随机取出一个元素
#
# my_set.clear()
# print(f'被清空后，集合为{my_set}')                                             #清空集合
#
# #取两个集合的差集
# #结果得到一个新集合，原来两个集合不变
# set1={1,3,5,7,9}
# set2={1,2,4,6,8}
# set3=set1.difference(set2)                                           #取差集
# print(f'输出set1：{set1}')
# print(f'输出set2：{set2}')
# print(f'set1与set2的差集为{set3}')
#
# #消除两个集合的差集
# set1={1,3,5,7,9}
# set2={1,2,4,6,8}
# set1.difference_update(set2)
# print(f'消除差集后，集合1结果为{set1}')
# print(f'消除差集后，集合2结果为{set2}')
#
# set4=set1.union(set2)
# print(f'两集合和并结果{set4}')                                 #两个集合合并为一个集合
#
#
# set5=len(set4)
# print(f'集合set4的元素数量有{set5}')                              #统计集合内元素的个数
#
# set6={1,2,3,4,5,6,7,8,9}
# for element in set:
#     print(f'集合的元素有{element}')
#
# #字典
# my_dict={'王一航':59,'王晨':100,'赵树山':60}                  #定义字典
# my_dict={}#定义空字典
# my_dict=dict()
# my_dict={'王一航':59,'王晨':100,'赵树山':60}
# score=my_dict['王晨']
# print(f'王晨的考试成绩为{score}')
#
# my_score_dict={
#     '王晨':{
#         '语文':98,
#         '数学':100,
#         '英语':99
#     },
#     '王一航':{
#         '语文': 59,
#         '数学': 57,
#         '英语': 56
#     },
#     '赵树山':{
#         '语文': 60,
#         '数学': 61,
#         '英语': 59
#     }
# }                       #字典的嵌套
# print(my_score_dict)
# score=my_score_dict['王晨']['语文']
# print(f'王晨的语文成绩为{score}')
#
#
# my_dict={'王一航':59,'王晨':100,'赵树山':60}
# my_dict['帅哥']=99
# print(f'字典经过新增元素后为{my_dict}')                          #新增元素
#
# my_dict['王晨']=99.5
# print(f'字典经过更新后为{my_dict}')                             #更新元素
#
# my_dict.pop('帅哥')
# print(f'字典经过删除元素后为{my_dict}')                             #删除元素
#
# #my_dict.clear()
# #print(f'字典被清空后为{my_dict}')
#
# keys=my_dict.keys()
# print(f'字典的全部keys是{keys}')                                            #获取全部的key
#
# #遍历字典
# for key in keys:
#     print(f'字典的key为{key}')
#     print(f'字典的value是：{my_dict[key]}')#方法一
#
# for key in my_dict:
#     print(f'字典的key为{key}')
#     print(f'字典的value是：{my_dict[key]}')#方法二
#
# num=len(my_dict)
# print(f"字典内的元素有{num}个")
#
# # my_dict={'王一航':59,'王晨':100,'赵树山':60}

# my_list={1,3,5,2,4}
# my_tuple=(1,3,5,2,4)
# my_str="abcdegf"
# my_set={1,3,5,2,4}
# my_dict={'key1':5,'key2':3,'key3':1,'key4':2,'key5':4}
#
# print(f'列表  元素个数有{len(my_list)}')
# print(f'元组  元素个数有{len(my_tuple)}')
# print(f'字符串  元素个数有{len(my_str)}')
# print(f'集合  元素个数有{len(my_set)}')
# print(f'字典  元素个数有{len(my_dict)}')                                               #统计元素个数
#
# print(f'列表  元素最大值为{max(my_list)}')
# print(f'元组  元素最大值为{max(my_tuple)}')
# print(f'字符串  元素最大值为{max(my_str)}')
# print(f'集合  元素最大值为{max(my_set)}')
# print(f'字典  元素最大值为{max(my_dict)}')                                        #找出元素最大值
#
# print(f'列表  元素最小值为{min(my_list)}')
# print(f'元组  元素最小值为{min(my_tuple)}')
# print(f'字符串  元素最小值为{min(my_str)}')
# print(f'集合  元素最小值为{min(my_set)}')
# print(f'字典  元素最小值为{min(my_dict)}')                                       #找出元素最小值
#
# print(f'列表转列表的结果为{list(my_list)}')
# print(f'元组转列表的结果为{list(my_tuple)}')
# print(f'字符串转列表的结果为{list(my_str)}')
# print(f'集合转列表的结果为{list(my_set)}')
# print(f'字典转列表的结果为{list(my_dict)}')                                           #容器转列表
#
# print(f'列表转元组的结果为{tuple(my_list)}')
# print(f'元组转元组的结果为{tuple(my_tuple)}')
# print(f'字符串转元组的结果为{tuple(my_str)}')
# print(f'集合转元组的结果为{tuple(my_set)}')
# print(f'字典转元组的结果为{tuple(my_dict)}')                                       #容器转元组
#
# print(f'列表转字符串的结果为{str(my_list)}')
# print(f'元组转字符串的结果为{str(my_tuple)}')
# print(f'字符串转字符串的结果为{str(my_str)}')
# print(f'集合转字符串的结果为{str(my_set)}')
# print(f'字典转字符串的结果为{str(my_dict)}')                                   #容器转字符串
#
# print(f"列表转集合的结果为{set(my_list)}")
# print(f'元组转集合的结果为{set(my_tuple)}')
# print(f'字符串转集合的结果为{set(my_str)}')
# print(f'集合转集合的结果为{set(my_set)}')
# print(f'字典转集合的结果为{set(my_dict)}')                                                  #容器转集合
#
# print(f'列表对象的排序结果：{sorted(my_list)}')
# print(f'列表对象的排序结果：{sorted(my_tuple)}')
# print(f'列表对象的排序结果：{sorted(my_str)}')
# print(f'列表对象的排序结果：{sorted(my_set)}')
# print(f'列表对象的排序结果：{sorted(my_dict)}')                                              #给容器内元素排列
#
# print(f'列表对象的反向排序结果：{sorted(my_list,reverse=True)}')
# print(f'列表对象的反向排序结果：{sorted(my_tuple,reverse=True)}')
# print(f'列表对象的反向排序结果：{sorted(my_str,reverse=True)}')
# print(f'列表对象的反向排序结果：{sorted(my_set,reverse=True)}')
# print(f'列表对象的反向排序结果：{sorted(my_dict,reverse=True)}')                           #给容器内元素反向排列
#
# def test_return():
#     return 1,'hello',True
#
# x,y,z=test_return()
# print(x)
# print(y)
# print(z)
#
# def user_info1(name,age,gender):
#     print(f'名字是：{name},年龄是：{age},性别是：{gender}')
# user_info1('小明',20,'男')#位置参数
# user_info1(name='王一航',age=18,gender='男')#关键字参数
#
# def user_info2(name,age,gender='男'):
#     print(f'名字是：{name},年龄是：{age},性别是：{gender}')
# user_info2('王晨','18')
#
# def user_info3(*args):
#     print(f'args参数的类型为：{type(args)},内容是：{args}')
#
# user_info3(1,2,3,'王晨','王一航')
#
# def user_info4(**kwargs):
#     print(f'args参数的类型为：{type(kwargs)},内容是：{kwargs}')
#
# user_info4(name='王一航',age=18,gender='男'  )
#
# def test_func(compute):
#     result=compute(1,2)
#     print(f'compute参数的类型是：{type(compute)}')
#     print(f'计算的结果是{result}')
#
# def compute(x,y):
#     return x+y
#
# test_func(compute)                                                   #传入计算逻辑，而非传入数据
#
# def test_func(compute):
#     result=compute(1,2)
#     print(f'compute参数的类型是：{type(compute)}')
#     print(f'计算的结果是{result}')
#
# test_func(lambda x,y:x+y)

#文件编码
# UTF-8是全国通用的编码格式
#打开文件
# f=open("E:\C语言练习文件夹\while循环.cpp",'r',encoding='UTF-8')
# print(type(f))
#print(f'读取10个字节的结果：{finvalue.read(10)}')
# print(f'读取所有字节的结果：{f.read()}')                              #下一个read从上一个read的结尾处开始读

# lines=f.readlines()#读取文件的全部行，封装到列表中
# print(f'lines对象的类型是{type(lines)}')
# print(f'lines对象的内容是：{lines}')

# line1=f.readline()                                                  #读取文件的一行，封装到列表中
# line2=f.readline()
# line3=f.readline()
# line4=f.readline()

# print(f'第一行的的内容是：{line1}')
# print(f'第二行的的内容是：{line2}')
# print(f'第三行的的内容是：{line3}')
# print(f'第四行的的内容是：{line4}')

#for循环读取文件行
import time
# for line in f:
#     print(f'每一行数据是{line}')
#
# time.sleep(5000000)
# f.close()#完成文件的关闭操作
# with open("E:\C语言练习文件夹\while循环.cpp",'r',encoding='UTF-8')as f:
#     for line in f:
#         print(f'每一行的数据是{line}')                             #通过with open 语法打开文件，可以自动关闭


import time
# f=open("E:/test.txt",'w',encoding='UTF-8')                    #打开文件，不存在的文件
# f.write("Hello World")                                        #内容写在内存中
# f.flush()                                                     #将内存中积攒的内容，写入到硬盘的文件中
# time.sleep(600000)
# f.close()                                                     #close内部内置了flush

# f=open("E:/test.txt",'w',encoding='UTF-8')                            #打开文件，不存在的文件
# f.write('王晨')                                                      #w模式，文件存在，会清空原有内容

# f=open("E:/test2.txt",'a',encoding='UTF-8')                                 #文件的追加写入操作
# f.write('王晨')
# f.flush()
# f.close()

# f=open("E:/test2.txt",'a',encoding='UTF-8')                                        #a模式，文件存在，会在原有内容后面继续写入
#
# f.write('\n今年十八，是个男性')                                                       #可以使用"\n"来写出换行符
# f.close()



#第三周
#捕获异常的语法

#基本捕获语法
# try:
#     f=open("E:/python练习文件.txt","r",encoding="utf-8")
# except:
#     print("出现异常了，因为文件不存在，我将open的模式，改为w模式去打开")
#     f=open("E:/python练习文件.txt","w",encoding="utf-8")                                   #基本捕获语法


#捕获指定异常
# try:
#     print(name)
# except NameError as e:
#     print("出现了变量未定义的异常")                                                             #捕获指定异常
#     print(e)
# #捕获多个异常
# try:
#     print(name)
#
# except (NameError,ZeroDivisionError) as e:
#
#     print("出现了变量未定义的异常 或者 除以0的异常错误")

#捕获所有异常
# try:
#     print('name')
#     # 1/0
# except Exception as e:
#     print("出现异常了")                                                                          # 捕获所有的异常
# else:
#     print("王晨就是最帅的")                                  #   else在没有出错时输出else的语句
# finally:
#     print("我是finally有没有异常我都要执行")
# f.close()                                             #有无异常finally都要执行代码

                                                                                                  # 异常的传递性
# def func1():                                            #定义一个出现异常的方法
#     print("func1 正常执行")
#     num=1/0                      #肯定有异常，除以0的异常
#     print('func1 结束执行')
#
# def func2():                                            #定义一个无异常的方法，调用上面的方法
#     print('func2 开始执行')
#     func1()
#     print('func2 结束执行')
#
# def main():                                             #定义一个方法，调用上面的方法
#     try:
#         func2()
#     except Exception as e:
#         print(f'出现异常了，异常的信息是：{e}')
#
# main()


# import time                            #导入python内置的time模块（time.py这个代码文件）                    #使用import导入
# print('你好')
# time.sleep(1)                                          #通过 ' . ' 来确定层级关系
# print('我好')
#
# from time import sleep                        #使用from调用time的sleep功能
# print('你好')
# sleep(1)
# print('我好')
#
# from time import *                                             #使用*导入time模块的全部功能
# sleep(1)
# print('我好')
#
# import time as t
# print('你好')
# sleep(1)
# print('我好')
#
# from time import sleep as s                          #使用s别名可以省略
# print('你好')
# s(1)
# print('我好')

#演示自定义模块
                                               #导入自定义模块使用
# import my_module1
# my_module1.test1(11,13)
#
# from my_module1 import test1
# from my_module2 import test
#
# test(1,2)



# from my_module1 import test1                     #__main__变量：表示只有当程序是直接执行的才会进入if内部，如果是被导入的，if无法进入     if __name__ == '__main__':

# from my_module1 import *
# test1(1,2)
# test2(1,2)                                   #__all__变量                    __all__=['test1']

                                                                        #创建一个包

"""
演示python包
"""
#创建一个包
                                                                  #导入自定义的包中的模块，并使用

# import my_package.my_module3
# import my_package.my_module4
#
# my_package.my_module3.info_print1()
# my_package.my_module4.info_print2()
#
# from my_package import my_module3
# from my_package import my_module4
#
# my_module3.info_print1()
# my_module4.info_print2()

# from my_package import *
# my_module3.info_print1()
# #my_module4.info_print2()                                                    #通过__all__变量，控制import

#导包
from pyecharts.charts import Line
from pyecharts.options import TitleOpts, LegendOpts, ToolboxOpts, VisualMapOpts

#创建一个折线图对象
line=Line()
#给折线图对象添加x轴的数据
line.add_xaxis(['中国','美国','英国'])
#给折线图对象添加y轴的数据
line.add_yaxis("GDP",[30,20,10])
#通过render方法，将代码生成为图像
line.render()

#设置全局配置项set_global_opts来设置
line.set_global_opts(
    title_opts=TitleOpts(title='GDP展示',pos_left='center',pos_bottom='1%'),
    legend_opts=LegendOpts(is_show=True),
    toolbox_opts=ToolboxOpts(is_show=True),
    visualmap_opts=VisualMapOpts(is_show=True)

)
line.render()






























