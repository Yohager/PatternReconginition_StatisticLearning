# -*- coding: utf-8 -*-
#author: yohager
#OS experiments 1 : process and resources


class RCB:
    def __init__(self):
        self.Rid = None
        self.Initial = 0
        self.Remain = 0
        self.Waiting_list = []

class process_handle_sector:
    def __init__(self):
        #定义资源1的阻塞队列
        self.R1 = RCB()
        self.R1.Rid = 'R1'
        self.R1.Initial = 1
        self.R1.Remain = 1

        self.R2 = RCB()
        self.R2.Rid = 'R2'
        self.R2.Initial = 2
        self.R2.Remain = 2

        self.R3 = RCB()
        self.R3.Rid = 'R3'
        self.R3.Initial = 3
        self.R3.Remain = 3

        self.R4 = RCB()
        self.R4.Rid = 'R4'
        self.R4.Initial = 4
        self.R4.Remain = 4
        #初始化三个不同优先级的就绪队列
        self.RL = [[],[],[]]
        self.current_process = None
        self.current_process_pid = -1
        self.RCB_list = [self.R1,self.R2,self.R3,self.R4]
        self.resource_list = [self.R1.Waiting_list,
                              self.R2.Waiting_list,
                              self.R3.Waiting_list,
                              self.R4.Waiting_list]
        self.destroyed_process = []

class PCB:
    def __init__(self):
        self.Pid = None
        self.type = 'ready' #对于一个进程来说：ready/running/blocked
        self.resources = []
        self.resources_num = []
        self.priority = None
        self.request_resources = None
        self.request_resources_num = -1
        self.parent = None
        self.children = []


#初始化系统
def init_function(init):
    #这里需要加东西
    return init

#用于更新当前运行的进程，并给出输出
def scheduler(process_sector):
    #print(process_sector.current_process.Pid)
    return process_sector.current_process.Pid

#创建一个进程
def create_process(command,process_sector):
    #create PCB data structure
    process = PCB()
    #initialize PCB using parameters 包括进程的ID，优先级、状态等
    process.Pid = command[1]
    process.priority = int(command[2])
    if process_sector.current_process != None:
        process.parent = process_sector.current_process
        process_sector.current_process.children.append(process)
    #寻找当前执行的进程，并将原本在就绪队列中的进程pop出来然后改为正在运行的程序
    if process.priority == 2:
        process_sector.RL[0].append(process)
        process.type = 'ready'
        #优先级为2的进程，直接放入就绪队列1，并且判断是否是就绪队列1中的唯一元素，如果是直接执行？是否有问题？
    elif process.priority == 1:
        process_sector.RL[1].append(process)
        process.type = 'ready'
    else:
        #此时为优先级为0的进程，虚拟进程
        process_sector.RL[2].append(process)
        process.type = 'Virtual'
    '''
    上面的思路很简单，创建进程后先将创建的进程放入就绪队列中，其他先不动，下面再考虑当前进程的问题
    '''
    if process_sector.current_process == None:
        if len(process_sector.RL[0]) != 0:
            # 如果system级别的进程就绪队列不为空的话，从就绪队列2中取出system级别的队首就绪进程作为当前进程
            process_sector.current_process = process_sector.RL[0][0]
            process_sector.current_process.Pid = process_sector.RL[0][0].Pid
            process_sector.RL[0][0].type = 'running'
            process_sector.RL[0][0].parent = 'root'
            process_sector.RL[0].pop(0)
        elif len(process_sector.RL[0]) == 0 and len(process_sector.RL[1]) != 0:
            process_sector.current_process = process_sector.RL[1][0]
            process_sector.current_process.Pid = process_sector.RL[1][0].Pid
            process_sector.RL[1][0].type = 'running'
            process_sector.RL[1][0].parent = 'root'
            process_sector.RL[1].pop(0)
        else:
            print('ERROR,process is a virtual process')
    else:
        if process_sector.current_process.priority == 2:
            pass
        elif process_sector.current_process.priority == 1:
            '''
            这个地方不存在当前进程优先级为1同时优先级为2的就绪队列中还有很多进程！
            '''
            if len(process_sector.RL[0])!=0:
                #这种情况只存在一种可能性就是刚刚创建的进程优先级为2
                process_sector.current_process.type = 'ready'
                process_sector.RL[1].append(process_sector.current_process)
                process_sector.current_process = process_sector.RL[0][0]
                process_sector.current_process_pid = process_sector.RL[0][0].Pid
                process_sector.RL[0][0].type = 'running'
                process_sector.RL[0].pop(0)
            else:
                pass
        else:
            pass

def block2ready_function(waiting_list,RCB_use,ready_list):
    count = 0
    RCB_use_cal_remain = RCB_use.Remain
    #print('RCB_use',RCB_use.Rid)
    if len(waiting_list) != 0:
        #计算在阻塞队列中有多少进程在释放资源后可以进入就绪队列
        for i in waiting_list:
            if i.request_resources_num <= RCB_use_cal_remain:
                count +=1
                RCB_use_cal_remain -= i.request_resources_num
            else:
                RCB_use_cal_remain -= i.request_resources_num
        #print('可以从阻塞队列出来的进程数为：',count)
        #count计算了所有可以进入就绪队列的阻塞进程数量，遍历这些进程，拿走资源且进入就绪状态
        if count <= 0:
            #表示释放的资源连阻塞队列的第一个进程都无法满足
            pass
        else:
            #这种情况下释放的资源可以满足一些阻塞的进程使用
            for j in range(count):
                #print(waiting_list[j].priority)
                waiting_list[j].type = 'ready'
                waiting_list[j].resources.append(RCB_use.Rid)
                #print('此处的资源为：',waiting_list[j].resources)
                waiting_list[j].resources_num.append(waiting_list[j].request_resources_num)
                RCB_use.Remain -= waiting_list[j].request_resources_num
                waiting_list[j].request_resources = None
                waiting_list[j].request_resources_num = -1
                if waiting_list[j].priority == 2:
                    ready_list[0].append(waiting_list[j])
                    waiting_list.pop(j)
                elif waiting_list[j].priority == 1:
                    ready_list[1].append(waiting_list[j])
                    #print('The process has been insert to RList1!')
                    waiting_list.pop(j)
                else:
                    ready_list[2].append(waiting_list[j])
                    waiting_list.pop(j)
    else:
        #此时阻塞队列为空，不进行任何操作
        pass

#对于单个进程进行销毁
def destroyed_single_process(process,process_sector):
    process.type = 'destroyed'
    process_sector.destroyed_process.append(process)
    #print('正在销毁的进程的PID以及占用资源的数量：',process.Pid,process.resources)
    #if process.Pid == 'B':
        #print('当前R3的资源数量：',process_sector.R3.Remain)
        #print('进程B的资源：',process.resources[0],process.resources_num[0])
    if process.resources == []:
        pass
    else:
        for i in range(len(process.resources)):
            if process.resources[i] == 'R1':
                process_sector.R1.Remain += process.resources_num[i]
            elif process.resources[i] == 'R2':
                process_sector.R2.Remain += process.resources_num[i]
            elif process.resources[i] == 'R3':
                process_sector.R3.Remain += process.resources_num[i]
            elif process.resources[i] == 'R4':
                process_sector.R4.Remain += process.resources_num[i]
            else:
                print('error,this process does not have this kind of resource!')
    #释放资源后判断是否存在在阻塞队列中的进程可以进入就绪队列
    for j in range(len(process_sector.resource_list)):
        block2ready_function(process_sector.resource_list[j],process_sector.RCB_list[j],process_sector.RL)
    if process.Pid == process_sector.current_process_pid:
        if len(process_sector.RL[0]) != 0:
            process_sector.current_process = process_sector.RL[0][0]
            process_sector.current_process_pid = process_sector.RL[0][0].Pid
            process_sector.RL[0].pop(0)
        elif len(process_sector.RL[0]) == 0 and len(process_sector.RL[1]) != 0:
            process_sector.current_process = process_sector.RL[1][0]
            process_sector.current_process_pid = process_sector.RL[1][0].Pid
            process_sector.RL[1].pop(0)
    elif process in process_sector.RL[0]:
        process_sector.RL[0].remove(process)
    elif process in process_sector.RL[1]:
        process_sector.RL[1].remove(process)
    elif process in process_sector.R1.Waiting_list:
        process_sector.R1.Waiting_list.remove(process)
    elif process in process_sector.R2.Waiting_list:
        process_sector.R2.Waiting_list.remove(process)
    elif process in process_sector.R3.Waiting_list:
        process_sector.R3.Waiting_list.remove(process)
    elif process in process_sector.R4.Waiting_list:
        process_sector.R4.Waiting_list.remove(process)
    else:
        print('Error,can not find where is the process')

#递归删除进程及其子进程（树状结构）
def destroy_recursion(process,process_sector):
    destroyed_single_process(process,process_sector)
    for j in range(len(process_sector.resource_list)):
        block2ready_function(process_sector.resource_list[j],process_sector.RCB_list[j],process_sector.RL)
    if process.children != []:
        for i in process.children:
            destroy_recursion(i,process_sector)


def destroy_process(command,process_sector):
    destroy_pid = command[1]
    '''
    需要销毁的进程一共有三大类，第一类是当前进程，第二类是在就绪队列中，第三类是在阻塞队列中
    '''
    if destroy_pid == process_sector.current_process_pid:
        #print('需要销毁的进程是当前执行的进程！')
        destroy_recursion(process_sector.current_process,process_sector)
    if len(process_sector.RL[0]) != 0:
        for i in process_sector.RL[0]:
            if destroy_pid == i.Pid:
                #print('需要销毁的进程在就绪队列0')
                destroy_recursion(i,process_sector)
    if len(process_sector.RL[1]) !=0:
        for j in process_sector.RL[1]:
            if destroy_pid == j.Pid:
                #print('需要销毁的进程在就绪队列1')
                destroy_recursion(j,process_sector)
    if len(process_sector.R1.Waiting_list)!=0:
        for k in process_sector.R1.Waiting_list:
            if destroy_pid == k.Pid:
                #print('需要销毁的进程在资源1的阻塞队列中')
                destroy_recursion(k,process_sector)
    if len(process_sector.R2.Waiting_list)!=0:
        for m in process_sector.R2.Waiting_list:
            if destroy_pid == m.Pid:
                #print('需要销毁的进程在资源2的阻塞队列中')
                destroy_recursion(m,process_sector)
    if len(process_sector.R3.Waiting_list) != 0:
        for n in process_sector.R3.Waiting_list:
            if destroy_pid == n.Pid:
                #print('需要销毁的进程在资源3的阻塞队列中')
                destroy_recursion(n,process_sector)
    if len(process_sector.R4.Waiting_list) != 0:
        for q in process_sector.R4.Waiting_list:
            if destroy_pid == q.Pid:
                #print('需要销毁的进程在资源4的阻塞队列中')
                destroy_recursion(q,process_sector)

#时间周期转完更换当前进程
def change_current_process(process_sector):
    if len(process_sector.RL[0]) != 0:
        if process_sector.current_process.priority == 2:
            process_sector.current_process.type = 'ready'
            process_sector.RL[0].append(process_sector.current_process)
            process_sector.current_process = process_sector.RL[0][0]
            process_sector.current_process_pid = process_sector.RL[0][0].Pid
            process_sector.RL[0].pop(0)
        elif process_sector.current_process.priority == 1:
            process_sector.current_process.type = 'ready'
            process_sector.RL[1].append(process_sector.current_process)
            process_sector.current_process = process_sector.RL[0][0]
            process_sector.current_process_pid = process_sector.RL[0][0].Pid
            process_sector.RL[1].pop(0)
    elif len(process_sector.RL[0]) == 0 and len(process_sector.RL[1])!=0:
        #这里表示就绪队列1为空，没有优先级为2的进程在等待，直接考虑就绪队列2
        if process_sector.current_process.priority == 1:
            process_sector.current_process.type = 'ready'
            process_sector.RL[1].append(process_sector.current_process)
            process_sector.current_process = process_sector.RL[1][0]
            process_sector.current_process_pid = process_sector.RL[1][0].Pid
            process_sector.RL[1].pop(0)
        else:
            pass
    else:
        #这里表示两个就绪队列都为空，则当前没有其他的进程，这里不考虑第三优先级的进程
        pass

#请求资源操作
def request_resources(command,process_sector):
    request_r = command[1]
    request_num = int(command[2])
    process_sector.current_process.request_resources = request_r
    process_sector.current_process.request_resources_num = request_num
    if process_sector.current_process.request_resources == 'R1':
        if process_sector.current_process.request_resources_num <= process_sector.R1.Remain:
        #此时可以给当前进程分配资源
            process_sector.R1.Remain -= process_sector.current_process.request_resources_num
            process_sector.current_process.resources.append('R1')
            process_sector.current_process.resources_num.append(process_sector.current_process.request_resources_num)
            process_sector.current_process.request_resources = None
            process_sector.current_process.request_resources_num = -1
        else:
        #此时不能分配资源
            process_sector.current_process.request_resources = request_r
            process_sector.current_process.request_resources_num = request_num
            process_sector.current_process.type = 'blocked'
            process_sector.resource_list[0].append(process_sector.current_process)
            if len(process_sector.RL[0])!=0:
                process_sector.current_process = process_sector.RL[0][0]
                process_sector.current_process_pid = process_sector.RL[0][0].Pid
                process_sector.RL[1].pop(0)
            elif len(process_sector.RL[0])==0 and len(process_sector.RL[1])!=0:
                process_sector.current_process = process_sector.RL[1][0]
                process_sector.current_process_pid = process_sector.RL[1][0].Pid
                process_sector.RL[1].pop(0)
            else:
                pass
    elif process_sector.current_process.request_resources == 'R2':
        if process_sector.current_process.request_resources_num <= process_sector.R2.Remain:
        #此时可以给当前进程分配资源
            process_sector.R2.Remain -= process_sector.current_process.request_resources_num
            process_sector.current_process.resources.append('R2')
            process_sector.current_process.resources_num.append(process_sector.current_process.request_resources_num)
            process_sector.current_process.request_resources = None
            process_sector.current_process.request_resources_num = -1
        else:
        #此时不能分配资源
            process_sector.current_process.request_resources = request_r
            process_sector.current_process.request_resources_num = request_num
            process_sector.current_process.type = 'blocked'
            process_sector.resource_list[1].append(process_sector.current_process)
            if len(process_sector.RL[0])!=0:
                process_sector.current_process = process_sector.RL[0][0]
                process_sector.current_process_pid = process_sector.RL[0][0].Pid
                process_sector.RL[1].pop(0)
            elif len(process_sector.RL[0])==0 and len(process_sector.RL[1])!=0:
                process_sector.current_process = process_sector.RL[1][0]
                process_sector.current_process_pid = process_sector.RL[1][0].Pid
                process_sector.RL[1].pop(0)
            else:
                pass
    elif process_sector.current_process.request_resources == 'R3':
        if process_sector.current_process.request_resources_num <= process_sector.R3.Remain:
        #此时可以给当前进程分配资源
            process_sector.R3.Remain -= process_sector.current_process.request_resources_num
            process_sector.current_process.resources.append('R3')
            process_sector.current_process.resources_num.append(process_sector.current_process.request_resources_num)
            process_sector.current_process.request_resources = None
            process_sector.current_process.request_resources_num = -1
        else:
        #此时不能分配资源
            process_sector.current_process.request_resources = request_r
            process_sector.current_process.request_resources_num = request_num
            process_sector.current_process.type = 'blocked'
            process_sector.resource_list[2].append(process_sector.current_process)
            if len(process_sector.RL[0])!=0:
                process_sector.current_process = process_sector.RL[0][0]
                process_sector.current_process_pid = process_sector.RL[0][0].Pid
                process_sector.RL[0].pop(0)
            elif len(process_sector.RL[0])==0 and len(process_sector.RL[1])!=0:
                process_sector.current_process = process_sector.RL[1][0]
                process_sector.current_process_pid = process_sector.RL[1][0].Pid
                process_sector.RL[1].pop(0)
            else:
                pass
    elif process_sector.current_process.request_resources == 'R4':
        if process_sector.current_process.request_resources_num <= process_sector.R4.Remain:
        #此时可以给当前进程分配资源
            process_sector.R4.Remain -= process_sector.current_process.request_resources_num
            process_sector.current_process.resources.append('R4')
            process_sector.current_process.resources_num.append(process_sector.current_process.request_resources_num)
            process_sector.current_process.request_resources = None
            process_sector.current_process.request_resources_num = -1
        else:
        #此时不能分配资源
            process_sector.current_process.request_resources = request_r
            process_sector.current_process.request_resources_num = request_num
            process_sector.current_process.type = 'blocked'
            process_sector.resource_list[3].append(process_sector.current_process)
            if len(process_sector.RL[0])!=0:
                process_sector.current_process = process_sector.RL[0][0]
                process_sector.current_process_pid = process_sector.RL[0][0].Pid
                process_sector.RL[1].pop(0)
            elif len(process_sector.RL[0])==0 and len(process_sector.RL[1])!=0:
                process_sector.current_process = process_sector.RL[1][0]
                process_sector.current_process_pid = process_sector.RL[1][0].Pid
                process_sector.RL[1].pop(0)
            else:
                pass
    else:
        print('Error,没有这种类型的资源可以申请！')

#释放资源
def release_resources(command,process_sector):
    release_r = command[1]
    release_num = int(command[2])
    #print('应该释放的资源为：',release_r,release_num)
    if release_r == 'R1':
        if 'R1' in process_sector.current_process.resources:
            release_index_r1 = process_sector.current_process.resources.index('R1')
            process_sector.current_process.resources_num[release_index_r1] -= release_num
            process_sector.R1.Remain += release_num
            block2ready_function(process_sector.R1.Waiting_list,process_sector.R1,process_sector.RL)
        else:
            print('ERROR,CurrentProcess does not own this resource!')
    elif release_r == 'R2':
        if 'R2' in process_sector.current_process.resources:
            release_index_r2 = process_sector.current_process.resources.index('R2')
            process_sector.current_process.resources_num[release_index_r2] -= release_num
            process_sector.R2.Remain += release_num
            block2ready_function(process_sector.R2.Waiting_list,process_sector.R2,process_sector.RL)
        else:
            print('ERROR,CurrentProcess does not own this resource!')
    elif release_r == 'R3':
        if 'R3' in process_sector.current_process.resources:
            release_index_r3 = process_sector.current_process.resources.index('R3')
            #print('在资源中的索引：',release_index_r3)
            process_sector.current_process.resources_num[release_index_r3] -= release_num
            #print('增加之前R3的数量',process_sector.R3.Remain)
            process_sector.R3.Remain += release_num
            #print('增加之后R3的数量',process_sector.R3.Remain)
            block2ready_function(process_sector.R3.Waiting_list,process_sector.R3,process_sector.RL)
        else:
            print('ERROR,CurrentProcess does not own this resource!')
    elif release_r == 'R4':
        if 'R4' in process_sector.current_process.resources:
            release_index_r4 = process_sector.current_process.resources.index('R4')
            process_sector.current_process.resources_num[release_index_r4] -= release_num
            process_sector.R4.Remain += release_num
            block2ready_function(process_sector.R4.Waiting_list,process_sector.R4,process_sector.RL)
        else:
            print('ERROR,CurrentProcess does not own this resource!')
    else:
        print('ERROR!检查输入文件是否错误！')
    if process_sector.current_process.priority == 2:
        pass
    elif process_sector.current_process.priority == 1:
        if len(process_sector.RL[0])!=0:
        #在释放了进程之后，存在优先级更高的进程从阻塞队列中出来，需要抢占运行
            process_sector.current_process.type = 'ready'
            process_sector.RL[1].append(process_sector.current_process)
            process_sector.current_process = process_sector.RL[0][0]
            process_sector.current_process_pid = process_sector.RL[0][0].Pid
            process_sector.RL[0][0].type = 'running'
            process_sector.RL[0].pop(0)

#读取txt文件
def load_test_shell(filepath):
    shell_txt = []
    with open(filepath,'r') as file_load:
        while True:
            lines = file_load.readline()
            if not lines:
                break
                pass
            shell_txt.append(lines.split())
    return shell_txt

#主函数
def main_function(shell_text,process_sector):
    init_process = init_function('init')
    result = []
    result.append(init_process)
    for i in shell_text:
        if i[0] == 'cr':
            create_process(i,process_sector)
            result.append(scheduler(process_sector))
        elif i[0] == 'to':
            change_current_process(process_sector)
            result.append(scheduler(process_sector))
        elif i[0] == 'req':
            request_resources(i,process_sector)
            result.append(scheduler(process_sector))
        elif i[0] == 'de':
            destroy_process(i,process_sector)
            result.append(scheduler(process_sector))
        elif i[0] == 'rel':
            release_resources(i,process_sector)
            result.append(scheduler(process_sector))
        else:
            print('Error!检查输入文件！')
    return result

if __name__ == '__main__':
    shell_txt = load_test_shell('3.txt')
    #print(shell_txt)
    process_sector = process_handle_sector()
    process_result = main_function(shell_txt,process_sector)
    for i in process_result:
        print(i,end=' ')
    print('\n')
    print('**********this is for debug**************')
    #print('the length of ready_List1',len(process_sector.RL[1]))
    #print('destroyed process list',process_sector.destroyed_process[0].Pid)
    print('当前进程的PID',process_sector.current_process_pid)
    print('当前进程的资源占用：',process_sector.current_process.resources)
    print('当前进程的资源占用数量：',process_sector.current_process.resources_num)
    #print('当前R3的资源数量',process_sector.R3.Remain)
    if len(process_sector.RL[0])!=0:
        print('在就绪队列0中（system级别的进程）进程PID：',end=' ')
        for j in process_sector.RL[0]:
            print(j.Pid,end=' ')
        print('\r')
    else:
        print('就绪队列0为空！\r')
    if len(process_sector.RL[1])!=0:
        print('在就绪队列1中（优先级为1级别的进程）进程PID：',end=' ')
        for i in process_sector.RL[1]:
            print(i.Pid,end=' ')
        print('\r')
    else:
        print('就绪队列1为空！\r')
        '''
    for i in process_sector.RL[1]:
        if i.Pid == 'B':
            print('进程B的资源以及数量',i.resources,i.resources_num)
            '''
    if len(process_sector.R1.Waiting_list)!=0:
        print('在资源1的阻塞队列中的进程PID：',end=' ')
        for k in process_sector.R1.Waiting_list:
            print(k.Pid,end=' ')
        print('\r')
    else:
        print('资源1的阻塞队列为空！\r')
    if len(process_sector.R2.Waiting_list)!=0:
        print('在资源2的阻塞队列中的进程PID：',end=' ')
        for m in process_sector.R2.Waiting_list:
            print(m.Pid,end=' ')
        print('\r')
    else:
        print('资源2的阻塞队列为空！\r')
    if len(process_sector.R3.Waiting_list)!=0:
        print('在资源3的阻塞队列中的进程PID：', end=' ')
        for n in process_sector.R3.Waiting_list:
            print(n.Pid,end=' ')
            print('阻塞进程的情况：',n.request_resources,n.request_resources_num)
            #print('进程B的资源申请情况',n.request_resources,n.request_resources_num)
        print('\r')
    else:
        print('资源3的阻塞队列为空！\r')
    if len(process_sector.R4.Waiting_list)!=0:
        print('在资源4的阻塞队列中的进程PID：', end=' ')
        for q in process_sector.R4.Waiting_list:
            print(q.Pid,end=' ')
        print('\r')
    else:
        print('资源4的阻塞队列为空！\r')
    if len(process_sector.destroyed_process)!=0:
        print('已经被销毁的进程PID：',end=' ')
        for s in process_sector.destroyed_process:
            print(s.Pid,end=' ')
        print('\r')
    else:
        print('销毁的进程队列为空！\r')
    print('\r')
    print('**********Resources Situation*********')
    print('Resources Remains:R1:%d个,R2:%d个,R3:%d个,R4:%d个'%(process_sector.R1.Remain,process_sector.R2.Remain,process_sector.R3.Remain,process_sector.R4.Remain))
    ''' 
    print('the length of R3 Waiting list',process_sector.R3.Waiting_list[0].Pid)
    print('R3 remains',process_sector.R3.Remain)
    print('destroyed process list',process_sector.destroyed_process[0].Pid)
    '''


