---
title: '[Note] Data structure in python'
date: 2024-02-07
permalink: /posts/2024/02/07/data-structure-in-python/
tags:
  - paper
  - writing
  - science
  - data structure
---


1. Singly Linked List (SLL)
======

```python
# Create a Singly LinkedList with below Properties (1.Head,2.Tail,3.Length) 
# methods (1.Push,2.Pop,3.Shift,4.Unshift,5.Get,6.Set,7.Insert,8.Remove,9.Reverse) 
import treevizer 
class Node: 
    def __init__(self,val): 
        self.data = val 
        self.next = None 
 
class SinglyLinkedList: 
    def __init__(self): 
        self.head = None 
        self.tail = None 
        self.length = 0 
 
    def print(self): 
        treevizer.to_png(self.head, structure_type="ll", dot_path="sll.dot", 
png_path="sll.png") 
 
    def push(self,nval): 
        nd = Node(nval) 
        if self.head is None: 
            self.head = nd 
            self.tail = self.head 
        else: 
            self.tail.next = nd 
            self.tail = nd 
        self.length +=1 
 
    def pop(self): 
        if self.length == 0: return None 
        if self.length == 1: 
            self.head = None 
            self.tail = None 
        else: 
            cNode = self.head 
            for x in range(1, self.length-1): 
                cNode = cNode.next 
            self.tail = cNode 
            cNode.next =None 
 
        self.length -=1 
        return self 
 
    def get(self, index): 
        if self.length > index and index >= 0: 
            cNode = self.head 
            if index == 0: return cNode.data 
            for x in range(1, index+1): 
                cNode = cNode.next 
            return cNode.data 
        return None 
	def set(self, index,nval): 
        if self.length > index and index >= 0: 
            cNode = self.head 
            if index == 0: 
                cNode.data = nval 
                return True 
            for x in range(1, index+1): 
                cNode = cNode.next 
            cNode.data = nval 
            return True 
        return False 
 
    def reverse(self): 
        prevNode, curNode = None, self.head 
        self.tail = curNode 
        while (curNode): 
            nextNode = curNode.next 
            curNode.next = prevNode 
            prevNode = curNode 
            curNode = nextNode 
        self.head = prevNode 
 
 
    def reversePos(self,start,end): 
        if start > 0 and end <self.length -1: 
            sNode = eNode = self.head 
            for x in range(1,start): 
                sNode = sNode.next 
            for x in range(1,end+1): 
                eNode = eNode.next 
 
            headNode = sNode 
            tailNode = eNode.next 
 
            prevNode = tailNode 
            curNode = headNode.next 
            while(curNode is not tailNode): 
                nextNode = curNode.next 
                curNode.next = prevNode 
                prevNode = curNode 
                curNode = nextNode 
            headNode.next = prevNode 
 
    def printSll(self): 
        cNode = self.head 
        lst = [] 
        while(cNode): 
            lst.append(cNode.data) 
            cNode = cNode.next 
        print(lst) 
		
if __name__ == '__main__': 
    sll = SinglyLinkedList() 
    for x in range(1,11): sll.push(x) 
    sll.print() 
 
    sll.printSll() 
    sll.reversePos(1,7) 
    sll.printSll() 
    print('Head [', sll.head.data,']') 
    print('Tail [',sll.tail.data,']')
```

2. Doubly Linked List (DLL)
======

```python
# Create a Doubly LinkedList with below properties (1.Head,2.Tail,3.Length) 
# Problem: DLL FLattening with ChildDLLs or SubChildDLLs 
import treevizer 
 
class Node: 
    def __init__(self,val=None): 
        self.next = None 
        self.prev = None 
        self.data = val 
        self.child = None 
 
class DoublyLinkedList: 
    def __init__(self): 
        self.head = None 
        self.tail = None 
        self.length = 0 
 
    def print(self): 
        treevizer.to_png(self.head, structure_type="ll", dot_path="dll.dot", 
png_path="dll.png") 
 
    def addChildDLL(self,index,chlDLL): 
        if index<0 or index>self.length-1: return None 
        if index==0: 
            cNode = self.head 
        elif index==self.length-1: 
            cNode = self.tail 
        else: 
            cNode = self.head 
            for x in range(1,index+1): cNode = cNode.next 
 
        cNode.child = chlDLL.head 
 
    def flattenDLL(self): 
        psNode = self.head 
        while(psNode): 
            if psNode.child is not None: 
                peNode = psNode.next 
                csNode = psNode.child 
                while(csNode): ceNode = csNode; csNode = csNode.next 
                # ------------------------ 
                psNode.next = psNode.child 
                psNode.child.prev = psNode 
                # ------------------------ 
                ceNode.next = peNode 
                peNode.prev = ceNode 
                # ------------------------ 
                psNode.child = None 
                #------------------------- 
            psNode = psNode.next
	
	def push(self,nval): 
        nd = Node(nval) 
        if self.head is None: 
            self.head = nd 
            self.tail = self.head 
        else: 
            self.tail.next = nd 
            nd.prev = self.tail 
            self.tail = nd 
        self.length +=1 
 
    def printDll(self): 
        # nd=self.head 
        # while nd: print(nd.stage,end='<->'); nd=nd.next 
        # print("\n") 
        print(self.buildMap(self.head)) 
 
    def buildMap(self,pNode): 
        tempMap = {} 
        while(pNode): 
            if pNode.child is not None: 
                tempMap[pNode.data] = self.buildMap(pNode.child) 
            else: 
                tempMap[pNode.data] = {} 
            pNode = pNode.next 
        return tempMap

if __name__ == '__main__': 
    dll = DoublyLinkedList(); dll1 = DoublyLinkedList(); dll3 = DoublyLinkedList() 
    dll11 = DoublyLinkedList(); dll33 = DoublyLinkedList() 
 
    for x in range(0,10): dll.push(x) 
    for x in range(10,15): dll1.push(x) 
    for x in range(30,40): dll3.push(x) 
    for x in range(120,125): dll11.push(x) 
    for x in range(330,333): dll33.push(x) 
 
    dll1.addChildDLL(2,dll11) 
    dll3.addChildDLL(3,dll33) 
    dll.addChildDLL(1,dll1); dll.addChildDLL(3,dll3) 
 
    dll.printDll() 
    dll.flattenDLL()   # DLL Flattening Test 
    dll.printDll() 
```

Hết.
