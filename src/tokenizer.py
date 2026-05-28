class Node:
    __slots__ = ['value', 'prev', 'next']
    
    def __init__(self,x):
        self.value = x
        self.prev = None
        self.next = None

import heapq
    
class Tokenizer:
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.encoder_rules = {}
        self.decoder_rules = {i: bytes([i]) for i in range(256)}
        self.merge_rules_list = []
        
    def createList(self, array:list) -> Node:
        previous = None
        head = None
        
        for n in array:
            node = Node(n)
            node.prev = previous
            if node.prev != None:
                node.prev.next = node
            else:
                head = node
                
            previous = node
            
        return head
    

    def train(self, corpus_path: str) -> None:
        with open(corpus_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        content_byte = content.encode('utf-8')
        content_byte = list(content_byte)
        
        head = self.createList(content_byte)
        
        from collections import defaultdict
        freq = defaultdict(int)
        curr = head
        
        while curr!=None and curr.next!=None:
            freq[(curr.value,curr.next.value)] += 1
            curr = curr.next
            
        occur = set()
        
        while len(self.merge_rules_list) < self.vocab_size - 256:
            maxx = 0
            target = None
            for key,value in freq.items():
                if value>maxx: 
                    maxx = value
                    target = key
                    
            id = 255 + len(self.merge_rules_list) + 1
            self.merge_rules_list.append(target)
            if id not in occur:
                occur.add(target)
                self.encoder_rules[target] = id
            
            curr = head
            while curr!=None and curr.next!=None:
                if (curr.value, curr.next.value) == target:
                    new_node = Node(id)
                    new_node.next = curr.next.next
                    new_node.prev = curr.prev
                    if curr.prev != None:
                        curr.prev.next = new_node
                        freq[(curr.prev.value,curr.value)] -= 1
                        if freq[(curr.prev.value,curr.value)] ==0: del freq[(curr.prev.value,curr.value)]
                        freq[(new_node.prev.value,new_node.value)] += 1
                    if curr.next.next != None:
                        curr.next.next.prev = new_node
                        freq[(curr.next.value,curr.next.next.value)] -= 1
                        if freq[(curr.next.value,curr.next.next.value)] == 0: del freq[(curr.next.value,curr.next.next.value)]
                        freq[(new_node.value,new_node.next.value)] += 1
                    freq[(curr.value,curr.next.value)] -=1
                    if freq[(curr.value,curr.next.value)] == 0: del freq[(curr.value,curr.next.value)]
                    if curr == head:
                        head = new_node
                    curr.prev, curr.next.next, curr.next.prev, curr.next = None,None,None,None
                    curr = new_node.next
                
                else:
                    curr = curr.next
            
        

    def encode(self, text: str) -> list[int]:
        pass

    def decode(self, tokens: list[int]) -> str:
        pass

    def save_to_json(self, file_path: str) -> None:
        pass

    def load_from_json(self, file_path: str) -> None:
        pass
    
corpus_path = "src/test.txt"

tokenizer = Tokenizer()
tokenizer.train(corpus_path)
print(tokenizer.merge_rules_list)
