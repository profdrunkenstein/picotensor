class Tensor:
    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return f"Tensor({self.data})"

    def __add__(self, other):
        if len(self.data) != len(other.data):
            raise ValueError("Tensors must be the same length!")
        
        result = [x + y for x, y in zip(self.data, other.data)]
        return Tensor(result)

    def __sub__(self, other):
        if len(self.data) != len(other.data):
            raise ValueError("Tensors must be the same length!")
        
        result = [x - y for x, y in zip(self.data, other.data)]
        return Tensor(result)

    def __mul__(self, other):
        if len(self.data) != len(other.data):
            raise ValueError("Tensors must be the same length!")
        
        result = [x * y for x, y in zip(self.data, other.data)]
        return Tensor(result)

    def __truediv__(self, other):
        if len(self.data) != len(other.data):
            raise ValueError("Tensors must be the same length!")
        
        for y in other.data:
            if y == 0:
                raise ZeroDivisionError("division by zero")
        
        result = [x / y for x, y in zip(self.data, other.data)]
        return Tensor(result)

    def sum(self):
        total = 0
        for x in self.data:
            total += x
        return total
        
    def __repr__(self):
     return f"Tensor({self.data})"
    
    def __len__(self):
     return len(self.data)
    
    def __getitem__(self, idx):
     return self.data[idx]

    def __setitem__(self, idx, value):
     self.data[idx] = value
     
    def mean(self):
     return self.sum() / len(self.data)
     
    def max(self):
     if len(self.data) == 0:
         raise ValueError("Empty Tensor")

     m = self.data[0]
     for x in self.data:
         if x > m:
             m = x
     return m
    
    def min(self):
     if len(self.data) == 0:
         raise ValueError("Empty Tensor")

     m = self.data[0]
     for x in self.data:
         if x < m:
             m = x
     return m
