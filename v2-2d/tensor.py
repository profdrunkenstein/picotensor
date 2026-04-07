class Tensor:
    def __init__(self, data):
        if not isinstance(data, list):
            data = [data] if data is not None else []

        self.data = data

        if len(data) == 0:
            self.ndim = 0
            self.shape = ()
        elif isinstance(data[0], list):
            self.ndim = 2
            row_length = len(data[0])
            for row in data:
                if len(row) != row_length:
                    raise ValueError("All rows must have the same length!")
            self.shape = (len(data), row_length)
        else:
            self.ndim = 1
            self.shape = (len(data),)

    def __repr__(self):
        if self.ndim == 0:
            return "Tensor([])"
        if self.ndim == 1:
            return f"Tensor({self.data})"

        indent = " " * 4
        lines = ["Tensor(["]
        for i, row in enumerate(self.data):
            comma = "," if i < len(self.data) - 1 else ""
            lines.append(f"{indent}{row}{comma}")
        lines.append("])")
        return "\n".join(lines)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.ndim == 1:
            return self.data[idx]

        if isinstance(idx, tuple):
            row, col = idx
            return self.data[row][col]

        result = self.data[idx]
        if isinstance(result, list):
            return Tensor(result)
        return result

    def __setitem__(self, idx, value):
        if isinstance(idx, tuple):
            row, col = idx
            if row >= self.shape[0] or col >= self.shape[1]:
                raise IndexError("Index out of bounds")
            self.data[row][col] = value
        else:
            if self.ndim == 2:
                if not isinstance(value, list):
                    raise TypeError("For 2D, you must assign a list to a row")
                if len(value) != self.shape[1]:
                    raise ValueError(f"Row must have length {self.shape[1]}")
            self.data[idx] = value

    # Element-wise operations
    def _elementwise_op(self, other, op):
        if self.shape != other.shape:
            raise ValueError(f"Tensors must have the same shape! Got {self.shape} and {other.shape}")
        if self.ndim == 1:
            return Tensor([op(x, y) for x, y in zip(self.data, other.data)])

        result = []
        for row_a, row_b in zip(self.data, other.data):
            row_result = [op(x, y) for x, y in zip(row_a, row_b)]
            result.append(row_result)
        return Tensor(result)

    def __add__(self, other):
        return self._elementwise_op(other, lambda x, y: x + y)

    def __sub__(self, other):
        return self._elementwise_op(other, lambda x, y: x - y)

    def __mul__(self, other):
        return self._elementwise_op(other, lambda x, y: x * y)

    def __truediv__(self, other):
        def safe_div(x, y):
            if y == 0:
                raise ZeroDivisionError("division by zero")
            return x / y
        return self._elementwise_op(other, safe_div)

    # Matrix multiplication
    def __matmul__(self, other):
        """Matrix multiplication (only 2D for now)"""
        if self.ndim != 2 or other.ndim != 2:
            raise ValueError("matmul only supports 2D tensors currently")

        rows_A, cols_A = self.shape
        rows_B, cols_B = other.shape

        if cols_A != rows_B:
            raise ValueError(f"Incompatible shapes for matrix multiplication: {self.shape} @ {other.shape}")

        result = []
        for i in range(rows_A):
            row_result = []
            for j in range(cols_B):
                total = 0
                for k in range(cols_A):
                    total += self.data[i][k] * other.data[k][j]
                row_result.append(total)
            result.append(row_result)

        return Tensor(result)

    # Reductions
    def sum(self):
        if self.ndim == 1:
            return sum(self.data)
        else:
            return sum(sum(row) for row in self.data)

    def mean(self):
        if len(self.data) == 0:
            raise ValueError("Empty Tensor")
        total_elements = len(self.data) * self.shape[1] if self.ndim == 2 else len(self.data)
        return self.sum() / total_elements

    def max(self):
        if len(self.data) == 0:
            raise ValueError("Empty Tensor")
        if self.ndim == 1:
            return max(self.data)
        return max(max(row) for row in self.data)

    def min(self):
        if len(self.data) == 0:
            raise ValueError("Empty Tensor")
        if self.ndim == 1:
            return min(self.data)
        return min(min(row) for row in self.data)

    # Utility methods 
    def flatten(self):
        """Flatten the tensor into 1D"""
        if self.ndim == 1:
            return Tensor(self.data.copy())
        flat = [x for row in self.data for x in row]
        return Tensor(flat)
