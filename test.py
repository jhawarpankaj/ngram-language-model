class A:
    def __init__(self):
        print("A")
class B(A):
    def __init__(self):
        super(B, self).__init__()
        print("B")

b = B()