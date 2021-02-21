import geometry.se3.py
htanh = Se3MatrixExpm()
x = torch.rand(2,4,4)

y = htanh(x)