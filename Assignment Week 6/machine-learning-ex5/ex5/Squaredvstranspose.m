?
theta = [1 ; 1];
X = [ones(m, 1) X];
B = ((X * theta) - y);
C =B'*B
D = sum(B.^2)
