# A primer on maths behind perceptrons:
## _Tushar Nandy_
#### Defining some terms:
1. Input matrix (N) with size 4X3
2. weight matrix (W) with size 3X1
3. $\rm Z = N \times W$
4. Activation function f(x) {tanh(x) in my case}
5. Actual output $\;\rm Y$. It is a matrix with size 4X1
6. Error function: $\rm \;\;E\,=\,Y-\,f(Z)$

#### The funda:
We want our weights to change over time so that our trained output starts getting reasonably closer to the actual outputs. This is the learning phase.
Converting this to maths, we want to 'minimise' the error function w.r.t the weights.
We will look at the change later on.

#### The essential derivative:
So, we are looking for $\Large{dE \over dW}$    
But we know that both E and W are column matrices or 'Vectors'.     

#### From derivative to gradient:

N = $\begin{bmatrix} A_1&A_1&A_3 \\ B_1&B_2&B_3 \\ C_1&C_2&C_3\\D_1&D_2&D_3\end{bmatrix}$   

W = $\begin{bmatrix}w_1\\w_2\\w_3\end{bmatrix}$

Z = $\begin{bmatrix}A_1w_1+A_2w_2+A_3w_3\\B...\\C...\\D...\end{bmatrix}=\begin{bmatrix}A\\B\\C\\D\end{bmatrix}$ (Suppose. I know this is an irrelevant notation XD)

f(Z) = $\begin{bmatrix}f(A_1w_1+A_2w_2+A_3w_3)\\f(B..)\\f(C..)\\f(D..)\end{bmatrix}$    

Y = $\begin{bmatrix}y_1\\y_2\\y_3\\y_4\end{bmatrix}$

E = $\begin{bmatrix}y_1-f(A)\\y_2-f(B)\\y_3-f(C)\\y_4-f(D)\end{bmatrix}$

NOW, $\;\;\Large{dE \over dW}=\frac{\partial E}{\partial w_1}{\normalsize\hat u_1}+\frac{\partial E}{\partial w_2}{\normalsize\hat u_2}+\frac{\partial E}{\partial w_3}{\normalsize\hat u_3}$

LET'S Check for the first term: $\Large{\frac{\partial E}{\partial w_1}}=\normalsize\begin{bmatrix}-f'(A)\times(A_1)\\-f'(B)\times(B_1)\\-f'(C)\times(C_1)\\-f'(D)\times(D_1)\end{bmatrix}$

SO, $\;\;\;\;$${\Large dE \over\Large dW} = \normalsize-\begin{bmatrix}f'(A)A_1&f'(A)A_2&f'(A)A_3\\f'(B)B_1&f'(B)B_2&f'(B)B_3\\f'(C)C_1&f'(C)C_2&f'(C)C_3\\f'(D)D_1&f'(D)D_2&f'(D)D_3\end{bmatrix}_{\large 4\times3} =\Large\frac{\partial E}{\partial f}\times\frac{\partial f}{\partial Z}\times\frac{\partial Z}{\partial w}$

#### From Gradient, we descent:
Our ultimate task is to change the weights to an optimum value. For that, we would require the change to be somewhat proportional to the error. Huge errors require big changes and small errors require miniscule or even no change sometimes.

$\qquad\qquad\qquad\rm So,\;the\;change\;\large[\Delta W]_{3\times1} =\Large(\frac{dE}{dW})^T\large\times E$

$\rm W_{new}=W+\Delta W$

Loop this a couple of times to achieve the desired weights
