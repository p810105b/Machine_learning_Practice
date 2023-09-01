import torch
import matplotlib.pyplot as plt

def plot(a, b, c):
    X = []
    Y_pred = []
    Y_targ = []
    for i in range(100):
        x = -10 + i/5 # from -5 to 5
        y_pred = a *  x**2 + b *  x + c
        y_targ = 2 *  x**2 + 3 *  x + 2
        
        X.append(x)
        Y_pred.append(y_pred)
        Y_targ.append(y_targ)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(X, Y_pred, linestyle='', marker = '*', color = 'red')
    plt.plot(X, Y_targ, color = 'blue')
    plt.legend(['Predict | a = %.2f | b = %.2f | c = %.2f' %(a, b, c),
                'Target  | a = 2.00 | b = 3.00 | c = 2.00'], title_fontsize = 15)

# Use GPU
device = 'cpu'
dtype = torch.float

# sample points(2000 pieces of data form -5 to 5)
x = torch.linspace(-5, 5, 2000, device = device, dtype = dtype)

# Target 
A, B, C = 2, 3, 2 
target = A * x**2 + B * x + C

# parameters setting
learning_rate = 3
epochs = 10

# generate random valus for predicted parameter
a = torch.rand(1, requires_grad = True, device = device, dtype = dtype)
b = torch.rand(1, requires_grad = True, device = device, dtype = dtype)
c = torch.rand(1, requires_grad = True, device = device, dtype = dtype)
print("y = %.2fx^2 + %.2fx + %.2f \n----------------------------"  %(a, b, c))
plt.figure(1)
plt.title('regression 0 time')
plot(a.item(), b.item(), c.item())
        
# By mean squared error = ((target - predict)**2) / n
MSELoss   = torch.nn.MSELoss(reduction ='sum')
optimizer = torch.optim.Adagrad([a, b, c], lr = learning_rate)

# training processing
for i in range(epochs):
    predict =  a * x**2 + b * x + c
    
    loss = MSELoss(predict, target)
    
    if i % 1 == 0:
        print(f'iteration : {i}, loss = {round(loss.item(), 3)}') # loss for 3 significant figures
    
    # get grad for loss function     
    loss.backward()
    
    # update parameters        
    optimizer.step()  
    
    # clear the grad after updating the parameters
    optimizer.zero_grad()
    
    a_value = a.item()
    b_value = b.item()
    c_value = c.item()
    
    # print and plot results
    if i == 0:
        print("y = %.2fx^2 + %.2fx + %.2f \n----------------------------"  %(a, b, c))
        plt.figure(2)
        plt.title('regression 1 time \n Loss = %.3f' %loss)
        plot(a_value, b_value, c_value)
    elif i == 2:
        print("y = %.2fx^2 + %.2fx + %.2f \n----------------------------"  %(a, b, c))
        plt.figure(3)
        plt.title('regression 3 times \n Loss = %.3f' %loss)
        plot(a_value, b_value, c_value)
    elif i == 9:
        print("y = %.2fx^2 + %.2fx + %.2f \n----------------------------"  %(a, b, c))
        plt.figure(4)
        plt.title('regression 10 times \n Loss = %.3f' %loss)
        plot(a_value, b_value, c_value)  