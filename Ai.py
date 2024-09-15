import numpy as np

with open("numeral",'r') as file:
    x=file.readline()
    y=file.readline()
    file.close()
print(x)
print("---------")
print(y)
def integrate(x, y):
    if len(x) != len(y):
        raise ValueError("Списки x и y должны быть одинаковой длины")

    n = len(x)
    integral = 0
    for i in range(n - 1):
        integral += (x[i + 1] - x[i]) * (y[i] + y[i + 1]) / 2

    return integral

def errorf(x,y):
    return x-y if (x-y)<0 else y-x


def PID(k1,k2,k3,x,y,t,dt):  #k - koeff, y - proccess var, x - setpoint

    I=0
    P=k1*errorf(x[t],y[t])
    i=0

    while i <= t:
        i+=dt
        Integr=errorf(integrate(x[:t],y[:t]),y[t])
        I=k2*Integr

    D=k3*(errorf(x[t],y[t])-errorf(x[t-1],y[t-1]))/dt
    z=P+D+I
    if z>1:
        z=1
    elif z<-1:
        z = -1
    x[t]-=z
    return x[t]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

np.random.seed(0)

inputs = 3
hidden_layer = 4
outputs = 1

weights_input_to_hidden = np.random.normal(size=(inputs, hidden_layer))
weights_hidden_to_output = np.random.normal(size=(hidden_layer, outputs))

bias_hidden = np.random.normal(size=(1, hidden_layer))
bias_output = np.random.normal(size=(1, outputs))



def forward_propagation(X):
    hidden_layer_input = np.dot(X, weights_input_to_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_to_output) + bias_output
    output = sigmoid(output_layer_input)

    return output, hidden_layer_output


def back_propagation(X, y, output, hidden_layer_output):
    global weights_input_to_hidden, weights_hidden_to_output, bias_hidden, bias_output

    output_error = y - output
    output_delta = output_error * sigmoid_derivative(output)

    hidden_error = output_delta.dot(weights_hidden_to_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_output)

    weights_hidden_to_output += hidden_layer_output.T.dot(output_delta)
    bias_output += np.sum(output_delta)

    weights_input_to_hidden += X.T.dot(hidden_delta)
    bias_hidden += np.sum(hidden_delta)



X_train = x
y_train = y

epochs = 10000
learning_rate = 0.1

for epoch in range(epochs):
    for i in range(len(X_train)):
        output, hidden_layer_output = forward_propagation(X_train[i])
        back_propagation(X_train[i], y_train[i], output, hidden_layer_output)

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Error: {np.mean(np.abs(y_train - np.array([forward_propagation(x)[0] for x in X_train])))}")


