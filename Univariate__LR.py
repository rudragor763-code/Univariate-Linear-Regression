import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

##Data preprocessing

data=np.genfromtxt("Data/data.csv",delimiter=",")
# print(data)
# print(data.shape)
##Feature
X=data[:,0]
# print(X.shape)
# print(X[0:3])
##Label
y=data[:,1]
# print(y[0:3])
# print(y.shape)


min_max_scale=MinMaxScaler()

scaled_X=min_max_scale.fit_transform(np.reshape(X,(X.shape[0],1)))
scaled_y=min_max_scale.fit_transform(np.reshape(y,(y.shape[0],1)))
# print(scaled_X.shape)
# print(scaled_y.shape)
# print(scaled_X[0:3])
# print(scaled_y[0:3])

scaled_X=np.reshape(scaled_X,scaled_X.shape[0])
scaled_y=np.reshape(scaled_y,scaled_y.shape[0])
# print(scaled_y.shape)
# print(scaled_X.shape)

##Data preprocessing ends

##Hyper parameter:
learning_rate=0.1
max_itr=1000

##linear_regression logic

def predict_y(X,m,b):
    return m*X+b

def gradient(X,y,m,b):
    y_hat=predict_y(X,m,b)
    dm=np.average(((y_hat-y)*X))
    db=np.average(y_hat-y)
    return dm,db

def loss(X,y,m,b):
    y_hat=predict_y(X,m,b)
    return  np.average(np.square(y-y_hat))

def gradient_descent(X,y,learning_rate,max_itr):
    m=0
    b=0
    losses=[]
    for i in range(max_itr):
        dm,db=gradient(X,y,m,b)
        m=m-learning_rate*dm
        b=b-learning_rate*db
        loss_value=loss(X,y,m,b)
        losses.append(loss_value)
    return m,b,losses,loss_value

m,b,losses,loss_value=gradient_descent(scaled_X,scaled_y,learning_rate,max_itr)

# print("m:",m)
# print("b:",b)
# print("loss_value:",loss_value)

##Plotting part of code
plt.title("Linear regression for expected marks vs study of hours")
plt.xlabel("Study of hours")
plt.ylabel("Expected marks")
colors=np.random.rand(len(scaled_X),3)
sizes=np.random.randint(20,500,size=len(scaled_X))
plt.scatter(scaled_X,scaled_y,alpha=0.6,c=colors,s=sizes)
yy=predict_y(scaled_X,m,b)
plt.plot(scaled_X,yy)
plt.show()

##plotting losses
plt.title("Number of Losses")
plt.xlabel("Number of iteration")
plt.ylabel("Losses")
plt.plot(losses)
plt.show()