import csv
import numpy as np
import scipy.linalg
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
from scipy.stats import multivariate_normal
from scipy.stats import bernoulli

print("version",1)



class GaussianMixture:
		
	def __init__(self,mean0,cov0,mean1,cov1):
		""" construct a mixture of two gaussians. mean0 is 2x1 vector of means for class 0, cov0 is 2x2 covariance matrix for class 0. 
				Similarly for class 1"""
		self.mean0 = mean0
		self.mean1 = mean1
		self.cov0 = cov0
		self.cov1 = cov1
		self.rv0 = multivariate_normal(mean0, cov0)
		self.rv1 = multivariate_normal(mean1, cov1)
	
	def plot(self,data=None):
		x1 = np.linspace(-4,4,100)
		x2 = np.linspace(-4,4,100)
		X1,X2 = np.meshgrid(x1,x2)
		pos = np.empty(X1.shape+(2,))
		pos[:,:,0] = X1
		pos[:,:,1]= X2
		a = self.rv1.pdf(pos)/self.rv0.pdf(pos)
		
		if data:
			nplots = 4
		else:
			nplots = 3
		fig,ax = pl.subplots(1,nplots,figsize = (5*nplots,5))
		[ax[i].spines['left'].set_position('zero') for i in range(0,nplots)]
		[ax[i].spines['right'].set_color('none') for i in range(0,nplots)]
		[ax[i].spines['bottom'].set_position('zero') for i in range(0,nplots)]
		[ax[i].spines['top'].set_color('none') for i in range(0,nplots)]
	
		ax[0].set_title("p(x1,x2|y = 1)")
		ax[1].set_title("p(x1,x2|y = 0)")
		ax[2].set_title("p(y = 1|x1,x2)")
		[ax[i].set_xlim([-4,4]) for i in range(0,3)]
		[ax[i].set_ylim([-4,4]) for i in range(0,3)]

		cn = ax[0].contourf(x1,x2,self.rv1.pdf(pos))
		cn2 = ax[1].contourf(x1,x2,self.rv0.pdf(pos))
		z = a/(1.0+a)
		cn3 = ax[2].contourf(x1,x2,z)
		ct = ax[2].contour(cn3,levels=[0.5])
		ax[2].clabel(ct)


		if data:
			X,Y = data
			colors = np.array(["blue" if target < 1 else "red" for target in Y])
			x = X[:,0]
			y = X[:,1]
			yis1 = np.where(Y==1)[0]
			yis0 = np.where(Y!=1)[0]
			ax[3].set_title("Samples colored by class")
			ax[3].scatter(x,y,s=30,c=colors,alpha=.5)
			ax[0].scatter(x[yis1],y[yis1],s=5,c=colors[yis1],alpha=.3)	
			ax[1].scatter(x[yis0],y[yis0],s=5,c=colors[yis0],alpha=.3)	
			ax[2].scatter(x,y,s=5,c=colors,alpha=.3)	
		pl.show()
	 
	def sample(self,n_samples,py,plot=False):
		"""samples Y according to py and corresponding features x1,x2 according to the gaussian for the corresponding class"""
		Y = bernoulli.rvs(py,size=n_samples)
		X = np.zeros((n_samples,2))
		for i in range(n_samples):
			if Y[i] == 1:
				X[i,:] = self.rv1.rvs()
			else:
				X[i,:] = self.rv0.rvs()
		if plot:
			self.plot(data=(X,Y))
		return X,Y


		
def load_data_(filename):
    with open(filename) as f:
        g = (",".join([i[1],i[2],i[4],i[5],i[6],i[7],i[9],i[11]]).encode(encoding='UTF-8') 
                for i in csv.reader(f,delimiter=",",quotechar='"'))
        data = np.genfromtxt(g, delimiter=",",names=True,
                dtype=(int,int,np.dtype('a6'),float,int,int,float,np.dtype('a1')))
    embark_dict = {b'S':0, b'C':1, b'Q':2, b'':3}
    survived = data['Survived']
    passenger_class = data['Pclass']
    is_female = (data['Sex'] == b'female').astype(int)
    age = data['Age']
    sibsp = data['SibSp']
    parch = data['Parch']
    fare = data['Fare']
    embarked = np.array([embark_dict[k] for k in data['Embarked']])
    # skip age for the moment because of the missing data
    X = np.vstack((passenger_class, is_female, sibsp, parch, fare, embarked)).T
    Y = survived
 
    return X, Y

def load_data():
	return load_data_("titanic_train.csv")
	

def load_test_data():
	return load_data_("titanic_test.csv")



def whitening_matrix(X):
    """The matrix of Eigenvectors that whitens the input vector X"""
    assert (X.ndim == 2)
    sigma = np.dot(X.T, X)
    e, m = scipy.linalg.eigh(sigma)
    return np.dot(m, np.diag(1.0/np.sqrt(e)))*np.sqrt((X.shape[0]-1))


def plot_svm(X, Y, svm_instance, xdim1=0, xdim2=1, minbound=(-3,-3),
        maxbound=(3,3), resolution=(100,100)):
    """ Plot any two dimensions from an SVM"""
    # build the meshgrid for the two dims we care about
    d = svm_instance.shape_fit_[1]
    n = resolution[0] * resolution[1]
    xx, yy = np.meshgrid(np.linspace(minbound[0], maxbound[0], resolution[0]),
            np.linspace(minbound[1], maxbound[1], resolution[1]))
    query2d = np.c_[xx.ravel(), yy.ravel()]
    query = np.zeros((n,d))
    query[:,xdim1] = query2d[:, 0]
    query[:,xdim2] = query2d[:, 1]

    Z = svm_instance.decision_function(query)
    Z = Z.reshape(xx.shape)

    fig = pl.figure(figsize=(10,10))
    ax = fig.add_subplot(111)

    ax.imshow(Z, interpolation='nearest',
            extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
            origin='lower', cmap=pl.cm.PuOr_r)
    contours = ax.contour(xx, yy, Z, levels=[0], linewidths=2,
            linetypes='--')
    ax.scatter(X[:, xdim1], X[:, xdim2], s=30, c=Y, cmap=pl.cm.Paired)
    # ax.set_xticks(())
    # pl.yticks(())
    ax.set_xlim((minbound[0], maxbound[0]))
    ax.set_ylim((minbound[1], maxbound[1]))
    pl.show()
 

def illustrate_preprocessing():
    x = np.random.multivariate_normal(np.array([5.0,5.0]),
            np.array([[5.0,3.0],[3.0,4.0]]),size=1000)
    x_demean = x - np.mean(x, axis=0)
    x_unitsd = x_demean/(np.std(x_demean,axis=0))
    x_whiten = np.dot(x_demean, whitening_matrix(x_demean))

    fig = pl.figure(figsize=(10,10))
    
    def mk_subplot(n, data, label):
        ax = fig.add_subplot(2,2,n)
        ax.scatter(data[:,0], data[:,1])
        ax.set_xlim((-10,10))
        ax.set_ylim((-10,10))
        ax.set_xlabel(label)

    mk_subplot(1, x, "Original")
    mk_subplot(2, x_demean, "De-meaned")
    mk_subplot(3, x_unitsd, "Unit SD")
    mk_subplot(4, x_whiten, "Whitened")
    pl.show()


def margins_and_hyperplane():
    #gen some data
    np.random.seed(0)
    n = 20
    X = (np.vstack((np.ones((n,2))*np.array([0.5,1]), 
        np.ones((n,2))*np.array([-0.5,-1]))) + np.random.randn(2*n,2)*0.3)
    Y = np.hstack((np.ones(n), np.zeros(n)))

    clf = svm.SVC(kernel='linear')
    clf.fit(X, Y)

    # Note the following code comes from a scikit learn example...
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xs = np.linspace(-2, 2)
    ys = a * xs - (clf.intercept_[0]) / w[1]
    
    # plot the parallels to the separating hyperplane that pass through the
    # support vectors
    b = clf.support_vectors_[0]
    ys_down = a * xs + (b[1] - a * b[0])
    b = clf.support_vectors_[-1]
    ys_up = a * xs + (b[1] - a * b[0])

    #draw a bad margin

    def line_point_grad(x, grad, p1):
        y = grad*(x - p1[0]) + p1[1]
        return y

    minp = X[np.argmin(X[:n,0])]
    maxp = X[n + np.argmax(X[n:,0])]
    yb = line_point_grad(xs, a*20, np.array([0.5*(minp[0]+maxp[0]),0.0]))
    yb_down = line_point_grad(xs, a*20, minp)
    yb_up = line_point_grad(xs, a*20, maxp)

    # plot the line, the points, and the nearest vectors to the plane
    fig = pl.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.plot(xs, ys, 'g-')
    ax.plot(xs, yb, 'r-')
    ax.plot(xs, yb_down, 'r--')
    ax.plot(xs, yb_up, 'r--')
    ax.plot(xs, ys_down, 'g--')
    ax.plot(xs, ys_up, 'g--')

    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                s=80, facecolors='none')
    ax.scatter([minp[0],maxp[0]], [minp[1],maxp[1]],
            s=80, facecolors='none')
    ax.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)
    ax.set_xlim((-2,2))
    ax.set_ylim((-2,2))
    pl.show()


def hard_data():
    #gen some data
    np.random.seed(0)
    epsilon = 0.05
    n = 5000
    X1 = np.random.randn(n,2)
    X2 = np.random.randn(n,2)
    valid1 = X1[:,0]**2 + X1[:,1]**2 < (0.5 - epsilon)
    valid2 = np.logical_and((X2[:,0]**2 + X2[:,1]**2 > (0.5 + epsilon)),
            (X2[:,0]**2 + X2[:,1]**2 < 1.0))

    X1 = X1[valid1]
    X2 = X2[valid2]
    Y1 = np.ones(X1.shape[0])
    Y2 = np.zeros(X2.shape[0])
    X = np.vstack((X1,X2))
    Y = np.hstack((Y1,Y2))
    Z = np.sqrt(2)*X[:,0]*X[:,1]
    return X, Y, Z

def nonlinear_example():
    X, Y, Z = hard_data()
    fig = pl.figure(figsize=(10,20))
    ax = fig.add_subplot(211)
    ax.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)
    ax = fig.add_subplot(212, projection='3d')
    ax.scatter(X[:,0]**2, X[:,1]**2, Z, c=Y, cmap=pl.cm.Paired)
    pl.show()

def nonlinear_svm():
    X, Y, Z = hard_data()
    clf = svm.SVC(kernel='rbf')
    clf.fit(X, Y)
    plot_svm(X, Y, clf, 0,1, (-1.5,-1.5), (1.5,1.5)) 
    

#if __name__ == "__main__":
#    nonlinear_example()
