import torch.nn as nn
import torch
import math
import time
import numpy as np
import itertools as it
from matplotlib import pyplot as plt
from matplotlib import cm
from myqr import myqr

# create new model that takes phi as input (multiple input dimensions)
class GP_new(nn.Module):
    def __init__(self, sigma_f, lengthscale, sigma_n):
        super(GP_new, self).__init__()
        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.log_sigma_f = nn.Parameter(torch.Tensor([sigma_f]).abs().log())
        self.log_lengthscale = nn.Parameter(torch.as_tensor(lengthscale, dtype=torch.float).abs().log())
        self.log_sigma_n = nn.Parameter(torch.Tensor([sigma_n]).abs().log())

    # the predict forward function
    def forward(self, y_train, phi, sq_lambda, L, m_test):

        # extract hyperparameters
        sigma_f = torch.exp(self.log_sigma_f)
        lengthscale = torch.exp(self.log_lengthscale)
        sigma_n = torch.exp(self.log_sigma_n)

        # number of basis functions
        m = sq_lambda.size(0)

        # input dimension
        diml = sq_lambda.size(1)

        # See the autograd section for explanation of what happens here.
        n = y_train.size(0)

        lprod=torch.ones(1)
        omega_sum=torch.zeros(m,1)
        for q in range(diml):
            lprod = lprod.mul( lengthscale[q].pow(2) )
            omega_sum = omega_sum.add( lengthscale[q].pow(2)*sq_lambda[:,q].view(m,1).pow(2) )

        inv_lambda_diag = ( sigma_f.pow(-2) .mul( lprod.pow(-0.5) ) .mul(
                                  torch.exp( 0.5*omega_sum ) ) ).mul(math.pow(2.0*math.pi,-diml/2)).view(m)

        Z = phi.t().mm(phi) + torch.diag( inv_lambda_diag ).mul(sigma_n.pow(2))
        phi_lam = torch.cat( (phi , inv_lambda_diag.sqrt().diag().mul(sigma_n)),0)  # [Phi; sign*sqrt(Lambda^-1)]

        _,r = torch.qr(phi_lam)
        v, _ = torch.gesv( phi.t().mm(y_train.view(n, 1)), Z)  # X,LU = torch.gesv(B, A); AX=B => v=(Phi'*Phi+sign^2I)\(Phi'*y)

        # compute phi_star
        nt = m_test.size(0)
        phi_star=torch.ones(m,nt)
        for q in range(diml):
            phi_star = phi_star.mul( torch.sin(sq_lambda[:,q].view(m,1)*(m_test[:,q].view(1,nt)+L[q])).div(math.sqrt(L[q])) )

        # predict
        f_test = phi_star.t().mm(v)
        tmp,_ = torch.trtrs(phi_star,r.t(),upper=False)  # solves r^T*u=phi_star, u=r*x
        tmp,_ = torch.trtrs(tmp,r) # solves r*x=u
        cov_f = torch.sum(phi_star.t().mul(tmp.t()), dim=1).mul(sigma_n.pow(2))
        out = (f_test, cov_f)
        return out

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'sigma_f={}, lengthscale={}, sigma_n={}'.format(
            torch.exp(self.log_sigma_f).item(), torch.exp(self.log_lengthscale), torch.exp(self.log_sigma_n).item()
        )


# Deep version with numerical integration and no backward implementation, just relying on autoback features
class NegMarginalLogLikelihood_phi_noBackward(nn.Module):
    def __init__(self):
        super(NegMarginalLogLikelihood_phi_noBackward, self).__init__()

    def forward(self, log_sigma_f, log_lengthscale, log_sigma_n, phi, y_train, sq_lambda):
        # extract hyperparameters
        sigma_f = torch.exp(log_sigma_f)
        lengthscale = torch.exp(log_lengthscale)
        sigma_n = torch.exp(log_sigma_n)

        # extract the data set size
        n = y_train.size(0)

        # number of basis functions
        m = sq_lambda.size(0)

        # input dimension
        dim = sq_lambda.size(1)

        # diagonal of inverse lambda matrix
        lprod=torch.ones(1)
        omega_sum=torch.zeros(m,1)
        for q in range(dim):
            lprod = lprod.mul( lengthscale[q].pow(2) )
            omega_sum = omega_sum.add( lengthscale[q].pow(2)*sq_lambda[:,q].view(m,1).pow(2) )

        inv_lambda_diag = ( torch.pow(lprod, -0.5).mul(torch.exp( 0.5*omega_sum )).mul(math.pow(2.0*math.pi,-dim/2)) ).view(m).mul(sigma_f.pow(-2))

        Z = phi.t().mm(phi) + torch.diag(inv_lambda_diag.mul(sigma_n.pow(2)))  # Z

        phi_lam = torch.cat( (phi , torch.diag( torch.sqrt(inv_lambda_diag) ).mul(sigma_n) ), 0)  # [Phi; sign*sqrt(Lambda^-1)]
        r = myqr(phi_lam)

        if torch.isinf(r.diag().pow(-1)).any().item()==1:
            return torch.from_numpy(np.array([np.inf])).float()  # tell the optimizer it's an illegal point

        tmp,_ = torch.trtrs( phi.t().mm(y_train.view(n, 1)) , r.t(), upper=False)  # solves r^T*u=phi*y, u=r*x
        v,_ = torch.trtrs(tmp, r) # solves r*x=u

        n = y_train.size(0)
        logQ = ( (n-m)*torch.log(sigma_n.pow(2))
                + Z.logdet() # + r.diag().abs().log().sum().mul(2) gives crazy low cost values...
                + torch.sum(torch.log(inv_lambda_diag.pow(-1))) )
        yQiy = ( y_train.dot(y_train) - v.view(m).dot(y_train.view(1, n).mm(phi).view(m) ) ).mul(sigma_n.pow(-2))

        nLL = 0.5*logQ + 0.5*yQiy  # neg log marg likelihood
        return nLL


class NegLOOCrossValidation_phi_noBackward(nn.Module):
    def __init__(self):
        super(NegLOOCrossValidation_phi_noBackward, self).__init__()

    def forward(self, log_sigma_f, log_lengthscale, log_sigma_n, phi, y_train, sq_lambda):
        # extract hyperparameters
        sigma_f = torch.exp(log_sigma_f)
        lengthscale = torch.exp(log_lengthscale)
        sigma_n = torch.exp(log_sigma_n)

        # extract the data set size
        n = y_train.size(0)

        # number of basis functions
        m = sq_lambda.size(0)

        # input dimension
        dim = sq_lambda.size(1)

        # diagonal of inverse lambda matrix
        lprod = torch.ones(1)
        omega_sum = torch.zeros(m,1)
        for q in range(dim):
            lprod = lprod.mul( lengthscale[q].pow(2) )
            omega_sum = omega_sum.add( lengthscale[q].pow(2)*sq_lambda[:,q].view(m,1).pow(2) )

        lambda_diag = torch.pow(lprod, 0.5).mul(torch.exp( omega_sum.mul(0.5).neg() )).mul(math.pow(2.0*math.pi,dim/2.0)).view(m).mul(sigma_f.pow(2))

        phi_lam = torch.cat( ( lambda_diag.sqrt().diag().mm(phi.t()) , torch.eye(n).mul(sigma_n) ), 0)  # [Phi; sign*sqrt(Lambda^-1)]
        r = myqr(phi_lam)

        if torch.isinf(r.diag().pow(-1)).any().item()==1:
            return torch.from_numpy(np.array([np.inf])).float()  # tell the optimizer it's an illegal point

        tmp = torch.trtrs( y_train.view(n, 1) , r.t(), upper=False)[0]  # solves r^T*u=y, u=r*x
        Kinv_y = torch.trtrs(tmp, r)[0] # solves r*x=u

        tmp = torch.trtrs(torch.eye(n),r.t(),upper=False)[0]  # solves r^T*u = I, u = r*x
        sigmas_sq = torch.trtrs(tmp,r)[0].diag().pow(-1) # solves r*x = u and extracts the inverse diagonal elements

        return sigmas_sq.log() .add( y_train.sub( y_train.sub( Kinv_y.mul(sigmas_sq) ) ).pow(2).div(sigmas_sq) ) .sum()


###########################################################################################
####################################### some other stuff (maybe put in another file)
###########################################################################################
def choleskZ(Z):
    # do numerical tweaks to ensure pos. def., qr hasn't got derivatives implemented...
    Z = (Z+Z.t()).mul(0.5) # enforce symmetry
    add = 1e-9
    iteration=1.0
    while True: # loop until cholesky works
        try:
            c = torch.cholesky(Z, upper=True)
            break
        except: # not positive definit
            Z = Z + torch.eye(Z.size(0)).mul(add*np.exp(np.log(100.0)*iteration))
            print('Not positive definite!!: %d' %(iteration))
            iteration+=1
            continue
    return Z,c

class buildPhi():
    def __init__(self,m,type='point',ni=400,int_method=3,tun=4,x0=None,unitvecs=None,Rlim=None):
        super(buildPhi, self).__init__()
        self.type=type
        self.tun=tun
        self.index = getIndex(m)
        if self.type=='int':
            if int_method is 1:
                # trapezoidal
                sc  = torch.ones(1,ni+1).mul(2.0)
                sc[0,0] = 1
                sc[0,ni]= 1
                fact = 1.0/2.0
            elif int_method is 2:
                # simpsons standard
                ni = 2*round(ni/2)
                sc=torch.ones(1,ni+1)
                sc[0,ni-1]=4
                sc[0,1:ni-1] = torch.Tensor([4,2]).repeat(1,int(ni/2-1))
                fact = 1.0/3.0
            else:
                # simpsons 3/8
                ni = 3*round(ni/3)
                sc=torch.ones(1,ni+1)
                sc[0,ni-1]=3; sc[0,ni-2]=3
                sc[0,1:ni-2] = torch.Tensor([3,3,2]).repeat(1,int(ni/3-1))
                fact = 3.0/8.0
            self.ni=ni
            self.sc=sc
            self.fact=fact
            self.x0=x0
            self.unitvecs=unitvecs
            self.Rlim=Rlim

    def getphi(self,model,m,n,mt,dom_points=None,train_x=None):
        phi = torch.ones(n,mt)
        diml = len(m)
        L = torch.empty(diml)

        if dom_points is None: # point meas
            dom_points=train_x
        mtest=model(dom_points).abs()
        for q in range(diml):
            L[q] = max(1.2*mtest[:,q].max(), math.pi*m[q]*model.gp.log_lengthscale[q].exp().detach().abs()/(2.0*self.tun) )
            sq_lambda = math.pi*self.index / (2.0*L)

        if self.type=='int':
            # st = time.time()
            if self.x0 is None: # 1D
                for q in range(n):
                    a = train_x[q,0]
                    b = train_x[q,1]
                    h = (b-a)/self.ni

                    zz = model( torch.linspace(a,b,self.ni+1).view(self.ni+1,1) )

                    intvals = torch.ones(self.ni+1,mt)
                    for w in range(diml):
                        intvals *= torch.sin((zz[:,w].view(self.ni+1,1)+L[w]).mul(sq_lambda[:,w].view(1,mt))).mul(math.pow(L[w],-0.5))

                    phi[q,:] = torch.sum(intvals*self.sc.t() , dim=0).mul(self.fact*h)
            else:
                if not model.pureGP:
                    for q in range(n):
                        h = 2*self.Rlim/self.ni

                        svec = torch.linspace(-self.Rlim,self.Rlim,self.ni+1).view(self.ni+1,1)

                        zz = model( self.x0[q,:].repeat(self.ni+1,1) + svec*self.unitvecs[q,:] )

                        intvals = torch.ones(self.ni+1,mt)
                        for w in range(diml):
                            intvals *= torch.sin((zz[:,w].view(self.ni+1,1)+L[w]).mul(sq_lambda[:,w].view(1,mt))).mul(math.pow(L[w],-0.5))

                        phi[q,:] = torch.sum(intvals*self.sc.t() , dim=0).mul(self.fact*h)

                else:
                    x0_start = self.x0 - self.Rlim*self.unitvecs
                    lambdaXin = sq_lambda[:,0]
                    lambdaYin = sq_lambda[:,1]
                    for q in range (n):
                        x0_s = x0_start[q,0]
                        y0_s = x0_start[q,1]

                        lambdaX = self.unitvecs[q,0]*lambdaXin
                        lambdaY = self.unitvecs[q,1]*lambdaYin

                        BX = (x0_s+L[0]) * lambdaXin
                        BY = (y0_s+L[1]) * lambdaYin

                        lambda_min = lambdaX - lambdaY
                        b_min = BX - BY
                        lambda_plus = lambdaX + lambdaY
                        b_plus = BX + BY

                        lambda_min_div = lambda_min
                        lambda_plus_div = lambda_plus

                        lambda_min_div[lambda_min.abs()<1e-14] = 1
                        lambda_plus_div[lambda_plus.abs()<1e-14] = 1

                        theInt = 0.5*( ( torch.sin(lambda_min*2*self.Rlim+b_min)-torch.sin(b_min) ) / lambda_min_div +
                                        (  torch.sin(b_plus) - torch.sin(lambda_plus*2*self.Rlim+b_plus)   ) / lambda_plus_div )

                        phi[q,:] = theInt.view(1,mt).mul(math.pow(L[0]*L[1],-0.5))

            # print(time.time()-st)
            return (phi,sq_lambda,L)

        if self.type=='point':
            zz = model(train_x)
            for q in range(diml):
                phi = phi.mul( torch.sin((zz[:,q].view(n,1)+L[q]).mul(sq_lambda[:,q].view(1,mt))).mul(math.pow(L[q],-0.5)) )
            return (phi, sq_lambda, L)

### print optimisation details
def optiprint(i,training_iterations,lossitem,lr,ls_step,model,L):
    diml=L.size(0)
    if diml is 1:
        print('Iter %d/%d - Loss: %.8f - LR: %.5f - LS iters: %0.0f - sigf: %.3f - l: %.3f - sign: %.5f - L: %.3f' % (i + 1, training_iterations, lossitem, lr, ls_step, model.gp.log_sigma_f.exp(), model.gp.log_lengthscale[0].exp(), model.gp.log_sigma_n.exp(), L[0] ))
    elif diml is 2:
        print('Iter %d/%d - Loss: %.8f - LR: %.5f - LS iters: %0.0f - sigf: %.3f - l1: %.3f - l2: %.3f - sign: %.5f - L1: %.3f - L2: %.3f' % (i + 1, training_iterations, lossitem, lr, ls_step, model.gp.log_sigma_f.exp(), model.gp.log_lengthscale[0].exp(), model.gp.log_lengthscale[1].exp(), model.gp.log_sigma_n.exp(), L[0], L[1] ))
    elif diml is 3:
        print('Iter %d/%d - Loss: %.8f - LR: %.5f - LS iters: %0.0f - sigf: %.3f - l1: %.3f - l2: %.3f - l3: %.3f - sign: %.5f - L1: %.3f - L2: %.3f - L3: %.3f' % (i + 1, training_iterations, lossitem, lr, ls_step, model.gp.log_sigma_f.exp(), model.gp.log_lengthscale[0].exp(), model.gp.log_lengthscale[1].exp(), model.gp.log_lengthscale[2].exp(), model.gp.log_sigma_n.exp(), L[0], L[1], L[2] ))


####### plot
### 1D
def makeplot(model,train_x,train_y,test_x,test_f,cov_f,truefunc,diml,meastype='point'):
    with torch.no_grad():
        fplot, ax = plt.subplots(1, 1, figsize=(4, 3))


        if meastype=='point':
            # Plot training data as red stars
            ax.plot(train_x.numpy(), train_y.numpy(), 'r*')

        # Plot true function as solid black
        ax.plot(test_x.numpy(), truefunc(test_x).numpy(), 'k')

        # plot latent outputs
        train_m = model(test_x)
        for w in range(diml):
            ax.plot(test_x.numpy(), train_m[:,w].numpy(), 'g')

        # plot 95% credibility region
        upper = torch.squeeze(test_f, 1) + 2*cov_f.pow(0.5)
        lower = torch.squeeze(test_f, 1) - 2*cov_f.pow(0.5)
        ax.fill_between(torch.squeeze(test_x,1).numpy(), lower.numpy(), upper.numpy(), alpha=0.5)

        # plot predictions
        ax.plot(test_x.numpy(), test_f.detach().numpy(), 'b')

        #ax.set_ylim([-2, 2])
        # ax.legend(['Observed Data', 'True', 'Predicted'])
        plt.show()

## 2D
def makeplot2D(model,X,Y,ntx,nty,test_f,cov_f,diml,test_x=None,truefunc=None,Z=None,train_x=None,type='point',vmin=-1,vmax=1):
    with torch.no_grad():
        if type=='point':
            fplot, ax = plt.subplots(2, 3, figsize=(27,9))

            ## true function & meas
            # Z = np.reshape(truefunc(test_x).numpy(),(ntx,nty))
            pc = ax[0,0].pcolor(X,Y,Z, cmap=cm.coolwarm)
            pc.set_clim(vmin,vmax)
            ax[0,0].plot(train_x[:,0].numpy(),train_x[:,1].numpy(),'go', alpha=0.3)

            ## prediction
            Zp = np.reshape(test_f.detach().numpy(),(ntx,nty))
            pc = ax[0,1].pcolor(X,Y,Zp, cmap=cm.coolwarm)
            pc.set_clim(vmin,vmax)

            ## covariance
            Zc = np.reshape(cov_f.detach().numpy(),(ntx,nty))
            pc = ax[0,2].pcolor(X,Y,Zc, cmap=cm.coolwarm)
            pc.set_clim(vmin,vmax)

            # plot latent outputs
            train_m = model(test_x)
            for w in range(diml):
                Zm = np.reshape(train_m[:,w].numpy(),(ntx,nty))
                pc = ax[1,w].pcolor(X,Y,Zm, cmap=cm.coolwarm)
                pc.set_clim(vmin,vmax)

            ## shared colorbar
            fplot.colorbar(pc, ax=ax.ravel().tolist())

            plt.show()
        else:
            fplot, ax = plt.subplots(2, 3, figsize=(27,9))

            ## true function & meas
            pc = ax[0,0].pcolor(X,Y,Z, cmap=cm.coolwarm)
            pc.set_clim(vmin,vmax)

            ## prediction
            Zp = np.reshape(test_f.detach().numpy(),(ntx,nty))
            pc = ax[0,1].pcolor(X,Y,Zp, cmap=cm.coolwarm)
            pc.set_clim(vmin,vmax)

            ## covariance
            Zc = np.reshape(cov_f.detach().numpy(),(ntx,nty))
            pc = ax[0,2].pcolor(X,Y,Zc, cmap=cm.coolwarm)
            pc.set_clim(vmin,vmax)

            # plot latent outputs
            train_m = model(test_x)
            for w in range(diml):
                Zm = np.reshape(train_m[:,w].numpy(),(ntx,nty))
                pc = ax[1,w].pcolor(X,Y,Zm, cmap=cm.coolwarm)
                pc.set_clim(vmin,vmax)

            ## shared colorbar
            fplot.colorbar(pc, ax=ax.ravel().tolist())

            plt.show()


def makeplot2D_new(filepath,vmin=-2,vmax=2,cmap=cm.coolwarm):
    try:
        (model, dataname, train_y, n, x0, unitvecs, Rlim, X, Y, Z, rec_fbp, err_fbp,
                ntx, nty, test_x, dom_points, m, diml, mt,
                test_f, cov_f, noise_std, nLL, buildPhi, opti_state, it_number) = \
            torch.load(filepath)

        with torch.no_grad():
            fplot, ax = plt.subplots(2, 3, figsize=(27,9))

            ## true function
            ax[0,0].set_title('Original')
            pc = ax[0,0].imshow(Z, extent=(X.min(), X.max(), Y.min(), Y.max()), cmap=cmap)
            pc.set_clim(vmin,vmax)

            ## fbp
            ax[0,1].set_title('FBP')
            pc = ax[0,1].imshow(rec_fbp, extent=(X.min(), X.max(), Y.min(), Y.max()), cmap=cmap)
            pc.set_clim(vmin,vmax)

            ## prediction
            ax[0,2].set_title('GP prediction')
            Zp = np.reshape(test_f.detach().numpy(),(ntx,nty))
            pc = ax[0,2].imshow(Zp, extent=(X.min(), X.max(), Y.min(), Y.max()), cmap=cmap)
            pc.set_clim(vmin,vmax)

            ## GP standard div
            ax[1,0].set_title('GP std')
            Zstd = np.reshape(cov_f.detach().sqrt().numpy(),(ntx,nty))
            pc = ax[1,0].imshow(Zstd, extent=(X.min(), X.max(), Y.min(), Y.max()), cmap=cmap)
            pc.set_clim(vmin,vmax)

            # plot latent outputs
            train_m = model(test_x)
            for w in range(diml):
                ax[1,w+1].set_title('latent output'+str(w+1))
                Zm = np.reshape(train_m[:,w].numpy(),(ntx,nty))
                pc = ax[1,w+1].imshow(Zm, extent=(X.min(), X.max(), Y.min(), Y.max()), cmap=cmap)
                pc.set_clim(vmin,vmax)

            if w==0:
                ax[1,2].set_axis_off()

            ## shared colorbar
            fplot.colorbar(pc, ax=ax.ravel().tolist())

            plt.show()

        # RMS error
        ground_truth = torch.from_numpy(Z).float().view(np.size(Z))
        error = torch.mean( (ground_truth - test_f.squeeze()).pow(2) ).sqrt()
        print('RMS error: %.10f' %(error.item()))
        print('RMS error fbp: %.10f' %(err_fbp))
    except:
        (model, dataname, train_y, n, train_x, X, Y, Z,
                        ntx, nty, test_x, dom_points, m, diml, mt,
                        test_f, cov_f, noise_std, nLL, buildPhi, opti_state, it_number) = \
                torch.load(filepath)

        with torch.no_grad():
            fplot, ax = plt.subplots(2, 3, figsize=(27,9))

            ## true function
            ax[0,0].set_title('Original')
            pc = ax[0,0].imshow(Z, extent=(X.min(), X.max(), Y.min(), Y.max()), cmap=cmap)
            pc.set_clim(vmin,vmax)

            ## prediction
            ax[0,1].set_title('GP prediction')
            Zp = np.reshape(test_f.detach().numpy(),(ntx,nty))
            pc = ax[0,1].imshow(Zp, extent=(X.min(), X.max(), Y.min(), Y.max()), cmap=cmap)
            pc.set_clim(vmin,vmax)

            ## GP standard div
            ax[0,2].set_title('GP std')
            Zstd = np.reshape(cov_f.detach().sqrt().numpy(),(ntx,nty))
            pc = ax[0,2].imshow(Zstd, extent=(X.min(), X.max(), Y.min(), Y.max()), cmap=cmap)
            pc.set_clim(vmin,vmax)

            # plot latent outputs
            train_m = model(test_x)
            for w in range(diml):
                ax[1,w].set_title('latent output'+str(w+1))
                Zm = np.reshape(train_m[:,w].numpy(),(ntx,nty))
                pc = ax[1,w].imshow(Zm, extent=(X.min(), X.max(), Y.min(), Y.max()), cmap=cmap)
                pc.set_clim(vmin,vmax)

            if w==0:
                ax[1,1].set_axis_off()
                ax[1,2].set_axis_off()

            ## shared colorbar
            fplot.colorbar(pc, ax=ax.ravel().tolist())

            plt.show()

        # RMS error
        ground_truth = torch.from_numpy(Z).float().view(np.size(Z))
        error = torch.mean( (ground_truth - test_f.squeeze()).pow(2) ).sqrt()
        print('RMS error: %.10f' %(error.item()))


### function that returns index vector
def getIndex(m):
    mt= np.prod(m) # total nr of basis functions
    diml = len(m) # dimension of latent output

    # create an index vector to store basis function permutations
    index=torch.empty(mt,diml)

    mmlist=[]
    for q in range(diml):
        mmlist.append(np.linspace(1, m[q], m[q]))

    # hard coded, but more than sufficient...
    if diml is 1:
        perm = list(it.product(mmlist[0]))
    elif diml is 2:
        perm = list(it.product(mmlist[0],mmlist[1]))
    elif diml is 3:
        perm = list(it.product(mmlist[0],mmlist[1],mmlist[2]))
    elif diml is 4:
        perm = list(it.product(mmlist[0],mmlist[1],mmlist[2],mmlist[3]))
    elif diml is 5:
        perm = list(it.product(mmlist[0],mmlist[1],mmlist[2],mmlist[3],mmlist[4]))

    for q in range(mt):
        index[q,:] = torch.from_numpy(np.asarray(list(it.chain.from_iterable(perm[q:q+1]))))

    return index


###### test functions
### 1D
def cos(points,omega=4*math.pi):
    omega=omega
    return torch.cos(torch.squeeze(points, 1) * omega)

def cos_int(points_lims,omega=4*math.pi):
    omega=omega
    return torch.sin(omega*points_lims[:,1])/omega - torch.sin(omega*points_lims[:,0])/omega

def step(points):
    out=points.clone()
    out[points<0.5]=-1
    out[points>0.5]=1
    return out.view(-1)

def step_int(points_lims):
    out=points_lims.clone()-0.5
    out1=out[:,0].clone()
    out1[out1<0]=-out1[out[:,0]<0]
    out2=out[:,1].clone()
    out2[out[:,1]<0]=-out2[out[:,1]<0]
    return out2-out1

def stepsin(points,omega=4*math.pi):
    step=0.5
    omega=omega
    y = torch.sin(torch.squeeze(points,1) * omega)
    y[torch.squeeze(points, 1) >= step] += -1.0
    y[torch.squeeze(points, 1) < step] += +1.0
    return y

def stepsin_int(points_lims,omega=4*math.pi):
    step=0.5
    omega=omega
    a0 = torch.clamp(points_lims[:,0], max=step)
    a1 = torch.clamp(points_lims[:, 1], max=step)
    a2 = torch.clamp(points_lims[:,0], min=step)
    a3 = torch.clamp(points_lims[:, 1], min=step)
    p1 = -torch.cos(omega*a1)/omega+torch.cos(omega*a0)/omega+1.0*(a1-a0)
    p2 = -torch.cos(omega * a3) / omega + torch.cos(omega * a2) / omega - 1.0 * (a3 - a2)
    return p1+p2

### 2D
def circlefunc2(points):
    y=-torch.ones(points.size(0))
    y[torch.sum((points-0.5).pow(2),dim=1).sqrt().view(y.size())>0.22]=1
    y[torch.sum((points-0.5).pow(2),dim=1).sqrt().view(y.size())>0.35]=-1
    indx1=torch.abs(points[:,0]-0.5).view(y.size())<0.075
    indx2=torch.abs(points[:,1]-0.5).view(y.size())<0.075
    y[indx1*indx2]=1
    return y

def circlefunc(points):
    y=torch.zeros(points.size(0))
    y[torch.sum((points-0.5).pow(2),dim=1).sqrt().view(y.size())<0.35]=1
    indx1=torch.abs(points[:,0]-0.5).view(y.size())<0.1
    indx2=torch.abs(points[:,1]-0.5).view(y.size())<0.1
    y[indx1*indx2]=0
    return y
