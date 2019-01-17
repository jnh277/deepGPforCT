import torch.nn as nn
import torch
import math

class GP_1D(nn.Module):
    def __init__(self, sigma_f, lengthscale, sigma_n):
        super(GP_1D, self).__init__()
        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.sigma_f = nn.Parameter(torch.Tensor([sigma_f]))
        self.lengthscale = nn.Parameter(torch.Tensor([lengthscale]))
        self.sigma_n = nn.Parameter(torch.Tensor([sigma_n]))
        self.training = True  # initially we want to be in training mode [don't need this]

    # the predict forward function
    def forward(self, x_train, y_train, m, x_test=None):

        index = torch.empty(1, m)
        for i in range(m):
            index[0, i]=i+1

        # See the autograd section for explanation of what happens here.
        n = x_train.size(0)

        # determine L automatically
        tun = 3.5  # tuning parameter
        L = max(1.5,math.pi*m*torch.sqrt(self.lengthscale.pow(2))/(2.0*tun))

        phi = ( 1/math.sqrt(L) ) * torch.sin(math.pi*index*(x_train+L)*0.5/L) # basis functions
        # diagonal of inverse lambda matrix
        inv_lambda_diag = ( self.sigma_f.pow(-2) * torch.pow(2.0*math.pi*self.lengthscale.pow(2), -0.5)*
                                  torch.exp( 0.5*self.lengthscale.pow(2)*pow(math.pi*index.t() / (2.0*L), 2) ) ).view(m)

        Z = phi.t().mm(phi) + self.sigma_n.pow(2) * torch.diag( inv_lambda_diag )
        phi_lam = torch.cat( (phi , self.sigma_n * torch.diag( torch.sqrt(inv_lambda_diag) )),0)  # [Phi; sign*sqrt(Lambda^-1)]

        _,r = torch.qr(phi_lam)
        v, _ = torch.gesv( phi.t().mm(y_train.view(n, 1)), Z)  # X,LU = torch.gesv(B, A); AX=B => v=(Phi'*Phi+sign^2I)\(Phi'*y)
        if x_test is None:
            out = (r, v, inv_lambda_diag, phi, Z)

        if x_test is not None:
            with torch.no_grad():
                phi_star = 1/math.sqrt(L) * torch.sin(math.pi*index.t()*(x_test.t()+L)*0.5/L)

                # solve
                f_test = phi_star.t().mm(v)
                tmp,_ = torch.trtrs(phi_star,r.t(),upper=False)  # solves r^T*u=phi_star, u=r*x
                tmp,_ = torch.trtrs(tmp,r) # solves r*x=u
                cov_f = self.sigma_n.pow(2)*torch.sum(phi_star.t() * tmp.t(), dim=1)
            out = (f_test, cov_f)
        return out


    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'sigma_f={}, lengthscale={}, sigma_n={}'.format(
            self.sigma_f.item(), self.lengthscale.item(), self.sigma_n.item()
        )

class NegMarginalLogLikelihood(nn.Module):
    def __init__(self):
        super(NegMarginalLogLikelihood, self).__init__()

    def forward(self, sigma_f, lengthscale, sigma_n, x_train, y_train, m):
        nll_st = NegMarginalLogLikelihood_st.apply
        return nll_st(sigma_f, lengthscale, sigma_n, x_train, y_train, m)

class NegMarginalLogLikelihood_st(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sigma_f, lengthscale, sigma_n, x_train, y_train, m):
        index = torch.empty(1, m)
        for i in range(m):
            index[0, i]=i+1

        # See the autograd section for explanation of what happens here.
        n = x_train.size(0)

        # determine L automatically
        tun = 3.5  # tuning parameter
        L = max(1.5*x_train.max(),math.pi*m*torch.sqrt(lengthscale.pow(2))/(2.0*tun))

        # compute the Phi matrix
        phi = ( 1/math.sqrt(L) ) * torch.sin(math.pi*index*(x_train+L)*0.5/L)  # basis functions
        # diagonal of inverse lambda matrix
        inv_lambda_diag = ( sigma_f.pow(-2) * torch.pow(2.0*math.pi*lengthscale.pow(2), -0.5)*
                                  torch.exp( 0.5*lengthscale.pow(2)*pow(math.pi*index.t() / (2.0*L), 2) ) ).view(m)

        Z = phi.t().mm(phi) + sigma_n.pow(2) * torch.diag( inv_lambda_diag )  # Z
        phi_lam = torch.cat( (phi , sigma_n * torch.diag( torch.sqrt(inv_lambda_diag) )),0)  # [Phi; sign*sqrt(Lambda^-1)]

        _,r = torch.qr(phi_lam)  # Z = r^T*r
        v, _ = torch.gesv( phi.t().mm(y_train.view(n, 1)), Z)  # X,LU = torch.gesv(B, A); AX=B => v=(Phi'*Phi+sign^2I)\(Phi'*y)

        n=y_train.size(0)
        logQ = ( (n-m)*torch.log(sigma_n.pow(2))
                + 2.0*torch.sum(torch.log(torch.abs(r.diag())))
                + torch.sum(torch.log(1/inv_lambda_diag)) )
        yQiy = sigma_n.pow(-2)*( y_train.dot(y_train) - v.view(m).dot(y_train.view(1, n).mm(phi).view(m) ) )

        nLL = 0.5*logQ + 0.5*yQiy  # neg log marg likelihood

        # save tensors for the backward pass
        ctx.save_for_backward(sigma_f, lengthscale, sigma_n, r, y_train, v, inv_lambda_diag, phi, Z, index, torch.as_tensor(L))

        return nLL

    @staticmethod
    def backward(ctx, grad_output):
        # load tensors from the forward pass
        sigma_f, lengthscale, sigma_n, r, y_train, v, inv_lambda_diag, phi, Z, index, L, = ctx.saved_tensors

        m = inv_lambda_diag.size(0)  # nr of basis functions
        n = y_train.size(0)  # nr of data points

        # computing Z\Lambda^-1
        Zil,_ = torch.trtrs(torch.diag(inv_lambda_diag),r.t(),upper=False)  # solves r^T*u=Lambda^-1, u=r*x
        Zil,_ = torch.trtrs(Zil,r) # solves r*x=u

        # terms involving loq|Q|
        dlogQ_dsigma_f = 2.0*m/sigma_f -2.0*(sigma_n.pow(2)/sigma_f)*torch.trace(Zil)

        omega_squared = pow(math.pi*index.t() / (2.0*L), 2)
        dlogQ_dlengthscale = ( torch.sum( 1-lengthscale.pow(2)*omega_squared )/lengthscale
                               -sigma_n.pow(2) * torch.trace( Zil.mm(torch.diag(( 1-lengthscale.pow(2)*omega_squared.view(m) )/lengthscale)) ) )

        dlogQ_dsigma_n = 2.0*(n-m)/sigma_n + 2.0*sigma_n*torch.trace(Zil)

        # terms involving invQ
        dyQiy_dsigma_f = -y_train.view(1, n).mm(phi).view(-1).dot( Zil.mm(v).view(-1) ) * 2.0/sigma_f

        dyQiy_dlengthscale = -y_train.view(1, n).mm(phi).mm(Zil).mm(torch.diag(( 1-lengthscale.pow(2)*omega_squared.view(m) )/lengthscale).mm(v))

        dyQiy_dsigma_n = ( (2.0/sigma_n)*( y_train.view(1, n).mm(phi).mm(Zil).mm(v) )
                              +(2.0/sigma_n.pow(3))*y_train.view(1, n).mm(phi).mm(v)  -(2.0/sigma_n.pow(3))*y_train.dot(y_train) )

        grad1 = 0.5*(dlogQ_dsigma_f+dyQiy_dsigma_f)  # derivative wrt sigma_f
        grad2 = 0.5*(dlogQ_dlengthscale+dyQiy_dlengthscale)  # derivative wrt lengthscale
        grad3 = 0.5*(dlogQ_dsigma_n+dyQiy_dsigma_n)  # derivative wrt sigma_n
        return grad1, grad2, grad3, None, None, None


# Deep version
class NegMarginalLogLikelihood_deep(nn.Module):
    def __init__(self):
        super(NegMarginalLogLikelihood_deep, self).__init__()

    def forward(self, sigma_f, lengthscale, sigma_n, m_train, y_train, m):
        nll_st = NegMarginalLogLikelihood_deep_st.apply
        return nll_st(sigma_f, lengthscale, sigma_n, m_train, y_train, m)

class NegMarginalLogLikelihood_deep_st(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sigma_f, lengthscale, sigma_n, m_train, y_train, m):
        index = torch.empty(1, m)
        for i in range(m):
            index[0, i]=i+1

        # See the autograd section for explanation of what happens here.
        n = m_train.size(0)

        # determine L automatically
        tun = 3.5  # tuning parameter
        L = max(1.5*m_train.max(),math.pi*m*torch.sqrt(lengthscale.pow(2))/(2.0*tun))

        # compute the Phi matrix
        phi = ( 1/math.sqrt(L) ) * torch.sin(math.pi*index*(m_train+L)*0.5/L)  # basis functions
        # diagonal of inverse lambda matrix, OBS FOR GENERALISATION: dependent on input dim
        inv_lambda_diag = ( sigma_f.pow(-2) * torch.pow(2.0*math.pi*lengthscale.pow(2), -0.5)*
                                  torch.exp( 0.5*lengthscale.pow(2)*pow(math.pi*index.t() / (2.0*L), 2) ) ).view(m)

        Z = phi.t().mm(phi) + sigma_n.pow(2) * torch.diag( inv_lambda_diag )  # Z
        phi_lam = torch.cat( (phi , sigma_n * torch.diag( torch.sqrt(inv_lambda_diag) )),0)  # [Phi; sign*sqrt(Lambda^-1)]

        _,r = torch.qr(phi_lam)
        v, _ = torch.gesv( phi.t().mm(y_train.view(n, 1)), Z)  # X,LU = torch.gesv(B, A); AX=B => v=(Phi'*Phi+sign^2I)\(Phi'*y)=Z\(Phi'*y)

        n=y_train.size(0)
        logQ = ( (n-m)*torch.log(sigma_n.pow(2))
                + 2.0*torch.sum(torch.log(torch.abs(r.diag())))
                + torch.sum(torch.log(1/inv_lambda_diag)) )
        yQiy = sigma_n.pow(-2)*( y_train.dot(y_train) - v.view(m).dot(y_train.view(1, n).mm(phi).view(m) ) )

        nLL = 0.5*logQ + 0.5*yQiy  # neg log marg likelihood

        # save tensors for the backward pass
        ctx.save_for_backward(sigma_f, lengthscale, sigma_n, r, m_train, y_train, v, inv_lambda_diag, phi, Z, index, torch.as_tensor(L))

        return nLL

    @staticmethod
    def backward(ctx, grad_output):
        # load tensors from the forward pass
        sigma_f, lengthscale, sigma_n, r, m_train, y_train, v, inv_lambda_diag, phi, Z, index, L, = ctx.saved_tensors

        m = inv_lambda_diag.size(0)  # nr of basis functions
        n = y_train.size(0)  # nr of data points

        # computing Z\Lambda^-1
        Zil,_ = torch.trtrs(torch.diag(inv_lambda_diag),r.t(),upper=False)  # solves r^T*u=Lambda^-1, u=r*x
        Zil,_ = torch.trtrs(Zil,r) # solves r*x=u

        # OBS FOR GENERALISATION: lengthscale derivatives dependent on input dim
        # terms involving loq|Q|
        dlogQ_dsigma_f = 2.0*m/sigma_f -2.0*(sigma_n.pow(2)/sigma_f)*torch.trace(Zil)

        omega_squared = pow(math.pi*index.t() / (2.0*L), 2)
        dlogQ_dlengthscale = ( torch.sum( 1-lengthscale.pow(2)*omega_squared )/lengthscale
                               -sigma_n.pow(2) * torch.trace( Zil.mm(torch.diag(( 1-lengthscale.pow(2)*omega_squared.view(m) )/lengthscale)) ) )

        dlogQ_dsigma_n = 2.0*(n-m)/sigma_n + 2.0*sigma_n*torch.trace(Zil)

        # terms involving invQ
        dyQiy_dsigma_f = -y_train.view(1, n).mm(phi).view(-1).dot( Zil.mm(v).view(-1) ) * 2.0/sigma_f

        dyQiy_dlengthscale = -y_train.view(1, n).mm(phi).mm(Zil).mm(torch.diag(( 1-lengthscale.pow(2)*omega_squared.view(m) )/lengthscale).mm(v))

        dyQiy_dsigma_n = ( (2.0/sigma_n)*( y_train.view(1, n).mm(phi).mm(Zil).mm(v) )
                              +(2.0/sigma_n.pow(3))*y_train.view(1, n).mm(phi).mm(v)  -(2.0/sigma_n.pow(3))*y_train.dot(y_train) )

        # collect the derivatives
        grad1 = 0.5*(dlogQ_dsigma_f+dyQiy_dsigma_f)  # derivative wrt sigma_f
        grad2 = 0.5*(dlogQ_dlengthscale+dyQiy_dlengthscale)  # derivative wrt lengthscale
        grad3 = 0.5*(dlogQ_dsigma_n+dyQiy_dsigma_n)  # derivative wrt sigma_n

        # now compute the m_train derivatives
        gradm = torch.zeros(n)  # tensor holding the partial derivatives
        dphi_dm = 0.5*math.pi*index.repeat(n,1)/(L.pow(3/2)) * torch.cos(math.pi*(m_train+L)*index*0.5/L)
        for k in range(n):
            # calculate dZ/dm
            dZ_dm = 2.0*phi[k,:].view(m,1).mm(dphi_dm[k,:].view(1,m))
            # start with the log|Q| part
            # compute Z\dZ_dm
            ZidZd_m,_ = torch.trtrs(dZ_dm,r.t(),upper=False)  # solves r^T*u=dZ/dm, u=r*x
            ZidZd_m,_ = torch.trtrs(ZidZd_m,r) # solves r*x=u
            dlogQ_dm = torch.trace(ZidZd_m)
            dyQiy_dm = -sigma_n.pow(-2)*( -y_train.view(1, n).mm(phi).mm(ZidZd_m).mm(v) + 2.0*dphi_dm[k,:].view(1,m).mul(y_train[k]).mm(v) )
            gradm[k] = 0.5*(dlogQ_dm+dyQiy_dm)

        return grad1, grad2, grad3, gradm.view(n,1), None, None
