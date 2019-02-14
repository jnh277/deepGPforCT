import torch.nn as nn
import torch
import math

# TODO: extend to 2D

class GP_1D(nn.Module):
    def __init__(self, sigma_f, lengthscale, sigma_n, covtype="se", nu=2.5):
        super(GP_1D, self).__init__()
        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.log_sigma_f = nn.Parameter(torch.Tensor([sigma_f]).abs().log())
        self.log_lengthscale = nn.Parameter(torch.Tensor([lengthscale]).abs().log())
        self.log_sigma_n = nn.Parameter(torch.Tensor([sigma_n]).abs().log())
        self.covtype = covtype
        self.nu = nu

    # the predict forward function
    def forward(self, x_train, y_train, m, x_test=None):

        # extract hyperparameters
        sigma_f = torch.exp(self.log_sigma_f)
        lengthscale = torch.exp(self.log_lengthscale)
        sigma_n = torch.exp(self.log_sigma_n)

        # create an index vector, index=[1 2 3...]
        index = torch.linspace(1, m, m).view(1,m)

        # See the autograd section for explanation of what happens here.
        n = x_train.size(0)

        # determine L automatically
        tun = 3 # tuning parameter
        L = max(1.2*x_train.max(),math.pi*m*torch.sqrt(lengthscale.detach().pow(2))/(2.0*tun))

        # compute phi
        phi = ( 1/math.sqrt(L) ) * torch.sin(math.pi*index*(x_train+L)*0.5/L) # basis functions

        # diagonal of inverse lambda matrix
        if self.covtype is "matern":
            dim = 1
            inv_lambda_diag = 1/( sigma_f.pow(2)*math.pow(2.0,dim)*math.pow(math.pi,dim/2.0)*math.gamma(self.nu+dim/2.0)*
                    math.pow(2.0*self.nu,self.nu)*( 2.0*self.nu*lengthscale.pow(-2) + pow(math.pi*index.t() / (2.0*L), 2) ).pow(-self.nu-dim/2.0)
                    / (math.gamma(self.nu)*lengthscale.pow(2.0*self.nu)) ).view(m)

        if self.covtype is "se":
            inv_lambda_diag = ( sigma_f.pow(-2) * torch.pow(2.0*math.pi*lengthscale.pow(2), -0.5)*
                                      torch.exp( 0.5*lengthscale.pow(2)*pow(math.pi*index.t() / (2.0*L), 2) ) ).view(m)

        Z = phi.t().mm(phi) + sigma_n.pow(2) * torch.diag( inv_lambda_diag )
        phi_lam = torch.cat( (phi , sigma_n * torch.diag( torch.sqrt(inv_lambda_diag) )),0)  # [Phi; sign*sqrt(Lambda^-1)]

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
                cov_f = sigma_n.pow(2)*torch.sum(phi_star.t() * tmp.t(), dim=1)
            out = (f_test, cov_f)
        return out


    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'sigma_f={}, lengthscale={}, sigma_n={}'.format(
            torch.exp(self.log_sigma_f).item(), torch.exp(self.log_lengthscale).item(), torch.exp(self.log_sigma_n).item()
        )


# create new model that takes phi as input
class GP_1D_new(nn.Module):
    def __init__(self, sigma_f, lengthscale, sigma_n, covtype="se", nu=2.5):
        super(GP_1D_new, self).__init__()
        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.log_sigma_f = nn.Parameter(torch.Tensor([sigma_f]).abs().log())
        self.log_lengthscale = nn.Parameter(torch.Tensor([lengthscale]).abs().log())
        self.log_sigma_n = nn.Parameter(torch.Tensor([sigma_n]).abs().log())
        self.covtype = covtype
        self.nu = nu

    # the predict forward function
    def forward(self, y_train, phi, m, L, m_test):
        # extract hyperparameters
        sigma_f = torch.exp(self.log_sigma_f)
        lengthscale = torch.exp(self.log_lengthscale)
        sigma_n = torch.exp(self.log_sigma_n)

        # create an index vector, index=[1 2 3...]
        index = torch.linspace(1, m, m).view(1,m)

        # See the autograd section for explanation of what happens here.
        n = y_train.size(0)

        # diagonal of inverse lambda matrix
        if self.covtype is "matern":
            dim = 1
            inv_lambda_diag = 1/( sigma_f.pow(2)*math.pow(2.0,dim)*math.pow(math.pi,dim/2.0)*math.gamma(self.nu+dim/2.0)*
                    math.pow(2.0*self.nu,self.nu)*( 2.0*self.nu*lengthscale.pow(-2) + pow(math.pi*index.t() / (2.0*L), 2) ).pow(-self.nu-dim/2.0)
                    / (math.gamma(self.nu)*lengthscale.pow(2.0*self.nu)) ).view(m)

        if self.covtype is "se":
            inv_lambda_diag = ( sigma_f.pow(-2) * torch.pow(2.0*math.pi*lengthscale.pow(2), -0.5)*
                                      torch.exp( 0.5*lengthscale.pow(2)*pow(math.pi*index.t() / (2.0*L), 2) ) ).view(m)

        # print(inv_lambda_diag[m-1])

        Z = phi.t().mm(phi) + sigma_n.pow(2) * torch.diag( inv_lambda_diag )
        phi_lam = torch.cat( (phi , sigma_n * torch.diag( torch.sqrt(inv_lambda_diag) )),0)  # [Phi; sign*sqrt(Lambda^-1)]

        _,r = torch.qr(phi_lam)
        v, _ = torch.gesv( phi.t().mm(y_train.view(n, 1)), Z)  # X,LU = torch.gesv(B, A); AX=B => v=(Phi'*Phi+sign^2I)\(Phi'*y)

        # compute phi_star
        phi_star = 1/math.sqrt(L) * torch.sin(math.pi*index.t()*(m_test.t()+L)*0.5/L)

        # predict
        f_test = phi_star.t().mm(v)
        tmp,_ = torch.trtrs(phi_star,r.t(),upper=False)  # solves r^T*u=phi_star, u=r*x
        tmp,_ = torch.trtrs(tmp,r) # solves r*x=u
        cov_f = sigma_n.pow(2)*torch.sum(phi_star.t() * tmp.t(), dim=1)
        out = (f_test, cov_f)
        return out

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'sigma_f={}, lengthscale={}, sigma_n={}'.format(
            torch.exp(self.log_sigma_f).item(), torch.exp(self.log_lengthscale).item(), torch.exp(self.log_sigma_n).item()
        )


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
    def forward(self, y_train, phi, sq_lambda, L, m_test):  # todo: general L:s
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
            lprod*=lengthscale[q].pow(2)
            omega_sum+=lengthscale[q].pow(2)*sq_lambda[:,q].view(m,1).pow(2)

        inv_lambda_diag = ( sigma_f.pow(-2) * math.pow(2.0*math.pi,diml/2) *torch.pow(lprod, -0.5)*
                                  torch.exp( 0.5*omega_sum ) ).view(m)

        Z = phi.t().mm(phi) + sigma_n.pow(2) * torch.diag( inv_lambda_diag )
        phi_lam = torch.cat( (phi , sigma_n * torch.diag( torch.sqrt(inv_lambda_diag) )),0)  # [Phi; sign*sqrt(Lambda^-1)]

        _,r = torch.qr(phi_lam)
        v, _ = torch.gesv( phi.t().mm(y_train.view(n, 1)), Z)  # X,LU = torch.gesv(B, A); AX=B => v=(Phi'*Phi+sign^2I)\(Phi'*y)

        # compute phi_star
        nt = m_test.size(0)
        phi_star=torch.ones(m,nt)
        for q in range(diml):
            phi_star *= 1/math.sqrt(L[q]) * torch.sin(sq_lambda[:,q].view(m,1)*(m_test[:,q].view(1,nt)+L[q]))

        # predict
        f_test = phi_star.t().mm(v)
        tmp,_ = torch.trtrs(phi_star,r.t(),upper=False)  # solves r^T*u=phi_star, u=r*x
        tmp,_ = torch.trtrs(tmp,r) # solves r*x=u
        cov_f = sigma_n.pow(2)*torch.sum(phi_star.t() * tmp.t(), dim=1)
        out = (f_test, cov_f)
        return out

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'sigma_f={}, lengthscale={}, sigma_n={}'.format(
            torch.exp(self.log_sigma_f).item(), torch.exp(self.log_lengthscale), torch.exp(self.log_sigma_n).item()
        )



# neg log marg like
class NegMarginalLogLikelihood(nn.Module):
    def __init__(self,covtype="se",nu=2.5):
        super(NegMarginalLogLikelihood, self).__init__()
        self.covtype = covtype
        self.nu = nu

    def forward(self, log_sigma_f, log_lengthscale, log_sigma_n, x_train, y_train, m):
        nll_st = NegMarginalLogLikelihood_st.apply
        return nll_st(log_sigma_f, log_lengthscale, log_sigma_n, x_train, y_train, m, self.covtype, self.nu)

class NegMarginalLogLikelihood_st(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_sigma_f, log_lengthscale, log_sigma_n, x_train, y_train, m, covtype, nu):
        # extract hyperparameters
        sigma_f = torch.exp(log_sigma_f)
        lengthscale = torch.exp(log_lengthscale)
        sigma_n = torch.exp(log_sigma_n)

        # create an index vector, index=[1 2 3...]
        index = torch.linspace(1, m, m).view(1,m)

        # See the autograd section for explanation of what happens here.
        n = x_train.size(0)

        # determine L automatically
        tun = 3  # tuning parameter
        L = max(1.5*x_train.max(),math.pi*m*torch.sqrt(lengthscale.detach().pow(2))/(2.0*tun))

        # compute the Phi matrix
        phi = ( 1/math.sqrt(L) ) * torch.sin(math.pi*index*(x_train+L)*0.5/L)  # basis functions

        # diagonal of inverse lambda matrix
        if covtype is "se":
            covtypeNum = torch.ones(1,dtype=torch.int32)
            inv_lambda_diag = ( sigma_f.pow(-2) * torch.pow(2.0*math.pi*lengthscale.pow(2), -0.5)*
                                      torch.exp( 0.5*lengthscale.pow(2)*pow(math.pi*index.t() / (2.0*L), 2) ) ).view(m)

        if covtype is "matern":
            covtypeNum = 2*torch.ones(1,dtype=torch.int32)
            dim = 1
            inv_lambda_diag = 1/( sigma_f.pow(2)*math.pow(2.0,dim)*math.pow(math.pi,dim/2.0)*math.gamma(nu+dim/2.0)*
                    math.pow(2.0*nu,nu)*( 2.0*nu*lengthscale.pow(-2) + pow(math.pi*index.t() / (2.0*L), 2) ).pow(-nu-dim/2.0)
                    / (math.gamma(nu)*lengthscale.pow(2.0*nu)) ).view(m)

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
        ctx.save_for_backward(sigma_f, lengthscale, sigma_n, r, y_train, v, inv_lambda_diag, phi, Z, index, torch.as_tensor(L), covtypeNum, torch.as_tensor(nu))

        return nLL

    @staticmethod
    def backward(ctx, grad_output):
        # load tensors from the forward pass
        sigma_f, lengthscale, sigma_n, r, y_train, v, inv_lambda_diag, phi, Z, index, L, covtypeNum, nu, = ctx.saved_tensors

        m = inv_lambda_diag.size(0)  # nr of basis functions
        n = y_train.size(0)  # nr of data points

        # computing Z\Lambda^-1
        Zil,_ = torch.trtrs(torch.diag(inv_lambda_diag),r.t(),upper=False)  # solves r^T*u=Lambda^-1, u=r*x
        Zil,_ = torch.trtrs(Zil,r) # solves r*x=u

        omega_squared = pow(math.pi*index.t() / (2.0*L), 2)

        if covtypeNum.item() is 1: # se
            # terms involving loq|Q|
            dlogQ_dlog_sigma_f = 2.0*m -2.0*(sigma_n.pow(2))*torch.trace(Zil)

            dlogQ_dlog_lengthscale = ( torch.sum( 1-lengthscale.pow(2)*omega_squared )
                                   -sigma_n.pow(2) * torch.sum( Zil*torch.diag( 1-lengthscale.pow(2)*omega_squared.view(m) )  ,dim=1) )

            dlogQ_dlog_sigma_n = 2.0*(n-m)+ 2.0*sigma_n.pow(2)*torch.trace(Zil)

            # terms involving invQ
            dyQiy_dlog_sigma_f = -y_train.view(1, n).mm(phi).view(-1).dot( Zil.mm(v).view(-1) ) * 2.0

            dyQiy_dlog_lengthscale = -y_train.view(1, n).mm(phi).mm(Zil).mm(torch.diag(( 1-lengthscale.pow(2)*omega_squared.view(m) )).mm(v))

            dyQiy_dlog_sigma_n = ( 2.0*( y_train.view(1, n).mm(phi).mm(Zil).mm(v) )
                                  +(2.0*sigma_n.pow(-2))*y_train.view(1, n).mm(phi).mm(v)  -(2.0*sigma_n.pow(-2))*y_train.dot(y_train) )

        if covtypeNum.item() is 2: # matern
            nu = nu.item()
            dim=1
            SD_fac = (math.pow(2.0,dim)*math.pow(math.pi,dim/2.0)*math.gamma(nu+dim/2.0)*math.pow(2.0*nu,nu))/math.gamma(nu)
            SD_par = 2*nu*lengthscale.pow(-2)+omega_squared

            # terms involving loq|Q|
            dlogQ_dlog_sigma_f = 2.0*m -2.0*(sigma_n.pow(2))*torch.trace(Zil)

            dlogQ_dlog_lengthscale = ( 2.0*nu*( 2.0*(nu+dim/2.0)*lengthscale.pow(-2)/SD_par - 1 ).sum()
                                       -sigma_n.pow(2) * torch.sum( Zil*torch.diag( 2.0*nu*( 2.0*(nu+dim/2.0)*lengthscale.pow(-2)/SD_par - 1 ).view(m)  )  ,dim=1)  )

            dlogQ_dlog_sigma_n = 2.0*(n-m)+ 2.0*sigma_n.pow(2)*torch.trace(Zil)

            # terms involving invQ
            dyQiy_dlog_sigma_f = -y_train.view(1, n).mm(phi).view(-1).dot( Zil.mm(v).view(-1) ) * 2.0

            dyQiy_dlog_lengthscale = -y_train.view(1, n).mm(phi).mm(Zil).mm(torch.diag( 2.0*nu*( 2.0*(nu+dim/2.0)*lengthscale.pow(-2)/SD_par - 1 ).view(m)  ).mm(v))

            dyQiy_dlog_sigma_n = ( 2.0*( y_train.view(1, n).mm(phi).mm(Zil).mm(v) )
                                  +(2.0*sigma_n.pow(-2))*y_train.view(1, n).mm(phi).mm(v)  -(2.0*sigma_n.pow(-2))*y_train.dot(y_train) )

        grad1 = 0.5*(dlogQ_dlog_sigma_f+dyQiy_dlog_sigma_f)  # derivative wrt log_sigma_f
        grad2 = 0.5*(dlogQ_dlog_lengthscale+dyQiy_dlog_lengthscale)  # derivative wrt log_lengthscale
        grad3 = 0.5*(dlogQ_dlog_sigma_n+dyQiy_dlog_sigma_n)  # derivative wrt log_sigma_n
        return grad1, grad2, grad3, None, None, None, None, None


# Deep version
class NegMarginalLogLikelihood_deep(nn.Module):
    def __init__(self,covtype="se",nu=2.5):
        super(NegMarginalLogLikelihood_deep, self).__init__()
        self.covtype = covtype
        self.nu = nu

    def forward(self, log_sigma_f, log_lengthscale, log_sigma_n, m_train, y_train, m):
        nll_st = NegMarginalLogLikelihood_deep_st.apply
        return nll_st(log_sigma_f, log_lengthscale, log_sigma_n, m_train, y_train, m, self.covtype, self.nu)

class NegMarginalLogLikelihood_deep_st(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_sigma_f, log_lengthscale, log_sigma_n, m_train, y_train, m, covtype, nu):
        # extract hyperparameters
        sigma_f = torch.exp(log_sigma_f)
        lengthscale = torch.exp(log_lengthscale)
        sigma_n = torch.exp(log_sigma_n)

        # create an index vector, index=[1 2 3...]
        index = torch.linspace(1, m, m).view(1,m)

        # extract the data set size
        n = m_train.size(0)

        # determine L automatically
        tun = 3  # tuning parameter
        L = max(1.5*m_train.max(),math.pi*m*torch.sqrt(lengthscale.detach().pow(2))/(2.0*tun))
        # print(L)

        # compute the Phi matrix
        phi = ( 1/math.sqrt(L) ) * torch.sin(math.pi*index*(m_train+L)*0.5/L)  # basis functions

        # diagonal of inverse lambda matrix, OBS FOR GENERALISATION: dependent on input dim
        if covtype is "se":
            covtypeNum = torch.ones(1,dtype=torch.int32)
            inv_lambda_diag = ( sigma_f.pow(-2) * torch.pow(2.0*math.pi*lengthscale.pow(2), -0.5)*
                                      torch.exp( 0.5*lengthscale.pow(2)*pow(math.pi*index.t() / (2.0*L), 2) ) ).view(m)

        if covtype is "matern":
            covtypeNum = 2*torch.ones(1,dtype=torch.int32)
            dim = 1
            inv_lambda_diag = 1/( sigma_f.pow(2)*math.pow(2.0,dim)*math.pow(math.pi,dim/2.0)*math.gamma(nu+dim/2.0)*
                    math.pow(2.0*nu,nu)*( 2.0*nu*lengthscale.pow(-2) + pow(math.pi*index.t() / (2.0*L), 2) ).pow(-nu-dim/2.0)
                    / (math.gamma(nu)*lengthscale.pow(2.0*nu)) ).view(m)

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
        ctx.save_for_backward(sigma_f, lengthscale, sigma_n, r, m_train, y_train, v, inv_lambda_diag, phi, Z, index, torch.as_tensor(L), covtypeNum, torch.as_tensor(nu))

        return nLL

    @staticmethod
    def backward(ctx, grad_output):
        # load tensors from the forward pass
        sigma_f, lengthscale, sigma_n, r, m_train, y_train, v, inv_lambda_diag, phi, Z, index, L, covtypeNum, nu, = ctx.saved_tensors

        m = inv_lambda_diag.size(0)  # nr of basis functions
        n = y_train.size(0)  # nr of data points

        # computing Z\Lambda^-1
        Zil,_ = torch.trtrs(torch.diag(inv_lambda_diag),r.t(),upper=False)  # solves r^T*u=Lambda^-1, u=r*x
        Zil,_ = torch.trtrs(Zil,r) # solves r*x=u

        omega_squared = pow(math.pi*index.t() / (2.0*L), 2)

        if covtypeNum.item() is 1: # se # OBS FOR GENERALISATION: lengthscale derivatives dependent on input dim
            # terms involving loq|Q|
            dlogQ_dlog_sigma_f = 2.0*m -2.0*(sigma_n.pow(2))*torch.trace(Zil)

            dlogQ_dlog_lengthscale = ( torch.sum( 1-lengthscale.pow(2)*omega_squared )
                                   -sigma_n.pow(2) * (torch.sum( Zil*torch.diag( 1-lengthscale.pow(2)*omega_squared.view(m) )  ,dim=1)).sum() )

            dlogQ_dlog_sigma_n = 2.0*(n-m)+ 2.0*sigma_n.pow(2)*torch.trace(Zil)

            # terms involving invQ
            dyQiy_dlog_sigma_f = -y_train.view(1, n).mm(phi).view(-1).dot( Zil.mm(v).view(-1) ) * 2.0

            dyQiy_dlog_lengthscale = -y_train.view(1, n).mm(phi).mm(Zil).mm(torch.diag(( 1-lengthscale.pow(2)*omega_squared.view(m) )).mm(v))

            dyQiy_dlog_sigma_n = ( 2.0*( y_train.view(1, n).mm(phi).mm(Zil).mm(v) )
                                  +(2.0*sigma_n.pow(-2))*y_train.view(1, n).mm(phi).mm(v)  -(2.0*sigma_n.pow(-2))*y_train.dot(y_train) )

        if covtypeNum.item() is 2: # matern
            nu = nu.item()
            dim=1
            SD_fac = (math.pow(2.0,dim)*math.pow(math.pi,dim/2.0)*math.gamma(nu+dim/2.0)*math.pow(2.0*nu,nu))/math.gamma(nu)
            SD_par = 2*nu*lengthscale.pow(-2)+omega_squared

            # terms involving loq|Q|
            dlogQ_dlog_sigma_f = 2.0*m -2.0*(sigma_n.pow(2))*torch.trace(Zil)

            dlogQ_dlog_lengthscale = ( 2.0*nu*( 2.0*(nu+dim/2.0)*lengthscale.pow(-2)/SD_par - 1 ).sum()
                                       -sigma_n.pow(2) * torch.sum( Zil*torch.diag( 2.0*nu*( 2.0*(nu+dim/2.0)*lengthscale.pow(-2)/SD_par - 1 ).view(m)  )  ,dim=1)  )

            dlogQ_dlog_sigma_n = 2.0*(n-m)+ 2.0*sigma_n.pow(2)*torch.trace(Zil)

            # terms involving invQ
            dyQiy_dlog_sigma_f = -y_train.view(1, n).mm(phi).view(-1).dot( Zil.mm(v).view(-1) ) * 2.0

            dyQiy_dlog_lengthscale = -y_train.view(1, n).mm(phi).mm(Zil).mm(torch.diag( 2.0*nu*( 2.0*(nu+dim/2.0)*lengthscale.pow(-2)/SD_par - 1 ).view(m)  ).mm(v))

            dyQiy_dlog_sigma_n = ( 2.0*( y_train.view(1, n).mm(phi).mm(Zil).mm(v) )
                                  +(2.0*sigma_n.pow(-2))*y_train.view(1, n).mm(phi).mm(v)  -(2.0*sigma_n.pow(-2))*y_train.dot(y_train) )

        grad1 = 0.5*(dlogQ_dlog_sigma_f+dyQiy_dlog_sigma_f)  # derivative wrt log_sigma_f
        grad2 = 0.5*(dlogQ_dlog_lengthscale+dyQiy_dlog_lengthscale)  # derivative wrt log_lengthscale
        grad3 = 0.5*(dlogQ_dlog_sigma_n+dyQiy_dlog_sigma_n)  # derivative wrt log_sigma_n

        # now compute the m_train derivatives
        gradm = torch.zeros(n)  # tensor holding the partial derivatives
        dphi_dm = 0.5*math.pi*index.repeat(n,1)/(L.pow(3/2)) * torch.cos(math.pi*(m_train+L)*index*0.5/L)
        for k in range(n):
            # calculate dZ/dm
            dZ_dm = 2.0*phi[k,:].view(m,1).mm(dphi_dm[k,:].view(1,m))
            # compute Z\dZ_dm
            ZidZd_m,_ = torch.trtrs(dZ_dm,r.t(),upper=False)  # solves r^T*u=dZ/dm, u=r*x
            ZidZd_m,_ = torch.trtrs(ZidZd_m,r) # solves r*x=u
            # start with the log|Q| part
            dlogQ_dm = torch.trace(ZidZd_m)
            # now do the y^T*Q^-1*y part
            dyQiy_dm = -sigma_n.pow(-2)*( -y_train.view(1, n).mm(phi).mm(ZidZd_m).mm(v) + 2.0*dphi_dm[k,:].view(1,m).mul(y_train[k]).mm(v) )
            # sum up
            gradm[k] = 0.5*(dlogQ_dm+dyQiy_dm)

        return grad1, grad2, grad3, gradm.view(n,1), None, None, None, None


# Deep version with numerical integration
class NegMarginalLogLikelihood_deep_intMeas(nn.Module):
    def __init__(self,covtype="se",nu=2.5):
        super(NegMarginalLogLikelihood_deep_intMeas, self).__init__()
        self.covtype = covtype
        self.nu = nu

    def forward(self, log_sigma_f, log_lengthscale, log_sigma_n, phi_vec, y_train, sq_lambda):
        nll_st = NegMarginalLogLikelihood_deep_intMeas_st.apply
        return nll_st(log_sigma_f, log_lengthscale, log_sigma_n, phi_vec, y_train, sq_lambda, self.covtype, self.nu)

class NegMarginalLogLikelihood_deep_intMeas_st(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_sigma_f, log_lengthscale, log_sigma_n, phi_vec, y_train, sq_lambda, covtype, nu):
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

        # reform phi
        phi = phi_vec.view(n,m)

        # diagonal of inverse lambda matrix, OBS FOR GENERALISATION: dependent on input dim
        if covtype is "se":
            covtypeNum = torch.ones(1,dtype=torch.int32)

            lprod=torch.ones(1)
            omega_sum=torch.zeros(m,1)
            for q in range(dim):
                lprod*=lengthscale[q].pow(2)
                omega_sum+=lengthscale[q].pow(2)*sq_lambda[:,q].view(m,1).pow(2)

            inv_lambda_diag = ( sigma_f.pow(-2) * math.pow(2.0*math.pi,dim/2) *torch.pow(lprod, -0.5)*
                                      torch.exp( 0.5*omega_sum ) ).view(m)

        # if covtype is "matern":
        #     covtypeNum = 2*torch.ones(1,dtype=torch.int32)
        #     inv_lambda_diag = 1/( sigma_f.pow(2)*math.pow(2.0,dim)*math.pow(math.pi,dim/2.0)*math.gamma(nu+dim/2.0)*
        #             math.pow(2.0*nu,nu)*( 2.0*nu*lengthscale.pow(-2) + pow(math.pi*index.t() / (2.0*L), 2) ).pow(-nu-dim/2.0)
        #             / (math.gamma(nu)*lengthscale.pow(2.0*nu)) ).view(m)

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
        ctx.save_for_backward(sigma_f, lengthscale, sigma_n, r, y_train, v, inv_lambda_diag, phi_vec, sq_lambda, covtypeNum, torch.as_tensor(nu))

        return nLL

    @staticmethod
    def backward(ctx, grad_output):
        # load tensors from the forward pass
        sigma_f, lengthscale, sigma_n, r, y_train, v, inv_lambda_diag, phi_vec, sq_lambda, covtypeNum, nu, = ctx.saved_tensors

        m = inv_lambda_diag.size(0)  # nr of basis functions
        diml = sq_lambda.size(1)
        n = y_train.size(0)  # nr of data points

        # reform phi
        phi = phi_vec.view(n,m)

        # computing Z\Lambda^-1
        Zil,_ = torch.trtrs(torch.diag(inv_lambda_diag),r.t(),upper=False)  # solves r^T*u=Lambda^-1, u=r*x
        Zil,_ = torch.trtrs(Zil,r) # solves r*x=u

        if covtypeNum.item() is 1: # se # OBS FOR GENERALISATION: lengthscale derivatives dependent on input dim
            # allocate lengthscale
            dlogQ_dlog_lengthscale = torch.zeros(diml)
            dyQiy_dlog_lengthscale = torch.zeros(diml)

            # terms involving loq|Q|
            dlogQ_dlog_sigma_f = 2.0*m -2.0*(sigma_n.pow(2))*torch.trace(Zil)
            for q in range(diml):
                dlogQ_dlog_lengthscale[q] = ( torch.sum( 1-lengthscale[q].pow(2)*sq_lambda[:,q].pow(2) )
                                       -sigma_n.pow(2) * (torch.sum( Zil*torch.diag( 1-lengthscale[q].pow(2)*sq_lambda[:,q].pow(2) )  ,dim=1)).sum() )

            dlogQ_dlog_sigma_n = 2.0*(n-m)+ 2.0*sigma_n.pow(2)*torch.trace(Zil)

            # terms involving invQ
            dyQiy_dlog_sigma_f = -y_train.view(1, n).mm(phi).view(-1).dot( Zil.mm(v).view(-1) ) * 2.0

            for q in range(diml):
                dyQiy_dlog_lengthscale[q] = -y_train.view(1, n).mm(phi).mm(Zil).mm(torch.diag(( 1-lengthscale[q].pow(2)*sq_lambda[:,q].pow(2) )).mm(v))

            dyQiy_dlog_sigma_n = ( 2.0*( y_train.view(1, n).mm(phi).mm(Zil).mm(v) )
                                  +(2.0*sigma_n.pow(-2))*y_train.view(1, n).mm(phi).mm(v)  -(2.0*sigma_n.pow(-2))*y_train.dot(y_train) )

        # if covtypeNum.item() is 2: # matern # OBS: NOT IMPLEMENTED FOR DIMENSIONS>1
        #     nu = nu.item()
        #     SD_fac = (math.pow(2.0,dim)*math.pow(math.pi,dim/2.0)*math.gamma(nu+dim/2.0)*math.pow(2.0*nu,nu))/math.gamma(nu)
        #     SD_par = 2*nu*lengthscale.pow(-2)+omega_squared
        #
        #     # terms involving loq|Q|
        #     dlogQ_dlog_sigma_f = 2.0*m -2.0*(sigma_n.pow(2))*torch.trace(Zil)
        #
        #     dlogQ_dlog_lengthscale = ( 2.0*nu*( 2.0*(nu+dim/2.0)*lengthscale.pow(-2)/SD_par - 1 ).sum()
        #                                -sigma_n.pow(2) * torch.sum( Zil*torch.diag( 2.0*nu*( 2.0*(nu+dim/2.0)*lengthscale.pow(-2)/SD_par - 1 ).view(m)  )  ,dim=1)  )
        #
        #     dlogQ_dlog_sigma_n = 2.0*(n-m)+ 2.0*sigma_n.pow(2)*torch.trace(Zil)
        #
        #     # terms involving invQ
        #     dyQiy_dlog_sigma_f = -y_train.view(1, n).mm(phi).view(-1).dot( Zil.mm(v).view(-1) ) * 2.0
        #
        #     dyQiy_dlog_lengthscale = -y_train.view(1, n).mm(phi).mm(Zil).mm(torch.diag( 2.0*nu*( 2.0*(nu+dim/2.0)*lengthscale.pow(-2)/SD_par - 1 ).view(m)  ).mm(v))
        #
        #     dyQiy_dlog_sigma_n = ( 2.0*( y_train.view(1, n).mm(phi).mm(Zil).mm(v) )
        #                           +(2.0*sigma_n.pow(-2))*y_train.view(1, n).mm(phi).mm(v)  -(2.0*sigma_n.pow(-2))*y_train.dot(y_train) )

        grad1 = 0.5*(dlogQ_dlog_sigma_f+dyQiy_dlog_sigma_f)  # derivative wrt log_sigma_f
        grad2 = 0.5*(dlogQ_dlog_lengthscale+dyQiy_dlog_lengthscale)  # derivative wrt log_lengthscale
        grad3 = 0.5*(dlogQ_dlog_sigma_n+dyQiy_dlog_sigma_n)  # derivative wrt log_sigma_n

        # now compute the phi derivatives (can be done better using the properties of the single entry matrices J (see matrix cookbook))
        gradphi = torch.zeros(n,m)
        for k in range(n):
            for l in range(m):
                # compute Jkl and Jlk
                Jkl = torch.zeros(n,m)
                Jkl[k,l]=1.0
                Jlk = Jkl.t()

                # compute dZdphi
                dZdphi = phi.t().mm(Jkl) + Jlk.mm(phi)

                # compute Z\dZdphi
                ZidZdphi,_ = torch.trtrs(dZdphi,r.t(),upper=False)  # solves r^T*u=dZdphi, u=r*x
                ZidZdphi,_ = torch.trtrs(ZidZdphi,r) # solves r*x=u

                # compute Z\Jlk
                ZiJlk,_ = torch.trtrs(Jlk,r.t(),upper=False)  # solves r^T*u=Jlk, u=r*x
                ZiJlk,_ = torch.trtrs(ZiJlk,r) # solves r*x=u

                # log|Q| part
                dlogQ_dphi = torch.trace(ZidZdphi)

                # y^T*Q^-1*y part
                dyQiy_dphi = -sigma_n.pow(-2)*y_train.view(1, n).mm(
                                                  phi.mm(ZiJlk).mm(y_train.view(n,1))
                                                + Jkl.mm(v)
                                                - phi.mm(ZidZdphi).mm(v) )

                # sum up
                gradphi[k,l] = 0.5*(dlogQ_dphi + dyQiy_dphi)

        return grad1, grad2, grad3, gradphi.view(m*n,1), None, None, None, None, None
