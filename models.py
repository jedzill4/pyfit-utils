
import lmfit as lmf
import numpy as np


class EnzimaticModel(lmf.Model):
    r"""
    Enzimatic model for activation/inactivation of a process
    (Schoolfield et. al. 1980)
    input:
        x   : Data temperatures [K]
        rho : Development rate at 25 C [time ^ -1]
        dHA : Entalpy of activation [J mol ^-1]
        dHL : Low temperature inactivation entalpy [J mol^-1]
        dHH : High temperatue inactivation entalpy [J mol^-1]
        T12L : Temperature enzyme is 1/2 active and 1/2 low temperature inactive [K]
        T12H : Temperature enzime is 1/2 active and 1/2 high temperature active [K]
    output:
        rate

    """
    def __init__(self, independent_vars=['x'], prefix='', nan_policy='omit',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})

        r = 8.3144598 #[mol^-1  k^-1]
        rr = 1/r
        t298 = 1/298
        def enzimatic(x, rho, dHA, dHL, dHH, T12L, T12H):
            act = rho * (x*t298) * np.exp( (dHA*rr) * (t298 - 1/x) )
            high = np.exp( (dHH*rr) * (1/T12H - 1/x) ) if not np.any(np.isnan([dHH,T12H])) else 0
            low  = np.exp( (dHL*rr) * (1/T12L - 1/x) ) if not np.any(np.isnan([dHL,T12L])) else 0

            return act/(1+low+high)

        super(self.__class__, self).__init__(enzimatic, **kwargs)
        
        self.set_param_hint('rho', min=0, value=1)
        self.set_param_hint('T12L', min=273, max=300, value=285)
        self.set_param_hint('T12H', min=290, max=323, value=307)
        self.set_param_hint('dHA', min=0, value=11000)
        self.set_param_hint('dHH', min=0, value=76000)
        self.set_param_hint('dHL', min=0, value=18000)
        self.set_param_hint('ineq1', min=0, expr='dHH - dHA')
        self.set_param_hint('ineq2', max=0, expr='dHL - dHA')

    def guess(self, data, **kws):
        params = self.make_params()
        return params

class EnzimaticLowCutModel(lmf.Model):
    def __init__(self, independent_vars=['x'], prefix='', nan_policy='omit',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})

        r = 8.3144598 #[mol^-1  k^-1]
        rr = 1/r
        t298 = 1/298
        def enzimatic(x, rho, dHA, dHL, dHH, T12L, T12H):
            act = rho * (x*t298) * np.exp( (dHA*rr) * (t298 - 1/x) )
            high = np.exp( (dHH*rr) * (1/T12H - 1/x) ) if not np.any(np.isnan([dHH,T12H])) else 0
            low  = np.exp( (dHL*rr) * (1/T12L - 1/x) ) if not np.any(np.isnan([dHL,T12L])) else 0

            return act/(1+low+high)

        super(self.__class__, self).__init__(enzimatic, **kwargs)
        
        self.set_param_hint('rho', min=0, value=1)
        self.set_param_hint('T12L', min=273, max=300, value=285)
        self.set_param_hint('dHA', min=0, value=11000)
        self.set_param_hint('dHL', min=0, value=18000)
        self.set_param_hint('ineq2', max=0, expr='dHL - dHA')

        self.set_param_hint('T12H', value=np.nan,vary=False)
        self.set_param_hint('dHH',  value=np.nan,vary=False)

    def guess(self, data, **kws):
        params = self.make_params()
        return params
class EnzimaticHighCutModel(lmf.Model):
    def __init__(self, independent_vars=['x'], prefix='', nan_policy='omit',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})

        r = 8.3144598 #[mol^-1  k^-1]
        rr = 1/r
        t298 = 1/298
        def enzimatic(x, rho, dHA, dHL, dHH, T12L, T12H):
            act = rho * (x*t298) * np.exp( (dHA*rr) * (t298 - 1/x) )
            high = np.exp( (dHH*rr) * (1/T12H - 1/x) ) if not np.any(np.isnan([dHH,T12H])) else 0
            low  = np.exp( (dHL*rr) * (1/T12L - 1/x) ) if not np.any(np.isnan([dHL,T12L])) else 0

            return act/(1+low+high)

        super(self.__class__, self).__init__(enzimatic, **kwargs)
        

        self.set_param_hint('rho', min=0, value=1)
        self.set_param_hint('T12H', min=290, max=323, value=307)
        self.set_param_hint('dHA', min=0, value=11000)
        self.set_param_hint('dHL', min=0, value=18000)
        self.set_param_hint('ineq1', min=0, expr='dHH - dHA')
        
        self.set_param_hint('T12L', value=np.nan,vary=False)
        self.set_param_hint('dHL',  value=np.nan,vary=False)

    def guess(self, data, **kws):
        params = self.make_params()
        return params
class EnzimaticActivationModel(EnzimaticModel):
    def __init__(self, independent_vars=['x'], prefix='', nan_policy='omit',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})

        super(self.__class__.__name__, self).__init__(**kwargs)
        
        self.set_param_hint('T12L', value=np.nan,vary=False)
        self.set_param_hint('T12H', value=np.nan,vary=False)
        self.set_param_hint('dHH',  value=np.nan,vary=False)
        self.set_param_hint('dHL',  value=np.nan,vary=False)

class EnzimaticParentModel(lmf.Model):
    """
    Enzimatic model for activation/inactivation of a process
    (Parent and Tardieu 2012)
    input:
        x    : Data temperatures [K]
        rho  : Development rate at 25 C [time ^ -1]
        dHA  : Entalpy of activation [J mol ^-1]
        alpha: dHL/dHA rate
        T0   : dHL/dSL (inactivation low entalpy/inactivation low entropy) [K]
    output:
        rate
    """
    def __init__(self, independent_vars=['x'], prefix='', nan_policy='omit',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})

        def enzimatic_parent(x, rho,dHA, alpha, T0):
            P1 = rho * x * np.exp( -dHA*RR/x )
            P2 = 1 + np.exp( -alpha*(dHA*RR/x)*(1. - x/T0) ) 
            return P1/P2

        super(self.__class__.__name__, self).__init__(enzimatic_parent, **kwargs)
        
        self.set_param_hint('rho', min=0, value=1)
        self.set_param_hint('dHA', min=0, value=11000)
        self.set_param_hint('alpha', min=0, value=2)
        self.set_param_hint('Topt', min=273, max=323, value=305)

    def guess(self, data, **kws):
        params = self.make_params()
        return params

class BetaModel(lmf.Model):
    """
    Beta model for temperature responce for development
    (Yan and Hunt 1999)
    input:
        x    : Data temperatures [K]
        rmax : Max rate at optimum temperature
        Topt : optimum temperatura of development
        Tceil: ciling temerature where development ceases
    output:
        rate
    """
    def __init__(self, independent_vars=['x'], prefix='', nan_policy='omit',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})

        def beta(x, rmax, Topt, Tceil):
            const1 = 1./(Tceil - Topt)
            const2 = 1/Topt
            pow1 = Topt * const1
            return (Tceil - x) * (x*const2)**pow1

        super(self.__class__.__name__, self).__init__(beta, **kwargs)
        
        self.set_param_hint('rmax', min=0, value=1)
        self.set_param_hint('Topt', min=0, value=32)
        self.set_param_hint('Tceil', min=0, value=45)


    def guess(self, data, **kws):
        params = self.make_params()
        return params

#import scipy.stats.distributions.poisson.pmf as poisson_pdf
#class PoissonModel(lmf.Model):
#    """
#    Poisson distribution model
#    P(k|lambda) = lambda^k * exp(-lambda) / k!
#    model = amplitude*P(lambda)
#    input:
#        lambda      : rate
#        amplitude   : amplitude
#    """
#    def __init__(self, independent_vars=['x'], prefix='', nan_policy='omit',
#                 **kwargs):
#        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
#                       'independent_vars': independent_vars})
#
#        def poisson(x, amplitude=1, mu=1):
#            if np.any( x < 0 ):
#                raise ModelError("Data must be non negative")
#            k = floor(x)
##            return amplitude*np.exp(k*np.log(mu) - mu - np.log(factorial(k)) )
#            
#            return amplitude * poisson_pdf(k,mu) 
#            
#
#        super(self.__class__.__name__, self).__init__(poisson, **kwargs)
#
#        self.set_param_hint('amplitude', min=0, value=1)
#        self.set_param_hint('mu', min=0)
#
#    def guess(self, data, **kwargs):
#        params = self.make_params()
#
#        params[self.prefix+"amplitude"] = np.max(data)
#
#        x = self.independent_vars[0]
#        if x in kwargs:
#            params[self.prefix+"mu"] = np.mean(x)
#        else:
#            params[self.prefix+"mu"] = .5*len(data)
#
#        return params
