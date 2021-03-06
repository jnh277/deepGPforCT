import torch.nn as nn
import torch
import numpy as np

class Simpsons(nn.Module):
    def __init__(self, fcount_out=False, fcount_max=None, hmin=None):
        super(Simpsons, self).__init__()
        self.fcount_out = fcount_out
        self.count = fcount_out or fcount_max is not None
        self.fcount = 0
        # if self.count:
        #     self.fcount = 0
        if fcount_max is not None:
            self.fcount_max = fcount_max
        else:
            self.fcount_max = float('inf')
        self.hmin = hmin


    def _quad_simpsons_mem(self, f, a, fa, b, fb):
        """Evaluates the Simpson's Rule, also returning m and f(m) to reuse"""
        m = (a+b)/2.0
        fm = f(torch.as_tensor(m))
        if self.count:
            self.fcount += 1
        return m, fm, (b-a) / 6.0 * (fa + 4.0 * fm + fb)

    def _quad_asr(self, f, a, fa, b, fb, eps, whole, m, fm):
        """
        Efficient recursive implementation of adaptive Simpson's rule.
        Function values at the start, middle, end of the intervals are retained.
        """
        lm, flm, left = self._quad_simpsons_mem(f, a, fa, m, fm)
        rm, frm, right = self._quad_simpsons_mem(f, m, fm, b, fb)
        delta = left + right - whole
        if abs(delta) <= 15.0 * eps or self.fcount > self.fcount_max or abs(a-m) < self.hmin:
            result = left + right + delta / 15.0
        else:
            result = self._quad_asr(f, a, fa, m, fm, eps/2.0, left , lm, flm) +\
                self._quad_asr(f, m, fm, b, fb, eps/2.0, right, rm, frm)
        return result

    def _simps_adaptive(self, f, a, b, eps):
        """Integrate f from a to b using Adaptive Simpson's Rule with max error of eps."""
        fa, fb = f(torch.as_tensor(a)), f(torch.as_tensor(b))
        if self.count:
            self.fcount += 2
        m, fm, whole = self._quad_simpsons_mem(f, a, fa, b, fb)
        return self._quad_asr(f, a, fa, b, fb, eps, whole, m, fm)

    def forward(self, method, a, b, eps):
        # at = np.empty(1, dtype=np.float32)
        # bt = np.empty(1, dtype=np.float32)
        # at[0] = a
        # bt[0] = b
        at = a
        bt = b
        if self.hmin is None:
            self.hmin = abs(bt-at)*1e-6
        if self.count:
            self.fcount = 0
            result = self._simps_adaptive(method, at, bt, eps)
            if self.fcount_out:
                out = (result, self.fcount)
            else:
                out = result
        else:
            out = self._simps_adaptive(method, at, bt, eps)
        return out


class Simpsons2D(nn.Module):
    def __init__(self, fcount_out=False, fcount_max=None, hmin=None):
        super(Simpsons2D, self).__init__()
        self.fcount_out = fcount_out
        self.count = fcount_out or fcount_max is not None
        if self.count:
            self.fcount = 0
        if fcount_max is not None:
            self.fcount_max = fcount_max
        else:
            self.fcount_max = float('inf')
        self.hmin = hmin

    def _quad_simpsons2D_mem(self, f, x, y, fp):
        """Evaluates the 2D Simpson's Rule, also returning m and f(m) to reuse"""
        # with torch.no_grad():
        mx = (x[2]+x[0])/2.0

        x[1] = mx
        x[3] = x[0]
        x[4] = mx
        x[5] = x[2]
        x[7] = mx

        my = (y[6]+y[0])/2.0
        y[1] = y[0]
        y[3] = my
        y[4] = my
        y[5] = my
        y[7] = y[8]

        for i in [1, 3, 4, 5, 7]:
            fp[i] = f(torch.as_tensor(x[i]), torch.as_tensor(y[i]))

        if self.count:
            self.fcount += 5

        hx = (x[2] - x[0])
        hy = (y[6] - y[0])
        result = hx*hy/12.0*(-fp[0]+4.0*fp[1]-fp[2]+4.0*fp[3]+4.0*fp[5]-fp[6]+4.0*fp[7]-fp[8])

        return x, y, fp, result

    def _q(self, v, q):
        if torch.is_tensor(v):
            vo = v.clone()
        else:
            vo = v.copy()
        if q is 0:
            vo[6] = v[3]
            vo[2] = v[1]
            vo[8] = v[4]
            # vo[5] = v[9]
            # vo[7] = v[10]
        elif q is 1:
            vo[8] = v[5]
            vo[6] = v[4]
            vo[0] = v[1]
            # vo[3] = v[9]
            # vo[7] = v[11]
        elif q is 2:
            vo[8] = v[7]
            vo[2] = v[4]
            vo[0] = v[3]
        elif q is 3:
            vo[0] = v[4]
            vo[6] = v[7]
            vo[2] = v[5]
        return vo

    def _quad2D_asr(self, f, x, y, fp, eps, whole):
        """
        Efficient recursive implementation of adaptive Simpson's rule.
        Function values at the start, middle, end of the intervals are retained.
        """
        # x[9] = x[1]; y[9] = (y[1] + y[4])/2.0       # avoid calculating these twice
        # x[10] = (x[4] + x[3])/2.0; y[10] = y[3]
        # x[11] = (x[5] + x[4])/2.0; y[11] = y[3]
        # x[12] = x[4]; y[12] = (y[1] + y[4])/2.0
        # for i in [9,10,11,12]:
        #     fp[i] = f(x[i], y[i])

        xq0, yq0, fq0, q0 = self._quad_simpsons2D_mem(f, self._q(x, 0), self._q(y, 0), self._q(fp, 0))
        xq1, yq1, fq1, q1 = self._quad_simpsons2D_mem(f, self._q(x, 1), self._q(y, 1), self._q(fp, 1))
        xq2, yq2, fq2, q2 = self._quad_simpsons2D_mem(f, self._q(x, 2), self._q(y, 2), self._q(fp, 2))
        xq3, yq3, fq3, q3 = self._quad_simpsons2D_mem(f, self._q(x, 3), self._q(y, 3), self._q(fp, 3))
        sum_q = q0 + q1 + q2 + q3
        # sum_q.backward()
        # print('success sum_q')
        delta = sum_q - whole
        if abs(delta) <= 15.0 * eps or self.fcount > self.fcount_max or min(abs(xq0[1]-xq0[0]), abs(yq0[3]-yq0[0])) < self.hmin:
            result = sum_q + delta / 15.0
        else:
            result = self._quad2D_asr(f, xq0, yq0, fq0, eps/4.0, q0) + \
                 self._quad2D_asr(f, xq1, yq1, fq1, eps / 4.0, q1) + \
                 self._quad2D_asr(f, xq2, yq2, fq2, eps / 4.0, q2) + \
                 self._quad2D_asr(f, xq3, yq3, fq3, eps / 4.0, q0)
        return result

    def _simps2D_adaptive(self, f, xa, xb, ya, yb, eps):
        """Integrate f from a to b using Adaptive Simpson's Rule with max error of eps."""
        fp = torch.Tensor(13)
        x = np.empty(13, dtype=np.float32)
        y = np.empty(13, dtype=np.float32)
        # x = torch.Tensor(13, requires_grad=False)
        # y = torch.Tensor(13, requires_grad=False)
        x[0] = xa; y[0] = ya
        x[2] = xb; y[2] = ya
        x[6] = xa; y[6] = yb
        x[8] = xb; y[8] = yb

        for i in [0, 2, 6, 8]:
            fp[i] = f(torch.as_tensor(x[i]), torch.as_tensor(y[i]))

        if self.count:
            self.fcount += 4

        x, y, fp, whole = self._quad_simpsons2D_mem(f, x, y, fp)
        return self._quad2D_asr(f, x, y, fp, eps, whole)

    def forward(self, method, xa, xb, ya, yb, eps):
        if self.hmin is None:
            self.hmin = min(abs(xb-xa),(yb-ya))*1e-6
        if self.count:
            self.fcount = 0
            result = self._simps2D_adaptive(method, xa, xb, ya, yb, eps)
            if self.fcount_out:
                out = (result, self.fcount)
            else:
                out = result
        else:
            out = self._simps2D_adaptive(method, xa, xb, ya, yb, eps)
        return out









