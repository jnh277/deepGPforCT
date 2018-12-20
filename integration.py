import torch.nn as nn
import torch

class Simpsons(nn.Module):
    def __init__(self, fcount_out=False, fcount_max=None, hmin=None):
        super(Simpsons, self).__init__()
        self.fcount_out = fcount_out
        self.count = fcount_out or fcount_max is not None
        if self.count:
            self.fcount = 0
        if fcount_max is not None:
            self.fcount_max = fcount_max
        else:
            self.fcount_max = float('inf')
        self.hmin = hmin


    def _quad_simpsons_mem(self, f, a, fa, b, fb):
        """Evaluates the Simpson's Rule, also returning m and f(m) to reuse"""
        m = torch.as_tensor((a+b)/2, dtype=torch.float32)
        fm = f(m)
        if self.count:
            self.fcount += 1
        return m, fm, abs(b-a) / 6.0 * (fa + 4.0 * fm + fb)

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
        fa, fb = f(a), f(b)
        if self.count:
            self.fcount += 2
        m, fm, whole = self._quad_simpsons_mem(f, a, fa, b, fb)
        return self._quad_asr(f, a, fa, b, fb, eps, whole, m, fm)

    def forward(self, method, a, b, eps):
        # n must be a multiple of 2
        at = torch.as_tensor(a, dtype=torch.float32)
        bt = torch.as_tensor(b, dtype=torch.float32)
        if self.hmin is None:
            self.hmin = abs(bt-at)*1e-10
        if self.count:
            self.fcount = 0
            result = self._simps_adaptive(method, at, bt, eps)
            out = (result, self.fcount)
        else:
            out = self._simps_adaptive(method, at, bt, eps)
        return out














