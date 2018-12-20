import torch.nn as nn
import torch

class Simpsons(nn.Module):
    def __init__(self):
        super(Simpsons, self).__init__()

    def _quad_simpsons_mem(self, f, a, fa, b, fb):
        """Evaluates the Simpson's Rule, also returning m and f(m) to reuse"""
        m = torch.as_tensor((a+b)/2, dtype=torch.double)
        fm = f(m)
        return m, fm, abs(b-a) / 6 * (fa + 4 * fm + fb)

    def _quad_asr(self, f, a, fa, b, fb, eps, whole, m, fm):
        """
        Efficient recursive implementation of adaptive Simpson's rule.
        Function values at the start, middle, end of the intervals are retained.
        """
        lm, flm, left  = self._quad_simpsons_mem(f, a, fa, m, fm)
        rm, frm, right = self._quad_simpsons_mem(f, m, fm, b, fb)
        delta = left + right - whole
        if abs(delta) <= 15 * eps:
            return left + right + delta / 15
        return self._quad_asr(f, a, fa, m, fm, eps/2, left , lm, flm) +\
            self._quad_asr(f, m, fm, b, fb, eps/2, right, rm, frm)

    def _simps_adaptive(self,f, a, b, eps):
        """Integrate f from a to b using Adaptive Simpson's Rule with max error of eps."""
        fa, fb = f(a), f(b)
        m, fm, whole = self._quad_simpsons_mem(f, a, fa, b, fb)
        return self._quad_asr(f, a, fa, b, fb, eps, whole, m, fm)

    def forward(self, method, a, b, eps):
        # n must be a multiple of 2

        return self._simps_adaptive(method, torch.as_tensor(a), torch.as_tensor(b), eps)














