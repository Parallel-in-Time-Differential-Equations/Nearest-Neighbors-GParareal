class Normalize():
    def __init__(self, mn, mx, norm_type=None):
        self.mn = mn
        self.mx = mx

        if norm_type is None:
            norm_type = 'identity'

        if norm_type.lower() not in ['identity', '-11']:
            raise NotImplementedError('Only identity and -11 are implemented')
        
        self.norm_type = norm_type.lower()
        
    def fit(self, x):
        mn, mx = self.mn, self.mx
        if self.norm_type == '-11':
            return 2*(x-mn)/(mx-mn)-1
        else:
            return x

    def inverse(self, x):
        mn, mx = self.mn, self.mx
        if self.norm_type == '-11':
            return (x+1)/2 * (mx-mn) + mn
        else:
            return x

    def get_scale(self):
        mn, mx = self.mn, self.mx
        if self.norm_type == '-11':
            return 2/(mx-mn)
        else:
            return 1