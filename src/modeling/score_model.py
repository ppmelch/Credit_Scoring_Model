from src.utils.utils import np 

class CreditScoreModel:

    def __init__(self, coef, intercept, features):

        self.coef = coef
        self.intercept = intercept
        self.features = features
        self.t1 = 569 
        self.t2 = 645 
        
    def score(self, X):

        Xv = np.asarray(X)

        z = Xv @ self.coef + self.intercept

        return z

    def credit_score(self, X):

        z = self.score(X)
        
        z = -z

        z_min = z.min()
        z_max = z.max()

        score_norm = (z - z_min) / (z_max - z_min)

        score_min = 300
        score_max = 850

        score = score_norm * (score_max - score_min) + score_min

        return score.astype(int)
    
    def predict(self, X):

        scores = self.credit_score(X)

        return np.array([self.classify(s) for s in scores])
    
    def classify(self, score):

        if score < self.t1:
            return 1   # Standard

        elif score < self.t2:
            return 0   # Poor

        else:
            return 2   # Good
            
