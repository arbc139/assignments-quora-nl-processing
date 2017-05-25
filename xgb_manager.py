
import xgboost as xgb

class XgbManager():
  def __init__(self, X_train, X_valid, X_test, y_train, y_valid):
    self.D_train = xgb.DMatrix(X_train, label=y_train)
    self.D_valid = xgb.DMatrix(X_valid, label=y_valid)
    self.D_test = xgb.DMatrix(X_test)
    self.booster = None
  
  def train(self, params, options, num_boost_round=400):
    # Default values
    options = {
      'early_stopping_rounds': 50 if not options['early_stopping_rounds'] else options['early_stopping_rounds'],
      'verbose_eval': 10 if not options['verbose_eval'] else options['verbose_eval'],
    }
    watchlist = [(self.D_train, 'train'), (self.D_valid, 'valid')]
    self.booster = xgb.train(
      params, self.D_train, num_boost_round, watchlist,
      early_stopping_rounds=options['early_stopping_rounds'],
      verbose_eval=options['verbose_eval'])
  
  def predict(self):
    if not self.booster:
      raise RuntimeError('XgbManager: Try predict before train!')
    return self.booster.predict(self.D_test)
  
  



