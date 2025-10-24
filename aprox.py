import pandas as pd

class vinho :
    def __init__(self, fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol):
        self.fixed_acidity = fixed_acidity
        self.volatile_acidity = volatile_acidity
        self.citric_acid = citric_acid
        self.residual_sugar = residual_sugar
        self.chlorides = chlorides
        self.free_sulfur_dioxide = free_sulfur_dioxide
        self.total_sulfur_dioxide = total_sulfur_dioxide
        self.density = density
        self.pH = pH
        self.sulphates = sulphates
        self.alcohol = alcohol
        
df = pd.read_csv('wine_quality_merged_quality.csv')
def fitness(x):
    return 55.76+0.0677*(x.fixed_acidity)-1.3279*(x.volatile_acidity)-0.1097*(x.citric_acid)+0.0436*(x.residual_sugar)-0.4837*(x.chlorides)+0.0060*(x.free_sulfur_dioxide)-0.0025*(x.total_sulfur_dioxide)-54.97*(x.density)+0.4393*(x.pH)+0.7683*(x.sulphates)+0.2670*(x.alcohol)

accuracy = 1
vinhos = vinho[
    
]
for index, row in df.iterrows():
    
    fit = fitness(v)
    pred = row.quality
    accuracy = (accuracy + (1 - abs(fit - pred) / 10)) / 2

    