from enum import Enum
import tab_views


class Regression(Enum):
    Linear = 0
    MultipleLinear = 1
    SupportVector = 2
    RandomForest = 3
    NeuralNetwork = 4

    def to_string(self) -> str:
        if self == Regression.Linear:
            return 'Linear Regression'
        elif self == Regression.MultipleLinear:
            return 'Multiple Linear Regression'
        elif self == Regression.SupportVector:
            return 'Support Vector Regression'
        elif self == Regression.RandomForest:
            return 'Random Forest Regression'
        elif self == Regression.NeuralNetwork:
            return 'Neural Network Regression'

    def tab_type(self):
        if self == Regression.Linear:
            return tab_views.LRTabView.LRTabView
        elif self == Regression.MultipleLinear:
            return tab_views.MLRTabView.MLRTabView
        elif self == Regression.SupportVector:
            return tab_views.SVRTabView.SVRTabView
        elif self == Regression.RandomForest:
            return tab_views.RFRTabView.RFRTabView
        elif self == Regression.NeuralNetwork:
            return tab_views.NNRTabView.NNRTabView
