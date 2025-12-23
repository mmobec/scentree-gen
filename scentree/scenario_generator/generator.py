import copy

class ScenarioGenerator:
    def get_best_model(self, X_train, y_train):
        pass
        # Crida estimator
        # Retorna una instancia del millor model ja entrenat 

    def prepare_data(self, X_train, y_train):
        return copy.copy(X_train), copy.copy(y_train)

    def get_predictions(self, X_train, y_train, X_test):
        # Manipular les dades
        X_train_manipulated, y_train_manipulated = self.prepare_data(X_train, y_train)

        best_model = self.get_best_model(X_train, y_train)

        #return best_model.predict(X_test)
        return None