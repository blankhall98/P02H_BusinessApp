#librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#clase: Bussiness Intelligence App
class BIA:

    def __init__(self,inputs):
        self.inputs = inputs
        self.load_database()

        self.menu()

    #loads database information
    def load_database(self):
        self.data = pd.read_csv(self.inputs['data_route']+self.inputs['file_name'])

    #saves new information in database
    def save_data(self):
        self.data.to_csv(self.inputs['data_route']+self.inputs['file_name'],index=False)

    #add new observation to database
    def add_observation(self):
        print('\n')
        print('Answer the following questions:')
        sex = input('SEX (F/M) : ')
        age = input('AGE : ')
        amount = input('AMOUNT SPENT : ')
        print('\n')
        new_obs = [sex,age,amount]
        n = len(self.data)
        self.data.loc[n] = new_obs
        self.save_data()

    #visualize dabase
    def show_base(self):
        print('\n')
        print('--- CLIENTS DATABASE ---')
        print(self.data)
        print('\n')

    #analize clients
    def analize_data(self):
        plt.figure()
        plt.title('Relationship Between Custumers Age and Purchase')
        plt.ylabel('Amount Spent')
        plt.xlabel('Customer Age')
        plt.plot(self.data['AGE'],self.data['AMOUNT_SPENT'],'o')
        plt.show()

        
        amount = np.array(self.data['AMOUNT_SPENT'])
        #create column sex_code
        sex_code = []
        for sex in self.data['SEX']:
            if sex == 'M':
                sex_code.append(1)
            else:
                sex_code.append(0)

        db_model_cov = pd.DataFrame()
        db_model_cov['AGE'] = self.data['AGE']
        db_model_cov['SEX'] = sex_code

        self.model = LinearRegression()

        self.model.fit(db_model_cov,amount)

        self.coef = self.model.coef_
        self.intercept = self.model.intercept_
        self.score = self.model.score(db_model_cov,amount)

        print('\n'+'Linear Regression Analysis '+ '\n')
        print(f'Model Score : {self.score}')
        print(f'Coef: {self.coef} Intercept: {self.intercept}')
        print(f'Relationship: AMOUNT SPENT ~ {self.intercept} + {self.coef[0]}*AGE + {self.coef[1]}*MEN')
        print('\n')

    #create predictions
    def predict(self):
        method = input('Use standard-list (s) or input a value to predict (i): ')
        if method == 's':
            profile_age = [10,20,30,40,50,60,70,80,90]
            profile_sex = [0,1,0,1,0,1,0,1,0]
            test_db = pd.DataFrame()
            test_db['AGE'] = profile_age
            test_db['SEX'] = profile_sex

        
            pred = self.model.predict(test_db)
            
            print('\n')
            for i in range(len(pred)):
                print(f'Someone with {profile_age[i]} years and sex {profile_sex[i]} should spend ${pred[i]}.')
            print('\n')

        elif method == 'i':
            age = float(input('select age to predict: '))
            sex = int(input('Select sex, female (0) male (1): '))
            prediction  = self.intercept + age*self.coef[0] + sex*self.coef[1]
            print(f'Prediction = ${prediction}')
            print('\n')

    #MENU
    def menu(self):
        log_out = False

        while(log_out == False):
            print('\n')
            print('--- MENU ---')
            print('\n')

            #options to execute
            print('1. Add Observation')
            print('2. Show Base')
            print('3. Analize Data')
            print('4. Predict')

            #choose an option
            print('\n')
            option = input('Select the index of the action you wish to perform: ')
            print('\n')

            #trigger action
            if option == '1':
                self.add_observation()
            elif option == '2':
                self.show_base()
            elif option == '3':
                self.analize_data()
            elif option == '4':
                self.predict()
            
            #continue or log-out
            c_or_l = input('Continue (c) or log-out (out): ')
            if c_or_l == 'out':
                log_out = True

# App
parametros = {
    'data_route': './data/',
    'file_name': 'clients.csv'
}

def main():
    bss = BIA(parametros)

if __name__ == '__main__':
    main()
