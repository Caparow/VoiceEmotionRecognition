import voice_module as vm
import views as v
from sklearn.externals import joblib

if __name__ == "__main__":
    exit_flag = 0
    print('Loading a model...')
    clf = joblib.load('model.pkl')
    v.draw_menu()
    option = int(raw_input("Choose your option: "))

    while option != 5:
        if option == 1:
            v.cls()
            vm.MakingPrediction('output.wav', clf)
        elif option == 2:
            v.cls()
            vm.TrainModel('model.pkl', 0)
            clf = joblib.load('model.pkl')
        elif option == 3:
            v.cls()
            vm.TestModel('model.pkl')
            clf = joblib.load('model.pkl')
        elif option == 4:
            v.cls()
            vm.TrainModel('model.pkl', 1)
            clf = joblib.load('model.pkl')

        v.goBack()
        option = int(raw_input("Choose your option: "))
