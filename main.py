import voice_module as vm
import views as v
from sklearn.externals import joblib
import random

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
            char = raw_input('\nIs this prediction correct? (Y/N)').upper()
            if char == 'N':
                v.select_em()
                em = int(raw_input("Choose your option: "))
                while (em < 1) and (em > 7):
                    em = int(raw_input("Choose your option: "))

                if em == 1:
                    newpath = 'AudioData/Added/a'
                elif em == 2:
                    newpath = 'AudioData/Added/d'
                elif em == 3:
                    newpath = 'AudioData/Added/f'
                elif em == 4:
                    newpath = 'AudioData/Added/h'
                elif em == 5:
                    newpath = 'AudioData/Added/n'
                elif em == 6:
                    newpath = 'AudioData/Added/sa'
                elif em == 7:
                    newpath = 'AudioData/Added/su'

                newpath += str(random.randrange(1, 1000, 1))+'.wav'

                wf = open('output.wav', 'rb')
                wf1 = open(newpath, 'wb')
                tmp = wf.read()
                wf1.write(tmp)
                wf.close()
                wf1.close()

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
