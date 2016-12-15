import os


def cls():
    os.system('cls' if os.name == 'nt' else 'clear')


def draw_menu():
    cls()
    print("VOICE EMOTION RECOGNITION:\n")
    print("1. Predict emotion by voice")
    print("2. Train model")
    print("3. Test model on training files and train model")
    print("4. Extract features from DataSet and train model")
    print("5. Exit")

def select_em():
    cls()
    print("SELECT YOUR EMOTION:\n")
    print("1. Angry")
    print("2. Disgusting")
    print("3. Fear")
    print("4. Happiness")
    print("5. Neutral")
    print("6. Sadness")
    print("7. Surprise")

def goBack():
    raw_input("\n\nPress enter to go back.")
    draw_menu()