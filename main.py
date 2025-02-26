import engine
import os
import sys
from tkinter import filedialog  # For browsing files
from ultralytics import YOLO


def load_menu():
    print("[1] Read Image with Baybayin Text")
    print("[2] Load all images in a directory and read")
    print("[3] Exit")
    command = input("Enter Command: ")
    
    return command


if __name__ == "__main__":
    print("Welcome to BaybayinOCR!")
    print("Made by Rio Almeria\n")
    
    print("NOTE: This can only read single word blocks of Baybayin Text.")
    print("Check the paper out at:")
    print("https://drive.google.com/file/d/1z3gomVBH_nBzAmOSufL_x4MN2uQHsGmm/")
    
    model = None
    
    try:
        print("\nPlease load the BaybayinOCR model.")
        input("Press ENTER to continue.")
        model_dir = filedialog.askopenfilename()
        model = YOLO(model_dir)
        print("Model loaded.\n")
    except Exception as e:
        print("Error loading the model.")
        sys.exit(1)
    
    while True:
        command = int(load_menu())
        
        if command == 1: 
            file = filedialog.askopenfilename()
            print(f"Loaded image from {file}")
            engine.read_image(file, model)
            
            
        elif command == 2:
            dir = filedialog.askdirectory()
            for file in os.listdir(dir):
                engine.read_image(os.path.join(dir, file), model)
        elif command == 3:
            print("\nThanks for using this program!")
            print("You can email me at rioalmeria@gmail.com")
            sys.exit(0)
        else:
            print("\nInvalid Command.\n")
            