import os
import shutil

PARTICIPANT_LIST = [
]

gestures = {
    "fist":     1,
    "palm":     2,
    "point":    3,
    "three":    4,
    "two":      5
}


def move():
    participant_counter = 1
    for participant in PARTICIPANT_LIST:
        print(participant)
        scenario_counter = 1
        for scenario in participant:
            new_dir = f"p{participant_counter}_s{scenario_counter}"
            try:
                os.mkdir(new_dir)
            except OSError:
                print("Already created")

            for file in os.listdir("OLD/" + scenario):
                print(file)
                gesture = file.split("_")[0]
                shutil.copyfile(f"OLD/{scenario}/{file}", f"{new_dir}/g{gestures[gesture]}.emg")
            scenario_counter += 1
        participant_counter += 1


if __name__ == '__main__':
    move()
