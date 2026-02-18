from datetime import datetime
import pandas as pd
import numpy as np
import json

def calculate_d_score():
    with open('./input/class.txt', 'r') as f:
        class_name = f.readline().strip()

    with open('./assets/config.json') as f:
        class_config = json.load(f)

    L = class_config[class_name]["L"]
    A = class_config[class_name]["A"]
    B = class_config[class_name]["B"]
    min_distance_prev = class_config[class_name]["Min Distance"]
    max_distance_prev = class_config[class_name]["Max Distance"]

    df = pd.read_csv("./output/results.csv")
    df.insert(0, 'Date', datetime.now().date().isoformat())
    df.insert(1, 'Time', datetime.now().time().strftime('%H:%M:%S'))
    df.insert(2, 'Class', class_name)
    df['D_Score'] = np.sqrt((df['L_calib']-L)**2+(df['A_calib']-A)**2+(df['B_calib']-B)**2)
    min_distance_new = df['D_Score'].min()
    max_distance_new = df['D_Score'].max()
    if min_distance_prev != -1 and min_distance_prev < min_distance_new:
        min_distance_new = min_distance_prev
    if max_distance_prev != -1 and max_distance_prev > max_distance_new:
        max_distance_new = max_distance_prev

    if min_distance_prev != min_distance_new or max_distance_prev != max_distance_new:
        class_config[class_name]["Min Distance"] = min_distance_new
        class_config[class_name]["Max Distance"] = max_distance_new
        with open('./assets/config.json', 'w') as f:
            json.dump(class_config, f, indent=4)

    df['D_Score'] = 1-(df['D_Score']-min_distance_new)/(max_distance_new-min_distance_new)
    cols_to_round = ['L_initial', 'A_initial', 'B_initial', 'L_calib', 'A_calib', 'B_calib']
    df[cols_to_round] = df[cols_to_round].round(2)
    v = df['D_Score'].clip(0, 1)
    df['D_Score'] = v.round(4)  # continuous 0-1
    df['D_Score_Category'] = (np.floor(v * 9).astype(int) + 1).clip(upper=9)
    
    # v = df['D_Score'].clip(0, 1)
    # df['D_Score'] = (np.floor(v * 9).astype(int) + 1).clip(upper=9)

    notes = ""
    if min_distance_prev != min_distance_new:
        notes += f"The minimum LAB Euclidean distance has been updated from {min_distance_prev:.4f} to {min_distance_new:.4f}. "
    if max_distance_prev != max_distance_new:
        notes += f"The maximum LAB Euclidean distance has been updated from {max_distance_prev:.4f} to {max_distance_new:.4f}."

    df['Notes'] = ""
    df.at[0, 'Notes'] = notes

    df.to_csv("./output/results.csv", index=False)
