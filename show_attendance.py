import pandas as pd
from glob import glob
import os
import tkinter
import csv
import tkinter as tk
from tkinter import *
   
def subjectchoose(text_to_speech):
    def calculate_attendance():
        Subject = tx.get()
        if not Subject:
            t = 'Please enter the subject name.'
            text_to_speech(t)
            return

        # Change directory to the subject's attendance folder
        attendance_folder = f"Attendance\\{Subject}"
        if not os.path.exists(attendance_folder):
            t = f"Attendance folder for subject '{Subject}' does not exist."
            text_to_speech(t)
            return
        # os.chdir(attendance_folder)

        # Find all CSV files in the folder
        filenames = glob(
            f"Attendance\\{Subject}\\{Subject}*.csv"
        )
        if not filenames:
            t = "No attendance files found for this subject."
            text_to_speech(t)
            return

        # Read CSV files
        df_list = [pd.read_csv(f) for f in filenames]
        if len(df_list) == 1:
            newdf = df_list[0]
        else:
            # Merge multiple DataFrames
            newdf = df_list[0]
            for i in range(1, len(df_list)):
                newdf = newdf.merge(df_list[i], how="outer")

        newdf.fillna(0, inplace=True)

        # Calculate attendance percentage
        newdf["Attendance"] = (newdf.iloc[:, 2:].mean(axis=1) * 100).round().astype(int).astype(str) + '%'

        attendance_file_path = os.path.join(attendance_folder, "attendance.csv")
        newdf.to_csv(attendance_file_path, index=False)

        # Display attendance in a new Tkinter window
        root = tkinter.Tk()
        root.title(f"Attendance of {Subject}")
        root.configure(background="black")
        cs = f"Attendance\\{Subject}\\attendance.csv"
        with open(cs) as file:
            reader = csv.reader(file)
            for r, row in enumerate(reader):
                for c, value in enumerate(row):
                    label = tkinter.Label(
                        root,
                        width=10,
                        height=1,
                        fg="yellow",
                        font=("times", 15, " bold "),
                        bg="black",
                        text=value,
                        relief=tkinter.RIDGE,
                    )
                    label.grid(row=r, column=c)

        root.mainloop()

    # Main Tkinter window for subject selection
    subject = Tk()
    subject.title("Subject...")
    subject.geometry("580x320")
    subject.resizable(0, 0)
    subject.configure(background="black")

    titl = tk.Label(subject, bg="black", relief=RIDGE, bd=10, font=("arial", 30))
    titl.pack(fill=X)

    titl = tk.Label(
        subject,
        text="Which Subject of Attendance?",
        bg="black",
        fg="green",
        font=("arial", 25),
    )
    titl.place(x=100, y=12)

    # Function to check attendance sheets for the subject
    def Attf():
        sub = tx.get()
        if sub == "":
            t = "Please enter the subject name!!!"
            text_to_speech(t)
        else:
            os.startfile(f"Attendance\\{sub}")

    attf = tk.Button(
        subject,
        text="Check Sheets",
        command=Attf,
        bd=7,
        font=("times new roman", 15),
        bg="black",
        fg="yellow",
        height=2,
        width=10,
        relief=RIDGE,
    )
    attf.place(x=360, y=170)

    sub = tk.Label(
        subject,
        text="Enter Subject",
        width=10,
        height=2,
        bg="black",
        fg="yellow",
        bd=5,
        relief=RIDGE,
        font=("times new roman", 15),
    )
    sub.place(x=50, y=100)

    tx = tk.Entry(
        subject,
        width=15,
        bd=5,
        bg="black",
        fg="yellow",
        relief=RIDGE,
        font=("times", 30, "bold"),
    )
    tx.place(x=190, y=100)

    fill_a = tk.Button(
        subject,
        text="View Attendance",
        command=calculate_attendance,
        bd=7,
        font=("times new roman", 15),
        bg="black",
        fg="yellow",
        height=2,
        width=12,
        relief=RIDGE,
    )
    fill_a.place(x=195, y=170)

    subject.mainloop()