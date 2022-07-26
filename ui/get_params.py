import tkinter as tk
from tkinter import filedialog
from tkinter import *


def browse_button():
    global folder_path
    filename = filedialog.askdirectory()
    folder_path.set(filename)
    print(filename)


def launcher(clearance_level):
    global folder_path
    if clearance_level == "dev":
        state = "active"
        entry_state = "normal"
    elif clearance_level == "subj":
        state = "disabled"
        entry_state = "disabled"
    
    root = tk.Tk()
    root.title('BCI Simulator Launcher')
    root.geometry('500x500')

    frame = tk.LabelFrame(root, text='Normalization mode', padx=5, pady=5)
    frame.pack(anchor='w', padx=10, pady=10)

    # Define user parameters and set default values
    normalize_mode = tk.IntVar()
    normalize_mode.set(0)
    game_mode = tk.StringVar()
    game_mode.set("DT_1D")
    leadfield_mode = tk.StringVar()
    leadfield_mode.set("brainstorm")
    trial_length = tk.IntVar()
    trial_length.set(1800)  # not in ms but frames. This is going to be only relevant for CP.
    trial_count = tk.IntVar()
    trial_count.set(24)  # NEW!
    max_cursor_velocity = tk.DoubleVar()
    max_cursor_velocity.set(5.5)
    max_target_velocity = tk.DoubleVar()
    max_target_velocity.set(5.5)
    folder_path = StringVar()
    experimental_param_array = tk.StringVar()
    experimental_param_array.set("BW90 BW120 BW5 BW60 BW30 CO1 CO2 CS150 CS200 CS250 CS300 CS350")
    # experimental_param_array.set("BW15")
    run_count = tk.IntVar()
    run_count.set(len(experimental_param_array.get().split(" ")))  # automatically detect number of runs. disable changing.

    R1 = tk.Radiobutton(frame, text="Normalize output control velocity", variable=normalize_mode, value=0,
                        anchor='w', state=state)  # , command=print_selection)
    R1.pack(anchor='w')

    R2 = tk.Radiobutton(frame, text="Normalize estimated power", variable=normalize_mode, value=1,
                        anchor='w', state=state)  # , command=print_selection)
    R2.pack(anchor='w')

    frame2 = tk.LabelFrame(root, text='Experiment mode', padx=5, pady=5)
    frame2.pack(anchor='w', padx=10, pady=5)

    R1 = tk.Radiobutton(frame2, text="CP", variable=game_mode, value="CP",
                        anchor='w', state=state)  # , command=print_selection)

    R1D = tk.Radiobutton(frame2, text="Discrete LR", variable=game_mode, value="DT_1D",
                         anchor='w', state=state)

    R2 = tk.Radiobutton(frame2, text="Discrete 2D", variable=game_mode, value="DT_2D",
                        anchor='w', state=state)  # , command=print_selection)

    RT = tk.Radiobutton(frame2, text="Discrete circular", variable=game_mode, value="DT_circ",
                        anchor='w', state=state)

    tk.Label(frame2, text="Trial length").grid(row=1, column=0)
    tk.Label(frame2, text="# Trials").grid(row=2, column=0)
    tk.Label(frame2, text="# Runs").grid(row=1,column=2)
    tk.Label(frame2, text="Param. Array").grid(row=2,column=2)

    e3 = tk.Entry(frame2, textvariable=trial_length, state=entry_state)
    e5 = tk.Entry(frame2, textvariable=trial_count, state=entry_state)
    e6 = tk.Entry(frame2, textvariable=run_count, state="disabled")
    e7 = tk.Entry(frame2, textvariable=experimental_param_array, state=entry_state)
    e3.grid(row=1, column=1)
    e5.grid(row=2, column=1)
    e6.grid(row=1, column=3)
    e7.grid(row=2, column=3)
    R1.grid(row=0, column=0)
    R1D.grid(row=0, column=1)
    R2.grid(row=0, column=2)
    RT.grid(row=0, column=3)

    frame3 = tk.LabelFrame(root, text='Leadfield registration', padx=5, pady=5)
    frame3.pack(anchor='w', padx=10, pady=5)

    R1 = tk.Radiobutton(frame3, text="FSAverage+Brainnetome atlas(recommended)", variable=leadfield_mode,
                        value="brainstorm", anchor='w', state=state)
    R1.pack(anchor='w')
    R2 = tk.Radiobutton(frame3, text="New York Head+xjview atlas", variable=leadfield_mode, value="sereega",
                        anchor='w', state=state)
    R2.pack(anchor='w')

    frame4 = tk.LabelFrame(root, text="Cursor parameters", padx=5, pady=5)
    frame4.pack(anchor='w', padx=10, pady=5)

    tk.Label(frame4, text="Maximum cursor velocity").grid(row=0)
    e7 = tk.Entry(frame4, textvariable=max_cursor_velocity, state=entry_state)
    e7.grid(row=0, column=1)
    tk.Label(frame4, text="Maximum target velocity").grid(row=1)
    e8 = tk.Entry(frame4, textvariable=max_target_velocity, state=entry_state)
    e8.grid(row=1, column=1)

    frame5 = tk.LabelFrame(root, text="Data storage", padx=5, pady=5)
    frame5.pack(anchor='w', padx=10, pady=5)
    lbl1 = Label(master=frame5, textvariable=folder_path)
    lbl1.grid(row=2, column=0)
    button2 = Button(frame5, text="Browse", command=browse_button)
    button2.grid(row=2, column=1)

    if clearance_level == "subj":
        frame6 = tk.LabelFrame(root, text="", padx=5, pady=5)
        frame6.pack(anchor='w', padx=10, pady=5)
        T = Text(frame6)
        T.pack()
        T.insert(tk.END, "SUBJECT MODE. Please select a folder that you can access, and then close this window to start the experiment.")

    root.mainloop()  # Call this to stop the menu selection phase

    # Convert tkinter user choice variables to actual variables
    parameters = {
        'normalize_mode': normalize_mode.get(),
        'game_mode': game_mode.get(),
        'leadfield_mode': leadfield_mode.get(),
        'trial_length': trial_length.get(),
        'trial_count': trial_count.get(),
        'max_cursor_velocity': max_cursor_velocity.get(),
        'max_target_velocity': max_target_velocity.get(),
        'target_physics_mode': "adjust_drag",
        'save_file_directory': folder_path,
        'experimental_param_array': experimental_param_array.get(),
        'run_count': run_count.get()
    }

    return parameters



