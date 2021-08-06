import tkinter as tk


def user_input():
    root = tk.Tk()
    root.title('BCI Simulator Configuration')
    root.geometry('500x500')

    frame = tk.LabelFrame(root, text='Normalization mode', padx=5, pady=5)
    frame.pack(anchor='w', padx=10, pady=10)

    # Define user parameters and set default values
    normalize_mode = tk.IntVar()
    normalize_mode.set(1)
    game_mode = tk.StringVar()
    game_mode.set("CP")
    leadfield_mode = tk.StringVar()
    leadfield_mode.set("brainstorm")
    trial_length = tk.IntVar()
    trial_length.set(1800)  # not in ms but frames
    trial_count = tk.IntVar()
    trial_count.set(1)

    velocity_gain = tk.DoubleVar()
    velocity_gain.set(1 / 10)
    max_cursor_velocity = tk.DoubleVar()
    max_cursor_velocity.set(15)
    max_target_velocity = tk.DoubleVar()
    max_target_velocity.set(15)
    visualization_mode = tk.BooleanVar()
    visualization_mode.set(False)
    summary_visualization_mode = tk.BooleanVar()
    summary_visualization_mode.set(False)
    experiment_mode = tk.BooleanVar()
    experiment_mode.set(True)

    R1 = tk.Radiobutton(frame, text="Normalize output control velocity", variable=normalize_mode, value=0,
                        anchor='w', state="active")  # , command=print_selection)
    R1.pack(anchor='w')

    R2 = tk.Radiobutton(frame, text="Normalize estimated power", variable=normalize_mode, value=1,
                        anchor='w', state="active")  # , command=print_selection)
    R2.pack(anchor='w')

    frame2 = tk.LabelFrame(root, text='Experiment mode', padx=5, pady=5)
    frame2.pack(anchor='w', padx=10, pady=5)

    R1 = tk.Radiobutton(frame2, text="Continuous pursuit", variable=game_mode, value="CP",
                        anchor='w', state="active")  # , command=print_selection)

    R2 = tk.Radiobutton(frame2, text="Discrete 4-directional", variable=game_mode, value="DT_rect",
                        anchor='w', state="active")  # , command=print_selection)

    RT = tk.Radiobutton(frame2, text="Discrete circular", variable=game_mode, value="DT_circ",
                        anchor='w', state="active")

    tk.Label(frame2, text="Trial length").grid(row=1)
    tk.Label(frame2, text="Number of trials").grid(row=2)

    e3 = tk.Entry(frame2, textvariable=trial_length)
    e5 = tk.Entry(frame2, textvariable=trial_count)
    e3.grid(row=1, column=1)
    e5.grid(row=2, column=1)
    R1.grid(row=0, column=0)
    R2.grid(row=0, column=1)
    RT.grid(row=0, column=2)

    frame3 = tk.LabelFrame(root, text='Leadfield registration', padx=5, pady=5)
    frame3.pack(anchor='w', padx=10, pady=5)

    R1 = tk.Radiobutton(frame3, text="FSAverage+Brainnetome atlas(recommended)", variable=leadfield_mode,
                        value="brainstorm", anchor='w')  # , command=print_selection)
    R1.pack(anchor='w')

    R2 = tk.Radiobutton(frame3, text="New York Head+xjview atlas", variable=leadfield_mode, value="sereega",
                        anchor='w')  # , command=print_selection)
    R2.pack(anchor='w')

    frame4 = tk.LabelFrame(root, text="Cursor parameters", padx=5, pady=5)
    tk.Label(frame4, text="Input velocity gain").grid(row=0)
    e6 = tk.Entry(frame4, textvariable=velocity_gain)
    e6.grid(row=0, column=1)
    frame4.pack(anchor='w', padx=10, pady=5)

    tk.Label(frame4, text="Maximum cursor velocity").grid(row=1)
    e7 = tk.Entry(frame4, textvariable=max_cursor_velocity)
    e7.grid(row=1, column=1)

    tk.Label(frame4, text="Maximum target velocity").grid(row=2)
    e8 = tk.Entry(frame4, textvariable=max_target_velocity)
    e8.grid(row=2, column=1)

    c1 = tk.Checkbutton(root, text='Enable real-time vector visualizations', variable=visualization_mode, onvalue=True,
                        offvalue=False, state="disabled")
    c1.pack(anchor='w', padx=10, pady=0)
    c2 = tk.Checkbutton(root, text='Enable summary visualizations', variable=summary_visualization_mode, onvalue=True,
                        offvalue=False, state="disabled")
    c2.pack(anchor='w', padx=10, pady=0)
    c3 = tk.Checkbutton(root, text='Override settings above and begin experiment protocol', variable=experiment_mode,
                        onvalue=True, offvalue=False, state="active")
    c3.pack(anchor='w', padx=10, pady=0)

    # label = tk.Label(root)
    # label.pack()
    root.mainloop()  # Call this to stop the menu selection phase

    # Convert tkinter user choice variables to actual variables
    parameters = {
        'normalize_mode': normalize_mode.get(),
        'game_mode': game_mode.get(),
        'leadfield_mode': leadfield_mode.get(),
        'trialLength': trial_length.get(),
        'trialCount': trial_count.get(),
        'velocity_gain': velocity_gain.get(),
        'visualization_mode': visualization_mode.get(),
        'summary_visualization_mode': summary_visualization_mode.get(),
        'experiment_mode': experiment_mode.get(),
        'max_cursor_velocity': max_cursor_velocity.get(),
        'max_target_velocity': max_target_velocity.get(),
        'target_physics_mode': "adjust_drag"
    }

    return parameters


def dev_input():
    """
    Settings used for EMBC 1-page paper submission.
    """

    # Define user parameters and set default values
    normalize_mode = 1
    game_mode = "CP"
    leadfield_mode = "brainstorm"
    trial_length = 1800
    trial_count = 27
    velocity_gain = 1 / 10
    max_cursor_velocity = 25  # gets used during calibration
    max_target_velocity = 0
    visualization_mode = False
    summary_visualization_mode = False
    experiment_mode = True
    target_physics_mode = "adjust_drag"  # or "adjust_distribution"

    # Convert tkinter user choice variables to actual variables
    parameters = {
        'normalize_mode': normalize_mode.get(),
        'game_mode': game_mode.get(),
        'leadfield_mode': leadfield_mode.get(),
        'trialLength': trial_length.get(),
        'trialCount': trial_count.get(),
        'velocity_gain': velocity_gain.get(),
        'visualization_mode': visualization_mode.get(),
        'summary_visualization_mode': summary_visualization_mode.get(),
        'experiment_mode': experiment_mode.get(),
        'max_cursor_velocity': max_cursor_velocity.get(),
        'max_target_velocity': max_target_velocity.get(),
        'target_physics_mode': target_physics_mode
    }

    return parameters
