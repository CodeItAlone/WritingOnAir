# WritingOnAir
a computer-vision tool that lets users write or draw in the air using only their finger and a webcam. The script uses real-time video capture to track hand or fingertip movement and converts those movements into digital strokes on a virtual canvas.


WritingOnAir ✍️✨

A computer-vision project that lets users write or draw in the air using only their finger and a webcam.
The script tracks fingertip movements in real time and converts them into strokes on a virtual canvas.

🚀 Features

Air Writing / Drawing using fingertip tracking

Real-time hand/finger detection (OpenCV / MediaPipe)

Virtual canvas rendering

Continuous drawing as you move your finger

Simple gesture-based control (draw, move, clear – depending on your implementation)

Works with any standard webcam

🧠 How It Works

The webcam feed is captured frame by frame.

A hand-tracking module detects the fingertip position.

As the fingertip moves, its coordinates are mapped onto a virtual canvas.

Lines are drawn automatically between consecutive positions.

The output (camera feed + drawing) is displayed on screen.

The main.py script orchestrates the whole pipeline — camera setup, detection, drawing, and UI updates.

📁 Project Structure
WritingOnAir/
│── main.py              # Entry point of the application
│── README.md            # Documentation
│── LICENSE              # MIT license
│── .gitignore
│
├── Jarvis/              # (Your module folder – rename if needed)
├── testing/             # Extra tests or scripts
└── venv/                # Virtual environment (ignored by Git)

🛠️ Technologies Used

Python 3.x

OpenCV

MediaPipe (if you’re using it)

NumPy

📦 Installation

Make sure Python is installed.

git clone https://github.com/CodeItAlone/WritingOnAIR.git
cd WritingOnAIR


Install dependencies:

pip install -r requirements.txt


(If you don’t have a requirements.txt, I can generate one for you.)

▶️ Run the Project
python main.py

📝 Future Improvements

Add color selection

Add eraser mode

Add gesture recognition

Add UI overlay buttons

Save drawing as image

📄 License

This project is licensed under the MIT License.

🤝 Contributing

Pull requests are welcome.
For major changes, open an issue first to discuss what you’d like to improve.
