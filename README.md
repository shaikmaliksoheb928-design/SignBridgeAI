# SignBridge AI — ISL + ASL Prototype

A browser-based hand-sign learning and recognition prototype. It uses MediaPipe Hand Landmarker to extract 21 hand landmarks, then a simple k-nearest-neighbour (k-NN) classifier learns labels from examples recorded by the user.

## Important scope and accuracy note

This is a **prototype**, not a complete ISL/ASL translator. ISL and ASL are distinct languages. This app keeps separate training libraries for each, but it does not ship with a validated sign-language dataset. You teach it static, one-hand poses by recording examples.

It does not currently recognize:
- every sign in ISL or ASL;
- moving/dynamic signs;
- two-handed signs;
- facial expressions or non-manual grammar;
- full sentence grammar or context.

Predictions are best treated as a demo / assistive prototype and must not be relied on for important communication. For a serious product, work with Deaf signers and qualified interpreters, use a consented and representative dataset, and evaluate accuracy across different people, lighting, cameras and signing styles.

## Requirements

- Windows, macOS or Linux
- Python 3.9+ (only used to serve the website locally)
- A webcam
- Modern Chrome or Edge recommended
- Internet connection for the first load (MediaPipe library and model are loaded from CDNs)

## Run on Windows

1. Download and extract `SignBridgeAI.zip`.
2. Open the extracted `SignBridgeAI` folder.
3. Click the folder address bar, type `cmd`, and press Enter.
4. Run:

   ```bash
   python -m http.server 8000
   ```

5. Open `http://localhost:8000` in Chrome or Edge.
6. Click **Start camera** and allow camera permission.

If `python` is not recognized, install Python from https://www.python.org/downloads/ and select **Add Python to PATH** during installation.

## Run on macOS / Linux

Open Terminal in this folder and run:

```bash
python3 -m http.server 8000
```

Then visit http://localhost:8000.

## Teach the model

1. Select **Indian Sign Language (ISL)** or **American Sign Language (ASL)**.
2. Start the camera.
3. In the Training Lab, enter a short label such as `HELLO`.
4. Hold the intended hand pose in the camera and click **Record 15 samples**.
5. Repeat the recording at least 2–3 times, changing the angle and position slightly.
6. Add more labels, each with its own examples.
7. Show one of your trained poses to the camera. When the prediction stabilizes, click **Add sign to sentence** or **Speak sign**.

For a fairer demo, record examples from the same camera position and lighting you will use during judging. More varied samples may improve robustness, but avoid mixing different signs under the same label.

## How it works

- MediaPipe Hand Landmarker detects one hand and returns 21 landmark points.
- The app normalizes each point relative to the wrist and palm width.
- Training samples are saved in the browser's `localStorage`, separately for ISL and ASL.
- A k-NN classifier compares live landmarks with saved examples and predicts the closest label.
- Browser Speech Synthesis reads the recognized label or sentence aloud.

## Privacy

Camera frames are processed in your browser. This app does not upload camera video or training samples to a server. Training samples remain in the current browser's local storage. Clearing browser site data will remove them.

## Troubleshooting

- **Camera blocked:** Use `http://localhost:8000` or HTTPS, then allow camera permission in browser settings.
- **Blank camera:** Close other apps that may be using the webcam, refresh, and try again.
- **Model not loading:** Check your internet connection; MediaPipe files load from public CDNs.
- **Weak predictions:** Record more samples, keep the hand fully visible, improve lighting, and use a plain background.
- **No sound:** Check system volume and browser speech-synthesis support.

## Suggested hackathon improvements

- Add a larger, reviewed ISL and ASL dataset with permission from contributors.
- Use temporal sequences for motion-based signs.
- Add two-hand tracking and facial/non-manual features.
- Add a confidence calibration and unknown-sign rejection test.
- Conduct user testing with Deaf signers and sign-language interpreters.
