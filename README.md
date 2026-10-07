# Dino Motion Controller

Play the Chrome dinosaur game with your body. A webcam tracks your pose in real time: **jump** to make the dino jump, and **squat** to make it duck.

## How it works

- **Pose estimation:** each webcam frame runs through Google's [MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) (about 17 ms per frame on a laptop CPU). The controller follows the midpoint of your hips.
- **Calibration:** you stand in position and press **S** to lock in your standing height. The line stays fixed so movement can't drift it.
- **Jump detection:** a jump fires on the first frame your hips move upward quickly (takeoff speed), not once you reach the top of the jump. A height threshold acts as a backup. Each jump fires once and re-arms when you land, which prevents double presses.
- **Duck detection:** when your hips drop below a duck line, the controller holds the Down arrow and releases it when you stand back up. Separate press and release depths keep the key from flickering.
- **Input:** key presses are sent to whichever window has focus, using `pynput`.

## Setup

Requires Python 3.10+ and a webcam.

```bash
pip install -r requirements.txt
python main.py
```

On Windows, you can double-click `run.bat` instead.

## Playing

1. Open `chrome://dino` in Chrome.
2. In the controller window, stand where you'll play, with your hips visible, and press **S**.
3. Click into the Chrome tab, then jump over cactuses and squat under birds.

| Key (in the controller window) | Action |
|---|---|
| `S` / `Enter` | Set standing height and start |
| `+` / `-` | More / less sensitive |
| `R` | Recalibrate |
| `Q` | Quit |

## Background

This started as a Geometry Dash controller. Measuring the latency (camera about 24 fps, pose model about 17 ms) showed that most of the delay came from the detection rule, not the hardware. Switching from height-based to takeoff-speed detection made jumps register about 3 frames (~125 ms) sooner in simulated testing. Geometry Dash still turned out to be a poor fit: it needs rapid back-to-back clicks and held inputs, and a human jump takes about half a second. The dinosaur game's pacing matches full-body movement, so the project moved there and added ducking.

## Credits

Pose model: `pose_landmarker_lite.task` from [MediaPipe](https://github.com/google-ai-edge/mediapipe) (Apache 2.0).
