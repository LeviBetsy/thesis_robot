import cv2
from pathlib import Path

def take_picture():
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]  # Goes up two levels from scripts/
    output_dir = project_root / "data" / "references"
    
    # Automatically create the folder if it doesn't exist yet
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving images to: {output_dir}")

    # 2. Find the next available index 'i' so we don't overwrite existing photos
    existing_files = list(output_dir.glob("ref*.jpg"))
    if existing_files:
        # Extracts numbers from filenames like 'ref.jpg' to find the maximum
        try:
            indices = [int(f.stem.replace("ref", "")) for f in existing_files]
            img_counter = max(indices) + 1
        except ValueError:
            img_counter = 0
    else:
        img_counter = 0

    # 3. Initialize the camera (0 is usually the default built-in/USB webcam)
    cam = cv2.VideoCapture(0)
    
    if not cam.isOpened():
        print("Error: Could not open camera.")
        return

    print("\n--- Camera Control Instructions ---")
    print("  Press 'SPACE' or 'S' to snap and save a frame")
    print("  Press 'ESC' or 'Q' to quit")
    print("-----------------------------------\n")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        # # Display the live stream in a window
        # cv2.imshow("Calibration Capture - Live Feed", frame)

        # # Wait for a key press (1 millisecond delay)
        # key = cv2.waitKey(1) & 0xFF

        # # Press 'ESC' or 'q' to exit the loop
        # if key == 27 or key == ord('q'):
        #     print("Closing camera stream...")
        #     break
        
        # # Press 'SPACE' or 's' to snap a picture
        # elif key == 32 or key == ord('s'):
        #     filename = f"ref{img_counter}.jpg"
        #     file_path = output_dir / filename
            
        #     # Save the frame image
        #     cv2.imwrite(str(file_path), frame)
        #     print(f"Successfully saved: {filename}")
            
        #     img_counter += 1
        filename = f"ref{img_counter}.jpg"
        file_path = output_dir / filename
        
        # Save the frame image
        cv2.imwrite(str(file_path), frame)
        print(f"Successfully saved: {filename}")
        break

    # Clean up and release the hardware resources
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    take_picture()