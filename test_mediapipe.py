
import cv2
import mediapipe as mp
from pathlib import Path

# Pick a sample full-face image from the dataset
IMAGE_PATH = next(Path("D:/Fatigue_project/data/dataset_new/train/yawn").glob("*.jpg"))
print(f"Processing: {IMAGE_PATH.name}")

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,         # Image mode (not video stream)
    max_num_faces=1,                # We only expect one face
    refine_landmarks=True,          # Adds extra landmarks for eyes + lips (important!)
    min_detection_confidence=0.5
)

# Load the image
img = cv2.imread(str(IMAGE_PATH))
if img is None:
    print("Failed to load image")
    exit()

print(f"Image size: {img.shape[1]}x{img.shape[0]}")

# MediaPipe expects RGB, but OpenCV uses BGR — convert
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Run face mesh detection
results = face_mesh.process(img_rgb)

# Check if a face was found
if not results.multi_face_landmarks:
    print("⚠  No face detected!")
    exit()

print(f"✓ Detected {len(results.multi_face_landmarks)} face(s)")

# Get the first face's landmarks
face_landmarks = results.multi_face_landmarks[0]
print(f"✓ Number of landmarks: {len(face_landmarks.landmark)}")

# Draw all landmarks on a copy of the image
img_annotated = img.copy()

# Draw the full mesh (tesselation)
mp_drawing.draw_landmarks(
    image=img_annotated,
    landmark_list=face_landmarks,
    connections=mp_face_mesh.FACEMESH_TESSELATION,
    landmark_drawing_spec=None,
    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
)

# Draw the eye contours in green
mp_drawing.draw_landmarks(
    image=img_annotated,
    landmark_list=face_landmarks,
    connections=mp_face_mesh.FACEMESH_LEFT_EYE,
    landmark_drawing_spec=None,
    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
)
mp_drawing.draw_landmarks(
    image=img_annotated,
    landmark_list=face_landmarks,
    connections=mp_face_mesh.FACEMESH_RIGHT_EYE,
    landmark_drawing_spec=None,
    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
)

# Draw lip contours
mp_drawing.draw_landmarks(
    image=img_annotated,
    landmark_list=face_landmarks,
    connections=mp_face_mesh.FACEMESH_LIPS,
    landmark_drawing_spec=None,
    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
)

# Show side-by-side: original vs annotated
cv2.imshow("Original", img)
cv2.imshow("Face Mesh (468 landmarks)", img_annotated)

# Save the annotated image so we have a record
output_path = Path("D:/Fatigue_project/outputs/facemesh_test.jpg")
output_path.parent.mkdir(exist_ok=True)
cv2.imwrite(str(output_path), img_annotated)
print(f"✓ Saved annotated image to: {output_path}")

print("\nPress any key on an image window to close.")
cv2.waitKey(0)
cv2.destroyAllWindows()

face_mesh.close()
print("Done.")