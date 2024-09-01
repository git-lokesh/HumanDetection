import cv2
import imutils
import argparse

def detect(frame, HOGCV):
    # Detecting humans using HOG descriptor
    bounding_box_cordinates, weights = HOGCV.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.03)
    
    person_count = len(bounding_box_cordinates)
    
    for i, (x, y, w, h) in enumerate(bounding_box_cordinates, 1):
        # Drawing bounding box and labeling detected humans
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'Person {i}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Displaying the status and total count on the frame
    cv2.putText(frame, 'Status: Detecting', (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, f'Total Persons: {person_count}', (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    cv2.imshow('Human Detection', frame)
    return frame

def detectByPathImage(path, output_path, HOGCV):
    # Reading and resizing the image
    image = cv2.imread(path)
    image = imutils.resize(image, width=min(800, image.shape[1]))
    
    # Detect humans in the image
    result_image = detect(image, HOGCV)
    
    # Saving the result image if an output path is provided
    if output_path is not None:
        cv2.imwrite(output_path, result_image)
    
    # Waiting for a key press and then closing all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def humanDetector(args, HOGCV):
    image_path = args.get("image")
    if image_path is not None:
        print('[INFO] Opening Image from path.')
        detectByPathImage(image_path, args.get('output'), HOGCV)

def argsParser():
    # Argument parsing for image input and output paths
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-i", "--image", required=True, help="Path to image file")
    arg_parse.add_argument("-o", "--output", type=str, default="output.jpg", help="Path to output image file")
    args = vars(arg_parse.parse_args())
    return args

if __name__ == "__main__":
    # Initialize HOG descriptor for detecting people
    HOGCV = cv2.HOGDescriptor()
    HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Parsing arguments
    args = argsParser()
    
    # Running the human detector
    humanDetector(args, HOGCV)
