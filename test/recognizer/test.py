from recognizer import StarbucksLogoDetector

if __name__ == "__main__":
    reference_logo_path = "starbucks_logo.png"
    test_image_path = "/test/image.PNG"

    detector = StarbucksLogoDetector(reference_logo_path)
    result = detector.detect_logo(test_image_path)

    if result.starbuck_logo:
        print("Starbucks logo detected!")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Matches: {result.inliers}/{result.matches}")
        print(f"Detected corners: {result.corners}")
    else:
        print("No Starbucks logo detected.")
        if result.error:
            print(f"Error: {result.error}")
        else:
            print(f"Confidence (too low): {result.confidence:.2f}")
