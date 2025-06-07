"""
Starbucks Logo Detection System
Detects Starbucks logo in input images using SIFT features
"""
import datetime
import shutil

import cv2
import numpy as np
import os
import argparse
from pathlib import Path


class StarbucksLogoDetector:
    def __init__(self, reference_logo_path, min_matches=10, ratio_threshold=0.7):
        """
        Initialize the Starbucks logo detector

        Args:
            reference_logo_path: Path to the reference Starbucks logo image
            min_matches: Minimum number of good matches required for detection
            ratio_threshold: Threshold for ratio test (lower = stricter)
        """
        self.min_matches = min_matches
        self.ratio_threshold = ratio_threshold
        self.detector = cv2.SIFT.create(nfeatures=500)

        # Load and process reference logo
        self.reference_logo = None
        self.reference_kp = None
        self.reference_desc = None

        if not self.load_reference_logo(reference_logo_path):
            raise ValueError(f"Could not load reference logo from {reference_logo_path}")

    def load_reference_logo(self, logo_path):
        """Load and extract features from reference Starbucks logo"""
        if not os.path.exists(logo_path):
            print(f"Error: Reference logo not found at {logo_path}")
            return False

        # Load logo image
        logo_img = cv2.imread(logo_path)
        if logo_img is None:
            print(f"Error: Could not load image {logo_path}")
            return False

        # Convert to grayscale
        self.reference_logo = cv2.cvtColor(logo_img, cv2.COLOR_BGR2GRAY)

        # Extract SIFT features
        self.reference_kp, self.reference_desc = self.detector.detectAndCompute(
            self.reference_logo, None
        )

        if self.reference_desc is None or len(self.reference_desc) == 0:
            print("Error: No features found in reference logo")
            return False

        print(f"Reference logo loaded: {len(self.reference_kp)} features extracted")
        return True

    def detect_logo(self, input_image_path, output_path=None, visualize=True):
        """
        Detect Starbucks logo in input image

        Args:
            input_image_path: Path to input image
            output_path: Path to save result (optional)
            visualize: Whether to draw bounding box around detected logo

        Returns:
            dict: Detection results with confidence, coordinates, etc.
        """
        # Load input image
        if not os.path.exists(input_image_path):
            return {"detected": False, "error": f"Input image not found: {input_image_path}"}

        input_img = cv2.imread(input_image_path)
        if input_img is None:
            return {"detected": False, "error": "Could not load input image"}

        # Convert to grayscale
        gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

        # Extract features from input image
        input_kp, input_desc = self.detector.detectAndCompute(gray_img, None)

        if input_desc is None or len(input_desc) == 0:
            return {"detected": False, "error": "No features found in input image"}

        # Match features
        bf = cv2.BFMatcher()
        try:
            matches = bf.knnMatch(self.reference_desc, input_desc, k=2)
        except cv2.error:
            return {"detected": False, "error": "Feature matching failed"}

        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair[0], match_pair[1]
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)

        print(f"Found {len(good_matches)} good matches")

        # Check if we have enough matches
        if len(good_matches) < self.min_matches:
            return {
                "detected": False,
                "matches": len(good_matches),
                "required_matches": self.min_matches,
                "confidence": len(good_matches) / self.min_matches
            }

        # Extract matched points
        src_pts = np.float32([self.reference_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([input_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find homography
        try:
            if len(good_matches) >= 4:
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if M is not None:
                    # Get reference logo dimensions
                    h, w = self.reference_logo.shape

                    # Define corners of reference logo
                    logo_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

                    # Transform corners to input image
                    detected_corners = cv2.perspectiveTransform(logo_corners, M)

                    # Calculate detection confidence based on inliers
                    inliers = np.sum(mask)
                    confidence = inliers / len(good_matches)

                    result = {
                        "detected": True,
                        "matches": len(good_matches),
                        "inliers": int(inliers),
                        "confidence": float(confidence),
                        "corners": detected_corners.reshape(-1, 2).tolist()
                    }

                    # Visualize result if requested
                    if visualize:
                        result_img = input_img.copy()

                        # Draw bounding box
                        cv2.polylines(result_img, [np.int32(detected_corners)], True, (0, 255, 0), 3, cv2.LINE_AA)

                        # Add text
                        cv2.putText(result_img, f'Starbucks Logo Detected!',
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(result_img, f'Confidence: {confidence:.2f}',
                                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(result_img, f'Matches: {inliers}/{len(good_matches)}',
                                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        # Save result if output path provided
                        if output_path:
                            cv2.imwrite(output_path, result_img)
                            result["output_saved"] = output_path

                        result["result_image"] = result_img

                    return result
                else:
                    return {"detected": False, "error": "Homography calculation failed"}
            else:
                return {"detected": False, "error": "Not enough matches for homography"}

        except cv2.error as e:
            return {"detected": False, "error": f"Homography error: {str(e)}"}

    def batch_detect(self, input_folder, output_folder=None):
        """
        Detect logos in multiple images

        Args:
            input_folder: Folder containing input images
            output_folder: Folder to save results (optional)
        """
        if not os.path.exists(input_folder):
            print(f"Error: Input folder not found: {input_folder}")
            return

        if output_folder:
            os.makedirs(output_folder, exist_ok=True)

        # Supported image extensions
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

        results = []
        for filename in os.listdir(input_folder):
            if any(filename.lower().endswith(ext) for ext in extensions):
                input_path = os.path.join(input_folder, filename)
                output_path = None

                if output_folder:
                    name, ext = os.path.splitext(filename)
                    output_path = os.path.join(output_folder, f"{name}_detected{ext}")

                print(f"Processing: {filename}")
                result = self.detect_logo(input_path, output_path)
                result["filename"] = filename
                results.append(result)

                if result["detected"]:
                    print(f"  ✓ Logo detected! Confidence: {result['confidence']:.2f}")
                else:
                    print(f"  ✗ No logo detected")

        return results


def main():
    parser = argparse.ArgumentParser(description='Detect Starbucks logo in images')
    parser.add_argument('--reference', '-r', required=True,
                        help='Path to reference Starbucks logo image')
    parser.add_argument('--input', '-i', required=True,
                        help='Path to input image or folder')
    parser.add_argument('--output', '-o',
                        help='Path to save output image(s)')
    parser.add_argument('--min-matches', type=int, default=10,
                        help='Minimum number of matches required (default: 10)')
    parser.add_argument('--ratio-threshold', type=float, default=0.7,
                        help='Ratio test threshold (default: 0.7)')
    parser.add_argument('--batch', action='store_true',
                        help='Process multiple images in a folder')

    args = parser.parse_args()

    try:
        # Initialize detector
        detector = StarbucksLogoDetector(
            args.reference,
            min_matches=args.min_matches,
            ratio_threshold=args.ratio_threshold
        )

        if args.batch:
            results = detector.batch_detect(args.input, args.output)

            # Print summary
            detected_count = sum(1 for r in results if r.get("detected", False))
            print(f"\nSummary: {detected_count}/{len(results)} images contained Starbucks logo")

        else:
            # Single image processing
            folder = Path("test/tests")
            new_folder = datetime.datetime.now().strftime("result_%Y%m%d_%H%M%S")
            new_path = folder / new_folder
            new_path.mkdir(parents=True, exist_ok=True)
            output = new_path / "output.png"
            input = new_path / args.input
            shutil.move(args.input, input)
            result = detector.detect_logo(input, output)

            if result["detected"]:
                print(f"✓ Starbucks logo detected!")
                print(f"  Confidence: {result['confidence']:.2f}")
                print(f"  Matches: {result['inliers']}/{result['matches']}")
                if args.output:
                    print(f"  Result saved to: {args.output}")
            else:
                print("✗ No Starbucks logo detected")
                if "error" in result:
                    print(f"  Error: {result['error']}")
                elif "confidence" in result:
                    print(f"  Confidence too low: {result['confidence']:.2f}")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()