"""
Starbucks Logo Detection System
Detects Starbucks logo in input images using SIFT features
"""

import cv2
import numpy as np
import os


class DetectionResult:
    def __init__(self, detected=False, confidence=0.0, matches=0, inliers=0, corners=None, error=None):
        self.starbuck_logo = detected
        self.confidence = confidence
        self.matches = matches
        self.inliers = inliers
        self.corners = corners if corners is not None else []
        self.error = error

    def __repr__(self):
        if self.error:
            return f"DetectionResult(Error: {self.error})"
        return (f"DetectionResult(Starbucks Logo: {self.starbuck_logo}, "
                f"Confidence: {self.confidence:.2f}, Matches: {self.matches}, "
                f"Inliers: {self.inliers}, Corners: {self.corners})")


class StarbucksLogoDetector:
    def __init__(self, reference_logo_path, min_matches=10, ratio_threshold=0.7):
        self.min_matches = min_matches
        self.ratio_threshold = ratio_threshold
        self.detector = cv2.SIFT.create(nfeatures=500)

        # Load and process reference logo
        if not self.load_reference_logo(reference_logo_path):
            raise ValueError(f"Could not load reference logo from {reference_logo_path}")

    def load_reference_logo(self, logo_path):
        if not os.path.exists(logo_path):
            print(f"Error: Reference logo not found at {logo_path}")
            return False

        logo_img = cv2.imread(logo_path)
        if logo_img is None:
            print(f"Error: Could not load image {logo_path}")
            return False

        self.reference_logo = cv2.cvtColor(logo_img, cv2.COLOR_BGR2GRAY)
        self.reference_kp, self.reference_desc = self.detector.detectAndCompute(self.reference_logo, None)

        if self.reference_desc is None or len(self.reference_desc) == 0:
            print("Error: No features found in reference logo")
            return False

        print(f"Reference logo loaded: {len(self.reference_kp)} features extracted")
        return True

    def detect_logo(self, input_image_path):
        if not os.path.exists(input_image_path):
            return DetectionResult(detected=False, error=f"Input image not found: {input_image_path}")

        input_img = cv2.imread(input_image_path)
        if input_img is None:
            return DetectionResult(detected=False, error="Could not load input image")

        gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_kp, input_desc = self.detector.detectAndCompute(gray_img, None)

        if input_desc is None or len(input_desc) == 0:
            return DetectionResult(detected=False, error="No features found in input image")

        bf = cv2.BFMatcher()
        try:
            matches = bf.knnMatch(self.reference_desc, input_desc, k=2)
        except cv2.error:
            return DetectionResult(detected=False, error="Feature matching failed")

        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair[0], match_pair[1]
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)

        if len(good_matches) < self.min_matches:
            confidence = len(good_matches) / self.min_matches if self.min_matches > 0 else 0
            return DetectionResult(detected=False, confidence=confidence, matches=len(good_matches))

        src_pts = np.float32([self.reference_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([input_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        if len(good_matches) < 4:
            return DetectionResult(detected=False, error="Not enough matches for homography")

        try:
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is None:
                return DetectionResult(detected=False, error="Homography calculation failed")

            inliers = int(np.sum(mask))
            confidence = inliers / len(good_matches)

            h, w = self.reference_logo.shape
            logo_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            detected_corners = cv2.perspectiveTransform(logo_corners, M).reshape(-1, 2).tolist()

            return DetectionResult(
                detected=True,
                confidence=confidence,
                matches=len(good_matches),
                inliers=inliers,
                corners=detected_corners
            )

        except cv2.error as e:
            return DetectionResult(detected=False, error=f"Homography error: {str(e)}")