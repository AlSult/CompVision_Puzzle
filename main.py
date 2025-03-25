import cv2
import numpy as np

def extract_puzzle_pieces(image, min_area=500):
    """Extract individual puzzle pieces from a shuffled image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=2)
    closed_edges = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pieces = []

    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            piece = image[y:y + h, x:x + w]
            pieces.append((piece, (x, y, w, h)))

    return pieces

def feature_matching_with_ransac(ref_img, piece, index):
    """Perform SIFT feature matching and compute alignment using RANSAC."""
    # SIFT detector
    sift = cv2.SIFT_create()

    # Compute keypoints and descriptors for the reference image and the puzzle piece
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    piece_gray = cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY)

    keypoints_ref, descriptors_ref = sift.detectAndCompute(ref_gray, None)
    keypoints_piece, descriptors_piece = sift.detectAndCompute(piece_gray, None)

    # Matching features using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(descriptors_piece, descriptors_ref, k=2)

    # Applying the ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        print(f"Not enough matches for puzzle piece {index + 1}")
        return None, keypoints_piece, keypoints_ref

    # Extracting matched keypoints
    src_pts = np.float32([keypoints_piece[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Computing the homography using RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        print(f"Homography computation failed for puzzle piece {index + 1}")
        return None, good_matches, keypoints_piece, keypoints_ref

    return H, good_matches, keypoints_piece, keypoints_ref

def make_white_transparent(piece):
    """Make white parts of the puzzle piece transparent."""
    # Converting the image to BGRA (adding an alpha channel)
    piece_bgra = cv2.cvtColor(piece, cv2.COLOR_BGR2BGRA)

    # Making white parts transparent (255, 255, 255)
    lower_white = np.array([255, 255, 255, 255])  # White with full opacity
    upper_white = np.array([255, 255, 255, 255])  # Upper bound for white

    # Setting white pixels to transparent (alpha = 0)
    piece_bgra[(piece_bgra[:, :, 0] == 255) & (piece_bgra[:, :, 1] == 255) & (piece_bgra[:, :, 2] == 255)] = [0, 0, 0, 0]

    return piece_bgra

def reconstruct_puzzle(ref_img, pieces, ref_h, ref_w):
    """Reconstruct the full puzzle by placing each piece back into its position."""
    # A blank canvas for the reconstructed image (same size as reference image), initialized with full transparency
    reconstructed_img = np.zeros((ref_h, ref_w, 4), dtype=np.uint8)

    # Placing each aligned piece back into the reconstructed image
    for idx, (piece, (x, y, w, h)) in enumerate(pieces):
        print(f"Processing piece {idx + 1}/{len(pieces)}")

        # Converting the piece to BGRA and make white parts transparent
        piece_bgra = make_white_transparent(piece)
        
        H, _, _, _ = feature_matching_with_ransac(ref_img, piece, idx)
        
        if H is not None:
            # The homography for debugging
            print(f"Piece {idx + 1}: Homography matrix:\n{H}")

            # Warpping the piece using homography to align it with the reference image
            aligned_piece = cv2.warpPerspective(piece_bgra, H, (ref_w, ref_h))

            # Overlaying the aligned piece onto the reconstructed image, preserving transparency
            print(f"Placing piece {idx + 1} at position ({x}, {y}) with size ({w}, {h})")

            # Using the alpha channel to overlay the piece
            alpha_mask = aligned_piece[:, :, 3] > 0  
            reconstructed_img[alpha_mask] = aligned_piece[alpha_mask]

    return reconstructed_img

def main():
    # Load the complete puzzle and shuffled image
    ref_img_path = "peppa/peppa.png"  # Full completed puzzle
    shuffled_img_path = "peppa/pz.png"  # Shuffled pieces

    ref_img = cv2.imread(ref_img_path)
    shuffled_img = cv2.imread(shuffled_img_path)

    if ref_img is None or shuffled_img is None:
        print("Error: One or more images could not be loaded!")
        return

    # Extracting puzzle pieces
    pieces = extract_puzzle_pieces(shuffled_img)

    # Getting the reference image's dimensions
    ref_h, ref_w, _ = ref_img.shape

    # Reconstructing the full puzzle
    reconstructed_img = reconstruct_puzzle(ref_img, pieces, ref_h, ref_w)

    # Converting the reconstructed image from BGRA to BGR for display
    reconstructed_img_bgr = cv2.cvtColor(reconstructed_img, cv2.COLOR_BGRA2BGR)

    # Displaying the final reconstructed puzzle
    cv2.imshow("Reconstructed Puzzle", reconstructed_img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
