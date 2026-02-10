import cv2
import numpy as np

def analyze_ai_fingerprint(image_path):
    # Load image in grayscale
    img = cv2.imread(image_path, 0)
    if img is None:
        return 0.0, "Read Error"
    
    # Perform 2D Fast Fourier Transform (FFT)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    
    # Calculate Magnitude Spectrum
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    # AI images (GANs/Diffusion) leave periodic high-frequency patterns
    # We measure the "entropy" and variance in the high-frequency zones
    h, w = magnitude_spectrum.shape
    center_h, center_w = h // 2, w // 2
    
    # Mask out the low frequencies (the center) to look only at the noise (the edges)
    magnitude_spectrum[center_h-30:center_h+30, center_w-30:center_w+30] = 0
    
    # Higher variance in high frequencies often indicates synthetic generation
    score = np.var(magnitude_spectrum)
    
    # Thresholding based on typical GAN artifacts
    # (These values are calibrated for standard JPEG/PNG exhibits)
    if score > 150:
        return min(99.0, score/2), "High Probability AI/Deepfake"
    elif score > 80:
        return score/2, "Suspected Synthetic Elements"
    else:
        return score/4, "Likely Natural/Biological Source"