import math
import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
from skimage.color import rgb2gray, rgba2rgb
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks


class PointlessDeskewImageProcessor:

    def __init__(self, text_analyzer, visualizer, plot_visualization):
        self.text_analyzer = text_analyzer
        self.visualizer = visualizer
        self.plot_visualization = plot_visualization

    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert an input image to grayscale.

        This function first checks if the input image is in RGBA format (4 channels). If it is,
        it converts the image to RGB format. Then, it converts the RGB or the original image
        (if it wasn't RGBA) to grayscale. If the input image is already in grayscale or has less
        than 3 channels, it returns the image as-is.

        Args:
            image (np.ndarray): An image array. The image can be in RGBA, RGB, or grayscale format.

        Returns:
            np.ndarray: The grayscale version of the input image.

        """
        # Convert RGBA to RGB if the image has 4 channels.
        if image.shape[-1] == 4:
            image = rgba2rgb(image)

        # Convert to grayscale.
        img_gray = rgb2gray(image) if image.ndim == 3 else image

        return img_gray

    def perform_hough_transform(
        self, img_gray: np.ndarray, sigma: float, num_angles: int
    ) -> Tuple[np.ndarray, List[float], List[float], np.ndarray]:
        """Performs the Hough Transform on a grayscale image to detect lines.

        This function applies the Canny edge detection algorithm to the input grayscale image
        to identify edges. It then performs the Hough Line Transform on these edges to detect lines.
        The Hough Transform detects lines by finding accumulations in the Hough space, represented by
        an accumulator array along with corresponding angles and distances for detected lines.

        Args:
            img_gray (np.ndarray): The input grayscale image array.
            sigma (float): The standard deviation for the Gaussian filter used in Canny edge detection.
            num_angles (int): The number of angles to sample in the Hough Transform, affecting the angular resolution.

        Returns:
            Tuple[np.ndarray, List[float], List[float], np.ndarray]: A tuple containing:
                - The accumulator array from the Hough Transform, indicating the strength of line detections.
                - A list of angles (in radians) corresponding to the peaks in the accumulator array, representing the detected line orientations.
                - A list of distances (in pixel units) from the origin to the detected lines, corresponding to the peaks in the accumulator array.
                - The edges detected in the input image as a result of applying the Canny edge detector.

        Example:
            >>> img_gray = np.array(Image.open('path/to/image').convert('L'))
            >>> accumulator, angles, distances, edges = perform_hough_transform(img_gray, sigma=2, num_angles=180)
            >>> # Use accumulator, angles, distances, and edges for further analysis or visualization
        """

        # Apply Canny edge detection to the grayscale image
        edges = canny(img_gray, sigma=sigma)

        # Perform the Hough Line Transform on the detected edges
        accumulator, angles, distances = hough_line(
            edges, np.linspace(-np.pi / 2, np.pi / 2, num_angles, endpoint=False)
        )

        # Convert angles and distances to lists and return them along with the accumulator
        return accumulator, angles.tolist(), distances.tolist(), edges

    def filter_and_correct_angles(self, angles_peaks: List[float]) -> List[float]:
        """Filters and corrects a list of angle peaks to ensure they fall within a specified range.

        This function performs two main operations on the input list of angles: correction and filtering.
        The correction step adjusts each angle by adding π/4, then modulo π/2, and subtracting π/4 again.
        This effectively rotates the angles within a π/2 range centered around 0. The purpose of this
        correction is to normalize the angles, ensuring that they are within a standard range for further processing.

        After correction, the function filters out any angles that do not fall within the specified minimum
        and maximum angle range (abs(π/4)). This step ensures that only small angles of interest are retained for further analysis or use.

        Args:
            angles_peaks (List[float]): A list of angles (in radians) to be corrected and filtered.

        Returns:
            List[float]: A list of corrected angles that fall within the specified minimum and maximum angle range.
        """
        # Correct the angles by normalizing them within a π/2 range centered around 0
        corrected_angles = [
            ((angle + np.pi / 4) % (np.pi / 2) - np.pi / 4) for angle in angles_peaks
        ]

        # Filter the corrected angles to retain only those within (-π/4, π/4) range
        corrected_angles = [
            angle for angle in corrected_angles if -np.pi / 4 <= angle <= np.pi / 4
        ]

        return corrected_angles

    def calculate_frequency_of_angles(
        self, angles_peaks: List[float]
    ) -> Dict[float, int]:
        """Calculates the frequency of each unique angle peak in a list.

        Iterates through the list of angle peaks to determine the frequency of each unique angle. It returns a dictionary
        where keys are the unique angles and values are their counts in the input list. This function treats each unique
        numerical value as a distinct angle, without considering numerical proximity.

        Args:
            angles_peaks (List[float]): A list of angles in radians, each represented as a floating-point number.

        Returns:
            Dict[float, int]: A dictionary mapping each unique angle to its frequency in the list.

        Example:
            >>> calculate_frequency_of_angles([0.0, 0.5, 0.0])
            {0.0: 2, 0.5: 1}

        Note:
        The function does not distinguish between angles that are numerically close but not exactly equal; each unique
        value in the input list is treated as a distinct angle. Consequently, the precision of the input angles directly
        affects the output frequency distribution.
        """

        return {peak: angles_peaks.count(peak) for peak in angles_peaks}

    def determine_skew_angle(self, freqs: Dict[float, int]) -> Optional[float]:
        """
        Determines the most frequent skew angle from a dictionary of angle frequencies.

        This function identifies the angle with the highest frequency in a given dictionary where keys represent angle values
        and values represent the frequency of those angles. The angle with the highest frequency is considered the most
        frequent skew angle. If the input dictionary is empty, the function returns None, indicating that no skew angle
        can be determined from an empty dataset.

        Args:
            Dict[float, int]: A dictionary with angles as keys (float) and their frequencies as values (int).

        Returns:
            Optional[float]: The angle with the highest frequency or None if the input dictionary is empty. This return
        value is of type Optional[float] to accommodate the possibility of an empty input.

        Example:
        >>> determine_skew_angle({0.0: 2, 0.5: 3, -0.5: 1})
        0.5
        """

        return max(freqs, key=freqs.get) if freqs else None

    def determine_skew(
        self,
        image: np.ndarray,
        sigma: float = 1.0,
        num_peaks: int = 20,
        min_deviation: float = 0.1,
        plot_visualization: bool = False,
    ) -> Tuple[Optional[float], np.ndarray, np.ndarray]:
        """
        Detects the skew angle of an image after performing a Hough transform to find lines in the image.

        The function first converts the image to grayscale and then applies a Hough transform to detect lines.
        It filters these lines to focus on a specific range of angles (-π/8 to π/8 radians) and calculates the
        frequency of occurrence of each angle within this range. The most frequent angle (skew angle) is considered
        the dominant line orientation, which can be indicative of the image's skew. Optionally, this function can
        also plot visualizations of the edges detected, the Hough transform space, and the detected lines on the
        original image for analysis and debugging purposes.

        Parameters:
        - image (Any): The input image on which skew detection will be performed.
        - sigma (float, optional): The standard deviation of the Gaussian filter used in edge detection, defaulting to 2.0.
        - num_peaks (int, optional): The number of peaks to identify in the Hough transform, defaulting to 20.
        - min_deviation (float, optional): The minimum deviation for angle calculations, affecting the granularity of the analysis, defaulting to 1.0 degree.
        - plot_visualization (bool, optional): If True, visualizations of the processing steps will be displayed, defaulting to False.

        Returns:
        - Tuple[Optional[float], np.ndarray, np.ndarray]: A tuple containing the detected skew angle in degrees (or None if not determined),
        the array of angles considered peaks in the Hough transform, and the array of corrected angles within the desired range.

        Note:
        The accuracy of the skew detection depends on the quality of the image, the appropriateness of the sigma value for
        edge detection, and the specified range for considering angle peaks. Adjusting these parameters may be necessary
        for optimal skew detection in different images.
        """

        # Convert image to grayscale
        img_gray = self.convert_to_grayscale(image)
        # Determine the number of angles to analyze based on the minimum deviation
        num_angles = round(180 / min_deviation)
        # Perform Hough transform on the grayscale image
        accumulator, angles, distances, edges = self.perform_hough_transform(
            img_gray, sigma, num_angles
        )
        # Find peaks in the Hough transform
        _, angles_peaks, dists_peaks = hough_line_peaks(
            accumulator, np.array(angles), np.array(distances), num_peaks=num_peaks
        )
        # Correct and filter angle peaks
        corrected_angles = self.filter_and_correct_angles(angles_peaks)

        # Optional visualization of the process
        if plot_visualization:
            self.visualizer.visualize_edges(
                edges
            )  # Visualize detected edges in the image
            self.visualizer.visualize_image_and_hough_space(
                img_gray, accumulator, angles, distances
            )  # Visualize the Hough space
            self.visualizer.visualize_hough_lines(
                img_gray, angles_peaks, dists_peaks
            )  # Visualize Hough lines on the image

        # Calculate the frequency of corrected angles and determine the skew angle
        freqs = self.calculate_frequency_of_angles(corrected_angles)
        skew_angle = self.determine_skew_angle(freqs)
        # Convert the skew angle to degrees
        skew_angle_deg = np.rad2deg(skew_angle) if skew_angle is not None else None

        return skew_angle_deg, angles_peaks, corrected_angles

    def process_image(
        self,
        image_path: str,
        plot_visualization: bool = False,
        image_scale_factor: float = 0.5,
    ):
        """
        Processes an image to detect and correct skew, then saves the corrected image.

        This function performs several steps to process an image, including loading, resizing,
        skew detection, skew correction, and optionally visualizing the corrected image. The image is first
        resized based on a given scale factor to optimize the skew detection process. Then, the skew of the
        resized image is detected. If a significant skew is identified, the original image is rotated to correct
        this skew and the corrected image is saved to the specified output path. Optionally, the corrected image
        can be displayed using matplotlib.

        Parameters:
        - image_path (str): The path to the input image to be processed.
        - output_path (str): The path where the corrected image should be saved.
        - plot_visualization (bool, optional): If True, displays the corrected image using matplotlib. Defaults to True.
        - image_scale_factor (float, optional): The factor by which the image should be scaled for processing. Defaults to 0.5.

        Returns:
        - tuple: Returns a tuple containing the detected skew angle (or None if not detected), the angles considered
        as peaks in the Hough transform, and the corrected angles within the desired range.

        """

        # Load the image from the specified path
        image = cv2.imread(image_path)

        # Calculate the new dimensions for resizing
        width = int(image.shape[1] * image_scale_factor)
        height = int(image.shape[0] * image_scale_factor)
        new_dim = (width, height)

        # Resize the image to the new dimensions
        resized_image = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)

        # Determine the skew of the resized image and get the skew angle and peaks
        skew_angle, angles_peaks, corrected_angles = self.determine_skew(
            resized_image, plot_visualization=self.plot_visualization
        )

        # If a skew angle is detected, correct the skew by rotating the original image
        if skew_angle is not None:
            # Rotate the original image to correct the skew
            rotated_image = Image.open(image_path).rotate(
                skew_angle, expand=True, fillcolor="white"
            )  # quality needs to be tested

            # Save the corrected image to the specified output path
            # rotated_image.save(output_path)
            # cv2.imwrite(output_path, rotated_image)
            skew_angle = round(skew_angle, 2)
            st.write(f"Hough thinks the angle is {skew_angle}°")
            print(f"Detected angle is {skew_angle}")
            # print(f"Rotated image saved to {output_path}")

            # Optionally, display the corrected image using matplotlib
            if plot_visualization:
                plt.imshow(rotated_image)
                plt.axis("off")
                plt.show()
        else:
            print("No significant skew detected.")

        return skew_angle, angles_peaks, corrected_angles, rotated_image

    def process_and_display_image_in_streamlit_app(
        self,
        image_path_or_buffer,
        allow_up_to_180_degrees,
    ):
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=(
                os.path.splitext(image_path_or_buffer)[1]
                if isinstance(image_path_or_buffer, str)
                else ".png"
            ),
        ) as tmp_file:
            if isinstance(image_path_or_buffer, str):
                # If image path is provided (selected image)
                img = Image.open(image_path_or_buffer)
            else:
                # If image buffer is provided (uploaded image)
                img = Image.open(image_path_or_buffer)
            img.save(tmp_file, format="PNG")
            tmp_file_path = tmp_file.name

        start_time = time.time()

        _, _, _, rotated_image = self.process_image(
            tmp_file_path,
            plot_visualization=self.plot_visualization,
            image_scale_factor=0.5,
        )
        if allow_up_to_180_degrees:
            _, rotated_image, _, _ = self.text_analyzer.orientation_rotation_estimation(
                rotated_image
            )

        st.image(rotated_image, use_column_width=True)
        end_time = time.time()
        st.write(f"Execution time: {end_time - start_time} seconds")
