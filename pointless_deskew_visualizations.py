from typing import NoReturn

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


class PointlessDeskewImageVisualizer:
    def __init__(self, in_streamlit=False):
        """
        Initializes the visualizer with the option to specify its execution context.

        Args:
            in_streamlit (bool): If True, indicates the visualizer is being run
                                 within a Streamlit application. Defaults to False.
        """
        self.in_streamlit = in_streamlit

    def _display_image(self, image: np.ndarray, title: str = "") -> NoReturn:
        """
        Displays an image using matplotlib or Streamlit based on the execution context.

        This is a helper function designed for internal use within the class to abstract away
        the details of image display mechanisms in different environments.

        Args:
            image (np.ndarray): The image to be displayed, represented as a numpy array.
            title (str): The title of the plot. Defaults to an empty string if not provided.

        Returns:
            NoReturn: This function does not return any value.
        """
        if self.in_streamlit:
            st.image(image, caption=title, use_column_width=True)
        else:
            plt.imshow(image, cmap="gray")
            plt.title(title)
            plt.axis("off")
            plt.show()

    def _plot_with_streamlit(self, fig) -> NoReturn:
        """
        Plots a matplotlib figure using Streamlit or matplotlib directly, based on the execution context.

        This method is designed as a helper function for internal use, facilitating the
        display of matplotlib figures in either a Streamlit app or a standard matplotlib
        output environment.

        Args:
            fig (matplotlib.figure.Figure): The matplotlib figure to be plotted.

        Returns:
            NoReturn: This method does not return any value.
        """
        if self.in_streamlit:
            st.pyplot(fig)
        else:
            plt.tight_layout()
            plt.show()

    def visualize_edges(self, edges: np.ndarray) -> NoReturn:
        """Displays the edges detected in an image as a grayscale plot.

        This method visualizes the edges detected in an image, emphasizing the effectiveness
        of the edge detection process. It assumes the edges are the output from an edge
        detection algorithm, such as the Canny edge detector, and displays them in white
        against a black background to facilitate assessment.

        Args:
            edges (np.ndarray): A 2D numpy array representing the detected edges, where the
                value of each pixel indicates the presence (values greater than 0) or
                absence (value of 0) of an edge at that pixel.

        Returns:
            NoReturn: This method does not return any value and directly displays the visualization.
        """
        edges_display = (edges * 255).astype(np.uint8)
        self._display_image(edges_display, "Edges detected")

    def visualize_hough_lines(
        self,
        original_image: np.ndarray,
        angle_peaks: np.ndarray,
        dist_peaks: np.ndarray,
    ) -> NoReturn:
        """Visualizes the original image with detected Hough lines superimposed.

        This method displays the original grayscale image and the same image with Hough lines
        overlaid in red. It uses the results of a Hough transform, specifically the most significant
        angles and distances of detected lines, to draw these lines on the original image. This visualization
        is helpful for assessing the effectiveness of line detection using the Hough transform.

        Args:
            original_image (np.ndarray): The original grayscale image as a 2D numpy array.
            angle_peaks (np.ndarray): Angles (in radians) of the most significant lines detected.
            dist_peaks (np.ndarray): Distances of the most significant lines detected.

        Returns:
            NoReturn: This method does not return any value and displays the visualization directly.
        """
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))

        # Original Image
        ax[0].imshow(original_image, cmap="gray")
        ax[0].set_title("Original Image")
        ax[0].set_axis_off()

        # Image with Hough lines
        ax[1].imshow(original_image, cmap="gray")
        for angle, dist in zip(angle_peaks, dist_peaks):
            (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
            ax[1].plot(
                (x0 - 1000 * np.sin(angle), x0 + 1000 * np.sin(angle)),
                (y0 + 1000 * np.cos(angle), y0 - 1000 * np.cos(angle)),
                "-r",
            )
        ax[1].set_xlim((0, original_image.shape[1]))
        ax[1].set_ylim((original_image.shape[0], 0))
        ax[1].set_title("Image with Hough lines")
        ax[1].set_axis_off()

        self._plot_with_streamlit(fig)

    def visualize_hough_space(
        self, accumulator: np.ndarray, angles: np.ndarray, distances: np.ndarray
    ) -> NoReturn:
        """Visualizes the Hough space resulting from the Hough Transform.

        This method creates an enhanced visualization of the Hough space, which is essentially
        the accumulator space of the Hough Transform. It plots the accumulator values as a heatmap,
        with the x-axis representing angles (in degrees for readability) and the y-axis representing
        distances. This heatmap illustrates how many times each line, characterized by its angle and
        distance from the origin, was voted as a potential line in the image during the Hough Transform.

        The intensity of the colors in the heatmap corresponds to the number of votes received: brighter
        colors indicate a higher number of votes, suggesting a stronger evidence of a line's presence
        at that angle and distance. Peaks in the Hough space (areas of high intensity) correspond to
        lines that are more likely to actually exist in the image. This visualization is useful for
        understanding the distribution and prominence of lines detected by the Hough Transform, allowing
        for an assessment of line detection performance and the identification of significant lines.

        Args:
            accumulator (np.ndarray): The accumulator array from the Hough Transform, indicating
                the number of votes for each angle-distance pair.
            angles (np.ndarray): The array of angles, in radians, used in the Hough Transform.
            distances (np.ndarray): The array of distances used in the Hough Transform.

        Returns:
            NoReturn: Does not return any value and displays the visualization directly.

        Interpretation:
            - Brighter areas in the plot indicate more votes for a particular line, suggesting a
            higher likelihood that the line exists in the image.
            - The angle and distance axes allow you to determine the orientation and position
            of these detected lines relative to the image frame.
            - Observing clusters or distinct peaks can help identify dominant line features
            in the image, useful for applications like edge detection, shape analysis, and
            feature extraction.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        angle_degrees = np.rad2deg(
            angles
        )  # Convert angles to degrees for easier interpretation

        # Using a colormap that highlights peaks more clearly
        cax = ax.imshow(
            accumulator,
            cmap="hot",
            aspect="auto",
            extent=[angle_degrees[0], angle_degrees[-1], distances[-1], distances[0]],
        )

        # Adding color bar for better understanding of intensity
        fig.colorbar(cax, ax=ax, label="Accumulator Counts")

        ax.set_title("Hough Space")
        ax.set_xlabel("Angles (degrees)")
        ax.set_ylabel("Distances (pixels)")

        # Adding grid for easier value reading
        ax.grid(True, color="blue", linestyle="-.", linewidth=0.5, alpha=0.5)

        self._plot_with_streamlit(fig)

    def visualize_image_and_hough_space(
        self,
        img_gray: np.ndarray,
        accumulator: np.ndarray,
        angles: np.ndarray,
        distances: np.ndarray,
    ) -> NoReturn:
        """Visualizes the original grayscale image and its corresponding Hough space side by side.

        This method aids in the comparative analysis of an image and its Hough transform output. The original image
        is displayed alongside the Hough space, where the latter visualizes the accumulator matrix from the Hough Transform
        as a heatmap. The heatmap intensity represents the number of votes each line received, with brighter colors
        indicating a higher likelihood of line presence. This side-by-side visualization allows for a direct comparison
        between the detected lines in the image space and their representation in the parameter space, facilitating a deeper
        understanding of the Hough Transform process and its effectiveness in line detection.

        Args:
            img_gray (np.ndarray): The original grayscale image as a 2D numpy array.
            accumulator (np.ndarray): The accumulator array from the Hough Transform, indicating
                the number of votes for each angle-distance pair.
            angles (np.ndarray): The array of angles, in radians, used in the Hough Transform.
            distances (np.ndarray): The array of distances used in the Hough Transform.

        Returns:
            NoReturn: This method does not return any value and directly displays the visualization.

        Interpretation:
            - The original image provides a visual context for the lines detected.
            - The Hough space heatmap shows how potential lines are distributed across different angles and distances,
            with peaks indicating a higher consensus on line presence.
            - Comparing these visualizations helps identify how well the Hough Transform has performed in detecting
            lines that correspond to actual features in the original image, and can also reveal the presence of noise
            or artifacts in the detection process.
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Original Image
        axes[0].imshow(img_gray, cmap="gray")
        axes[0].set_title("Original Image")
        axes[0].set_axis_off()

        # Hough Space
        angle_degrees = np.rad2deg(angles)  # Convert angles to degrees
        cax = axes[1].imshow(
            accumulator,
            cmap="hot",
            aspect="auto",
            extent=[angle_degrees[0], angle_degrees[-1], distances[-1], distances[0]],
        )
        axes[1].set_title("Hough Space")
        axes[1].set_xlabel("Angles (degrees)")
        axes[1].set_ylabel("Distances (pixels)")
        fig.colorbar(cax, ax=axes[1], label="Accumulator Counts")
        axes[1].grid(True, color="blue", linestyle="-.", linewidth=0.5, alpha=0.5)

        self._plot_with_streamlit(fig)
