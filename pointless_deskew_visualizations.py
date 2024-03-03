import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


class PointlessDeskewImageVisualizer:
    def __init__(self, in_streamlit=False):
        """
        Initialize the visualizer.

        Args:
            in_streamlit (bool): Specifies if the visualizer is being run in a Streamlit app.
        """
        self.in_streamlit = in_streamlit

    def visualize_edges(self, edges: np.ndarray):
        """
        Displays the edges detected in an image as a grayscale plot.

        This function takes a 2D numpy array representing the edges detected in an image,
        which is the output of the edge detection algorithm "Canny edge detector",
        and visualizes it using matplotlib. The visualization highlights the edges detected
        by displaying them in white against a black background, facilitating the assessment
        of the edge detection process's effectiveness.

        Parameters:
        - edges (np.ndarray): A 2D numpy array where the value of each pixel represents the
        presence (values greater than 0) or absence (value of 0) of an edge at that pixel.

        Returns:
        - None: This function does not return a value. It displays the visualization directly.

        """
        edges_display = (edges * 255).astype(np.uint8)

        if self.in_streamlit:
            st.image(edges_display, caption="Edges detected", use_column_width=True)
        else:
            plt.imshow(edges_display, cmap="gray")
            plt.title("Edges detected")
            plt.show()

    def visualize_hough_lines(
        self,
        original_image: np.ndarray,
        angle_peaks,
        dist_peaks,
    ):
        """
        Visualizes the original image alongside the image with detected Hough lines superimposed.

        This function takes an image and the results of a Hough transform, including the accumulator,
        the angles, and distances arrays, as well as the peaks in these arrays that correspond to the most
        significant lines detected in the image. It then plots two images side by side: the original image and
        the original image with the detected Hough lines superimposed in red. This is useful for
        understanding the effect of the Hough transform in the line detection task and for verifying the detected
        lines against the original image.

        Parameters:
        - original_image (np.ndarray): The original grayscale image as a 2D numpy array.
        - angle_peaks (np.ndarray): The angles (in radians) corresponding to the most significant lines detected.
        - dist_peaks (np.ndarray): The distances corresponding to the most significant lines detected.

        Returns:
        - None: This function does not return a value. It displays the visualization using matplotlib.
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

        if self.in_streamlit:
            st.pyplot(fig)
        else:
            plt.tight_layout()
            plt.show()

    def visualize_hough_space(
        self, accumulator: np.ndarray, angles: np.ndarray, distances: np.ndarray
    ):
        """
        Enhanced visualization of the Hough space (accumulator space) of the Hough Transform.

        Args:
            accumulator (np.ndarray): The accumulator array from the Hough Transform.
            angles (np.ndarray): The array of angles used in the Hough Transform.
            distances (np.ndarray): The array of distances used in the Hough Transform.
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

        plt.show()

    def visualize_image_and_hough_space(
        self,
        img_gray: np.ndarray,
        accumulator: np.ndarray,
        angles: np.ndarray,
        distances: np.ndarray,
    ):
        """
        Visualize the original image and its corresponding Hough space side by side.

        Args:
            img_gray (np.ndarray): The original grayscale image.
            accumulator (np.ndarray): The accumulator array from the Hough Transform.
            angles (np.ndarray): The array of angles used in the Hough Transform.
            distances (np.ndarray): The array of distances used in the Hough Transform.
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

        plt.tight_layout()
        plt.show()
