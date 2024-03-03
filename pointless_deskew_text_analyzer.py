import os
import tempfile
import time
from typing import List, Tuple
import streamlit as st
from doctr.io import DocumentFile
from PIL import Image
from transformers import BertTokenizer


class PointlessDeskewTextAnalyzer:

    def __init__(self, predictor, tokenizer):
        self.predictor = predictor
        self.tokenizer = tokenizer

    def normalize_scores(self, scores):
        """Normalizes scores from multiple document orientations to determine the best orientation.

        Args:
            scores (dict): Scores for each orientation, where keys are orientation names and values are scores.

        Returns:
            dict: Normalized scores for each orientation, with values between 0 and 1.

        The normalization process involves subtracting the minimum score from all scores and then dividing by the range of scores (max - min). This allows for better comparison of results across different orientations.
        """
        min_score = min(scores.values())
        max_score = max(scores.values())
        normalized_scores = {
            key: 1 - ((value - min_score) / (max_score - min_score))
            for key, value in scores.items()
        }
        sum_normalized_scores = sum(normalized_scores.values())
        adjusted_scores = {
            key: value / sum_normalized_scores
            for key, value in normalized_scores.items()
        }
        return adjusted_scores

    def bert_tokenizer_score_list_of_words(
        self, words: List[str], confidences: List[float], tokenizer: BertTokenizer
    ) -> float:
        """
        Computes a score for a list of words based on tokenization metrics using a BERT tokenizer,
        while also considering the confidence of each word and the total number of words.

        Args:
            words (List[str]): The list of words to be scored. Each word is a string.
            confidences (List[float]): The list of confidence scores corresponding to each word.
            tokenizer (BertTokenizer): An instance of BertTokenizer used for tokenizing the words.

        Returns:
            float: An average score for the list of words, where a lower score indicates a higher
                likelihood of the content being meaningful. The score is influenced by the presence
                of unknown tokens, the average length of subtokens, the proportion of known
                subtokens, and the confidence levels of the words.
        """

        if not words:  # Check if the list of words is empty
            return 1000  # Return a high penalty for empty inputs

        total_score = 0  # Initialize total score
        total_confidence_penalty = 0  # Initialize total confidence penalty
        total_word_length = 0

        for word, confidence in zip(
            words, confidences
        ):  # Iterate over each word and its confidence
            tokens = tokenizer.tokenize(word)  # Tokenize the current word

            if not tokens:  # Check if the word could not be tokenized at all
                total_score += 100  # Apply a heavy penalty for untokenizable words
                continue

            # Update total word length
            total_word_length += len(word)

            # Initialize score for the current word
            score = 0
            # Count known tokens, ignoring '[UNK]' which stands for unknown tokens
            known_token_count = sum(1 for token in tokens if token != "[UNK]")
            # Calculate the proportion of known tokens to total tokens
            known_token_proportion = known_token_count / len(tokens) if tokens else 0

            # Calculate average subtoken length, excluding '[UNK]', '[CLS]', and '[SEP]' tokens
            avg_subtoken_length = (
                sum(
                    len(token)
                    for token in tokens
                    if token not in ["[UNK]", "[CLS]", "[SEP]"]
                )
                / len(tokens)
                if tokens
                else 0
            )

            # Apply heuristic adjustments based on tokenization results
            if "[UNK]" in tokens:  # Penalize the presence of unknown tokens
                score += 20
            score += (
                2 - known_token_proportion * 5
            )  # Reward higher proportions of known tokens
            score += 2 - avg_subtoken_length  # Reward longer average subtoken lengths

            if len(tokens) > 1 and known_token_proportion == 1:
                score -= 2  # Lesser penalty for fully known compound words

            # Adjust score based on word confidence
            confidence_penalty = (1 - confidence) * 10  # Scale confidence penalty
            total_confidence_penalty += confidence_penalty

            total_score += (
                score + confidence_penalty
            )  # Update total score with the score for this word

        # Modify scoring to favor more and longer words
        avg_word_length = total_word_length / len(words) if words else 0
        words_score_bonus = len(words) ** 1.5  # Exponential bonus for more words
        length_score_bonus = (
            avg_word_length**2
        )  # Exponential bonus for longer average word length

        # Incorporate bonuses into the average score calculation
        adjusted_score = (
            total_score
            + total_confidence_penalty
            - words_score_bonus
            - length_score_bonus
        ) / max(1, len(words))

        return adjusted_score

    def analyze_ocr_results(
        self, docs: List[dict], tokenizer: BertTokenizer
    ) -> Tuple[int, dict]:
        """
        Analyzes OCR results from multiple document orientations to determine the best orientation.
        Now also considers the confidence of each word in the OCR results.

        Args:
            docs (List[dict]): List of OCR result documents for different orientations.
            tokenizer (BertTokenizer): An instance of BertTokenizer used for tokenizing the words.

        Returns:
            Tuple[int, dict]: The index of the best orientation and the OCR results document for that orientation.
        """
        scores = {}
        best_score = float("inf")
        best_index = -1

        for index, doc in enumerate(docs):
            list_of_words = []
            confidences = []
            for block in doc["blocks"]:
                for line in block["lines"]:
                    for word_info in line["words"]:
                        if len(word_info["value"]) > 1:
                            list_of_words.append(word_info["value"])
                            confidences.append(word_info["confidence"])
            # Adjust the function call to include confidences
            score = self.bert_tokenizer_score_list_of_words(
                list_of_words, confidences, tokenizer
            )
            scores[f"orientation {index}"] = score
            print(f"Score for orientation {index}: {score}")
            if score < best_score:
                best_score = score
                best_index = index

        # Normalize and adjust scores so their sum equals 1
        adjusted_scores = self.normalize_scores(scores)

        # Return both the index of the best orientation and the OCR results for that orientation
        return best_index, adjusted_scores

    def crop_center_vertically(self, image, height=200):
        """
        Crops the center portion of an image vertically to a specified height while maintaining the original width.
        This function calculates the vertical center of the image and crops the image to the specified height from
        this center point. The width of the image remains unchanged.

        Args:
            image (PIL.Image.Image): The image to be cropped. This should be an instance of a PIL Image.
            height (int, optional): The height of the crop in pixels. Defaults to 200 pixels. If the specified height
                is greater than the image height, the original image height is used, resulting in no vertical cropping.

        Returns:
            PIL.Image.Image: A new image object representing the vertically cropped image. This image has the same width
                as the original image and a height as specified by the `height` parameter, unless the original image
                is shorter, in which case the original height is preserved.
        """
        # Get the dimensions of the original image
        img_width, img_height = image.size

        # Calculate the top coordinate to start cropping from, ensuring it's centered vertically
        top = (img_height - height) // 2
        # Calculate the bottom coordinate by adding the desired height to the top coordinate
        bottom = top + height

        # Crop the image from the calculated top to bottom while keeping the full width
        # The crop box is defined as (left, upper, right, lower)
        return image.crop((0, top, img_width, bottom))

    def orientation_rotation_estimation(self, img: Image):
        """
        Estimates the orientation of an image by rotating it to several angles, applying OCR,
        and determining the best orientation based on OCR results and tokenization metrics.

        The function rotates the input image to 0, 90, -90, and 180 degrees, crops the center
        vertically for each rotation, and saves these variants as temporary files. It then
        processes each variant with an OCR predictor, analyzes the OCR results to estimate the
        most probable correct orientation of the image, and finally returns the estimated angle
        and the rotated image in this orientation.

        Args:
            img (Image): The input image to estimate orientation for. This should be a PIL Image instance.

        Returns:
            tuple: A tuple containing the estimated angle (as an integer from the set [0, 90, -90, 180])
                and the rotated image in the estimated correct orientation (as a PIL Image).
        """
        # Define potential rotation angles
        angles = [0, 90, -90, 180]
        temp_files = []  # Store paths to temporary files for OCR processing

        start_time = time.time()
        # Rotate, crop, and save each rotated image variant
        for angle in angles:
            cropped_image = img.rotate(
                angle, expand=True, fillcolor="white"
            )  # Rotate with angle, filling background with white
            cropped_image = self.crop_center_vertically(
                cropped_image, 300
            )  # Crop the rotated image to focus on the center

            # Convert the cropped image to RGB if it's in RGBA mode to avoid the OSError when saving as JPEG
            if cropped_image.mode == "RGBA":
                cropped_image = cropped_image.convert("RGB")

            # Save the cropped image to a temporary file for OCR processing
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".PNG")
            cropped_image.save(temp_file.name)
            temp_files.append(temp_file)

        end_time = time.time()
        print(f"Execution time rotate and save: {end_time - start_time} seconds")

        start_time = time.time()
        # Apply OCR on each cropped and rotated image, storing results
        ocr_results = []
        for temp_file in temp_files:
            page = DocumentFile.from_images(temp_file.name)[0]  # Load image for OCR
            result = self.predictor([page]).export()["pages"][
                0
            ]  # Process image with OCR and get results
            ocr_results.append(result)
        end_time = time.time()
        print(f"Execution time ocr: {end_time - start_time} seconds")

        # Analyze OCR results with tokenizer to find best rotation
        best_index, scores_normalized = self.analyze_ocr_results(
            ocr_results, self.tokenizer
        )

        # Cleanup temporary files
        for temp_file in temp_files:
            temp_file.close()  # Close the file
            os.unlink(temp_file.name)  # Delete the file

        # Determine and print the estimated best angle for the original image
        estimated_angle = angles[best_index]
        estimated_angle = round(estimated_angle, 2)
        print(f"Estimated angle: {estimated_angle}")
        st.write(
            f"Robert aka Rotation Bert corrects Houghs estimage by {estimated_angle}°"
        )

        # Rotate the original image to the estimated best orientation
        rotated_image = img.rotate(estimated_angle, expand=True, fillcolor="white")

        # Return the estimated angle and the rotated image
        return estimated_angle, rotated_image, ocr_results, scores_normalized
