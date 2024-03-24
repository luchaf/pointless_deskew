import os
import tempfile
import time
from typing import List, Tuple

import streamlit as st
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from PIL import Image
from transformers import BertTokenizer


class PointlessDeskewTextAnalyzer:

    def __init__(
        self,
        predictor=ocr_predictor(
            "db_mobilenet_v3_large", "crnn_mobilenet_v3_small", pretrained=True
        ),
        tokenizer=BertTokenizer.from_pretrained("bert-base-multilingual-cased"),
    ):
        self.predictor = predictor
        self.tokenizer = tokenizer

    def normalize_scores(self, scores: dict) -> dict:
        """Normalizes the scores for document orientations to identify the optimal orientation.

        This method normalizes the given scores for different document orientations, facilitating
        a more effective comparison across various orientations. Normalization is achieved by first
        adjusting the scores to a 0 to 1 scale, where each score is inversely proportional to its
        distance from the maximum score. This is done by subtracting each score from the minimum score,
        dividing by the range of scores, and then inverting this value. The scores are then adjusted
        to ensure that their sum equals 1.

        Args:
            scores (dict): A dictionary with keys representing orientation names and values representing
                        the scores associated with each orientation.

        Returns:
            dict: A dictionary containing the normalized scores for each orientation, with values
                adjusted to sum to 1, facilitating direct comparison and interpretation.

        Example:
            Given scores {'0': 10, '90': 15, '180': 5, '270': 20}, the method returns normalized scores
            such that each score is between 0 and 1, and all scores sum to 1.
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
        """Computes a score for a list of words using a BERT tokenizer and word confidences.

        This method evaluates a list of words based on their tokenization by a BERT tokenizer and
        the associated confidence scores for each word. The scoring criteria consider the presence of
        unknown tokens ('[UNK]'), the average length of the subtokens, the proportion of recognized
        subtokens, and the overall confidence in the words. The resulting score aims to estimate the
        quality and meaningfulness of the input content, with lower scores indicating higher quality.

        The scoring formula incorporates penalties for unknown tokens and rewards for longer subtokens
        and a higher proportion of recognized tokens. Additionally, it accounts for the confidence scores
        of the words, penalizing lower-confidence words more heavily. The score is normalized by the number
        of words, and bonuses are applied for a higher number of words and longer average word length,
        encouraging more substantial and confident input.

        Args:
            words (List[str]): The list of words to be evaluated.
            confidences (List[float]): Confidence scores for each word, ranging from 0 to 1.
            tokenizer (BertTokenizer): The BERT tokenizer instance to use for tokenizing words.

        Returns:
            float: The normalized score for the list of words. Lower scores indicate a higher
                likelihood of the content being meaningful and well-tokenized. The score is
                influenced by tokenization quality and word confidence levels.

        Example:
            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            >>> words = ["example", "words", "to", "tokenize"]
            >>> confidences = [0.9, 0.95, 1.0, 0.85]
            >>> score = bert_tokenizer_score_list_of_words(words, confidences, tokenizer)
            >>> print(score)
            The output will be a float representing the normalized score for the input words.
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
        """Analyzes OCR results from multiple document orientations to identify the optimal orientation.

        This method evaluates OCR results from documents in different orientations to determine
        which orientation yields the highest quality of text recognition. It uses a BERT tokenizer
        to tokenize the words in the OCR results and considers the confidence scores of each word
        to compute a score for each document orientation. The orientation with the lowest score,
        indicating the highest quality of OCR results, is identified as the best orientation.

        Each document's score is calculated based on the tokenization metrics and the confidence
        levels of the words, with adjustments made to normalize the scores across different orientations.
        The method returns the index of the best orientation in the input list and a dictionary of
        normalized scores for all orientations, facilitating a comparison of OCR quality across
        orientations.

        Args:
            docs (List[dict]): A list of dictionaries, each representing OCR results for a document
                            in a specific orientation. Each dictionary contains blocks, lines,
                            and words, with each word having a value and a confidence score.
            tokenizer (BertTokenizer): An instance of a BERT tokenizer for tokenizing the words
                                    in the OCR results.

        Returns:
            Tuple[int, dict]: A tuple containing:
                - The index of the best orientation in the input list, based on the analysis of OCR quality.
                - A dictionary with keys as orientation identifiers (e.g., 'orientation 0') and values as
                normalized scores, indicating the relative quality of OCR results for each orientation.

        Example:
            >>> docs = [{...}, {...}, {...}]  # OCR results for different orientations
            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            >>> best_index, adjusted_scores = analyze_ocr_results(docs, tokenizer)
            >>> print(f"Best orientation index: {best_index}")
            >>> for orientation, score in adjusted_scores.items():
            ...     print(f"{orientation}: {score}")
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
        """Crops the center portion of an image vertically to a specified height, maintaining the original width.

        This function identifies the vertical center of the given image and performs a crop operation to
        obtain a segment with the specified height, centered vertically. The width of the image is kept intact.
        If the specified height exceeds the height of the image, no vertical cropping is done, and the original
        image is returned as is.

        Args:
            image (PIL.Image.Image): The image to be cropped, provided as a PIL Image object.
            height (int, optional): The target height for the crop, in pixels. Defaults to 200. The function
                                    ensures that the crop does not exceed the original image's height.

        Returns:
            PIL.Image.Image: A new PIL Image object representing the vertically cropped section of the original image.
                            The cropped image will have the same width as the original and a height equal to the
                            specified `height` parameter, unless the original height is smaller, in which case the
                            original image height is retained.

        Example:
            >>> from PIL import Image
            >>> original_image = Image.open("path/to/your/image.jpg")
            >>> cropped_image = crop_center_vertically(original_image, 150)
            >>> cropped_image.show()
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
        """Estimates the image orientation using OCR and tokenization metrics.

        This function estimates the most probable orientation of an image by rotating it to several predefined angles
        (0, 90, -90, and 180 degrees), applying OCR to each rotated version, and analyzing the OCR results. The best
        orientation is determined based on OCR quality and tokenization metrics. The image is then rotated to this
        estimated best orientation.

        The process involves creating temporary files for each rotated image variant, which are then processed with OCR.
        The OCR results are analyzed to estimate the correct orientation of the image. This method also cleans up the
        temporary files created during the process.

        Args:
            img (Image): The input image for which to estimate the orientation. Expected to be a PIL Image instance.

        Returns:
            Tuple[int, Image, List[dict], dict]: A tuple containing:
                - The estimated rotation angle as an integer (from the set [0, 90, -90, 180]) that likely corrects the image orientation.
                - The rotated image in the estimated correct orientation as a PIL Image.
                - A list of dictionaries, each representing OCR results for the image at each tested orientation.
                - A dictionary with normalized scores for each orientation, aiding in understanding the OCR quality across orientations.

        Note:
            The estimation process includes a step of vertical center cropping on rotated images to focus OCR on the
            central part of the image. This helps improve the speed of orientation estimation.
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
