# Detect logo in single image
python recognizer_terminal.py -r starbucks_logo.png -i test_photo.jpg -o result.jpg

# Process multiple images
python recognizer_terminal.py -r starbucks_logo.png -i photos_folder/ -o results_folder/ --batch

# Adjust sensitivity (stricter detection)
python recognizer_terminal.py -r starbucks_logo.png -i image.jpg --min-matches 15 --ratio-threshold 0.6