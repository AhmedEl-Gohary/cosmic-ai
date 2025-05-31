from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import logging
from typing import List, Dict, Tuple, Any
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None


def load_model(model_dir: str = "codebert-cfp/best-model"):
    """Load the fine-tuned CodeBERT model and tokenizer"""
    global model, tokenizer, device

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.to(device)
        model.eval()

        logger.info(f"Model loaded successfully from {model_dir}")
        return True

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(traceback.format_exc())
        return False


def is_comment_or_empty(line: str, in_block_comment: bool) -> Tuple[bool, bool]:
    """
    Check if a line is a comment or empty line.
    Returns (should_skip, new_in_block_comment_state)
    """
    stripped = line.strip()

    # Handle block comments
    if in_block_comment:
        if "*/" in stripped:
            return True, False
        else:
            return True, True

    if stripped.startswith("/*"):
        if "*/" in stripped:
            return True, False
        else:
            return True, True

    # Handle single line comments and empty lines
    if not stripped or stripped.startswith("//") or stripped.startswith("#"):
        return True, False

    return False, False


def predict_line_cfp(line: str) -> Dict[str, Any]:
    """
    Predict COSMIC CFP for a single line of code.
    Returns dictionary with individual components and total.
    """
    if model is None or tokenizer is None:
        raise ValueError("Model not loaded")

    try:
        # Tokenize input
        inputs = tokenizer(line, truncation=True, padding=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)

        # Process predictions
        preds = outputs.logits.squeeze().cpu().numpy()

        # If the model outputs multiple values (E, X, R, W), split them
        # Otherwise, assume it's a single CFP value
        if len(preds.shape) == 0:  # Single value
            total_cfp = int(np.round(float(preds)))
            # For single output, we can't separate E, X, R, W
            return {
                'e': 0,
                'x': 0,
                'r': 0,
                'w': 0,
                'total': total_cfp
            }
        elif len(preds) == 4:  # Separate E, X, R, W predictions
            e, x, r, w = [int(np.round(float(p))) for p in preds]
            return {
                'e': e,
                'x': x,
                'r': r,
                'w': w,
                'total': e + x + r + w
            }
        else:  # Single CFP prediction
            total_cfp = int(np.round(np.sum(preds)))
            return {
                'e': 0,
                'x': 0,
                'r': 0,
                'w': 0,
                'total': total_cfp
            }

    except Exception as e:
        logger.error(f"Error predicting CFP for line: {str(e)}")
        return {
            'e': 0,
            'x': 0,
            'r': 0,
            'w': 0,
            'total': 0,
            'error': str(e)
        }


def process_code(code_content: str) -> Dict[str, Any]:
    """
    Process the entire code content and return analysis results.
    """
    lines = code_content.split('\n')
    results = []
    total_pred = 0
    line_count = 0
    in_block = False

    try:
        for i, raw_line in enumerate(lines):
            skip, in_block = is_comment_or_empty(raw_line, in_block)
            if skip:
                continue

            line = raw_line.rstrip("\n")
            line_count += 1

            # Get CFP prediction for this line
            prediction = predict_line_cfp(line)

            line_result = {
                'line_number': line_count,
                'actual_line_number': i + 1,
                'code': line,
                'e': prediction['e'],
                'x': prediction['x'],
                'r': prediction['r'],
                'w': prediction['w'],
                'total_cfp': prediction['total']
            }

            if 'error' in prediction:
                line_result['error'] = prediction['error']

            results.append(line_result)
            total_pred += prediction['total']

        # Calculate summary statistics
        avg_cfp = round(total_pred / line_count, 2) if line_count > 0 else 0
        max_cfp = max([r['total_cfp'] for r in results]) if results else 0

        return {
            'success': True,
            'results': results,
            'summary': {
                'total_lines': line_count,
                'total_cfp': total_pred,
                'average_cfp': avg_cfp,
                'max_cfp': max_cfp
            }
        }

    except Exception as e:
        logger.error(f"Error processing code: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'success': False,
            'error': str(e),
            'results': [],
            'summary': {}
        }


@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')


@app.route('/health')
def health():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not_loaded"
    return jsonify({
        'status': 'healthy',
        'model_status': model_status,
        'device': str(device) if device else 'unknown'
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    # Analyze code and return COSMIC CFP predictions
    try:
        if model is None:
            return jsonify({'success': False,'error':'Model not loaded.'}), 500
        data = request.get_json() or {}
        code = data.get('code','').strip()
        if not code:
            return jsonify({'success': False,'error':'No code provided.'}), 400

        return jsonify(process_code(code))
    except Exception as e:
        logger.error(f"analyze error: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False,'error':f'Server error: {e}'}), 500



@app.route('/model/reload', methods=['POST'])
def reload_model():
    """Reload the model (useful for development)"""
    try:
        data = request.get_json() or {}
        model_dir = data.get('model_dir', 'codebert-cfp/best-model')

        success = load_model(model_dir)

        return jsonify({
            'success': success,
            'message': 'Model reloaded successfully' if success else 'Failed to reload model'
        })

    except Exception as e:
        logger.error(f"Error reloading model: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def initialize_app():
    logger.info("Initializing CosmicScope application...")

    # Try to load the model
    model_dir = os.getenv('MODEL_DIR', 'codebert-cfp/best-model')
    success = load_model(model_dir)

    if not success:
        logger.warning("Failed to load model on startup. The /model/reload endpoint can be used to load it later.")

    logger.info("Application initialized")


if __name__ == '__main__':
    initialize_app()

    # Run the Flask app
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'

    app.run(host='0.0.0.0', port=port, debug=debug)