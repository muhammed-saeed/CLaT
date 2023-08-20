import logging
from flask import Flask, request, jsonify
from simpletransformers.t5 import T5Model, T5Args

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

model_args = T5Args()
model_args.max_length = 196
model_args.length_penalty = 1
model_args.num_beams = 10
model_args.evaluation_batch_size = 32

model_output_dir = "/local/musaeed/BESTT5TranslationModel"
model = T5Model("t5", model_output_dir, args=model_args, use_cuda=False)

app = Flask(__name__)

@app.route('/translate_english_to_pcm', methods=['POST'])
def translate_english_to_pcm():
    try:
        data = request.get_json()
        english_sentence = data['english_sentence']
        
        # Perform translation using the T5 model
        pcm_preds = model.predict([f"translate english to pcm: {english_sentence}"])
        pcm_result = pcm_preds[0]
        
        return jsonify({'pcm_translation': pcm_result})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/translate_pcm_to_english', methods=['POST'])
def translate_pcm_to_english():
    try:
        data = request.get_json()
        pcm_sentence = data['pcm_sentence']
        
        # Perform translation using the T5 model
        english_preds = model.predict([f"translate pcm to english: {pcm_sentence}"])
        english_result = english_preds[0]
        
        return jsonify({'english_translation': english_result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
