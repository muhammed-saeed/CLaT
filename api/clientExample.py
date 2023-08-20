import requests


# API endpoint URLs with the ngrok URL
ngrok_url = "https://dae7-134-96-105-142.ngrok-free.app"
english_to_pcm_url = f"{ngrok_url}/translate_english_to_pcm"
pcm_to_english_url = f"{ngrok_url}/translate_pcm_to_english"

# Test sentences
english_sentence = "how are you ? "
pcm_sentence = "how dey you ?"

# Translate English to PCM
english_to_pcm_payload = {"english_sentence": english_sentence}
pcm_translation_response = requests.post(english_to_pcm_url, json=english_to_pcm_payload)
pcm_translation = pcm_translation_response.json().get('pcm_translation', 'Translation error')
print(f"English to PCM: {english_sentence} -> {pcm_translation}")

# Translate PCM to English
pcm_to_english_payload = {"pcm_sentence": pcm_sentence}
english_translation_response = requests.post(pcm_to_english_url, json=pcm_to_english_payload)
english_translation = english_translation_response.json().get('english_translation', 'Translation error')
print(f"PCM to English: {pcm_sentence} -> {english_translation}")
