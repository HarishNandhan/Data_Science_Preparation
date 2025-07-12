import requests

def generate_completion(messages, max_tokens=1000, temperature=0.7):
    url = "https://api.euron.one/api/v1/euri/alpha/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer euri-2b3c07b523bc64b29052f7abe729232e3e2b6ff8d98eda2165724f56be104e3a"
    }
    payload = {
        "messages": messages,
        "model": "gpt-4.1-nano",
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    response = requests.post(url, headers=headers, json=payload)
    data = response.json()
    print(data)  # For debugging, can be removed later
    try:
        return data['choices'][0]['message']['content']
    except Exception:
        # Always return a string, even on error
        return str(data.get('error', {}).get('message', 'No valid response from API'))